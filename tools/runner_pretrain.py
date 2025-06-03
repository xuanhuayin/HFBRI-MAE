import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from sklearn.svm import SVC
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from models.HFBRI_MAE import compute_LRA
from pointnet2_ops import pointnet2_utils

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model

    train_dataloader_svm, test_dataloader_svm = builder.dataset_builder_svm(config.dataset.svm)
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_acc = 0
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)

    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])

        num_iter = 0
        mse = nn.MSELoss()

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids ,data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
                points_ori = points.cuda()
                batch_size = data.shape[0]
                # trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
                trot = Rotate(R=random_rotations(batch_size)).to('cuda')
                points = trot.transform_points(points)
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)
            elif dataset_name == 'ShapeNet_withnormal':
                points = data.cuda()
                points_ori = points.cuda()
                points1= points[:,:,0:3]
                points2 = points[:,:,3:6]
                batch_size = data.shape[0]
                trot = Rotate(R=random_rotations(batch_size)).to('cuda')
                points1 = trot.transform_points(points1)
                points2 = trot.transform_points(points2)
                points[:, :, 0:3] = points1
                points[:, :, 3:6] = points2
            elif dataset_name == 'OmniObject3D':
                # points = data.cuda()
                points = data[0].cuda()
                batch_size = points.shape[0]
                points_ori = points.cuda()
                # label = data[1].cuda()
                trot = Rotate(R=random_rotations(batch_size)).to('cuda')
                points = trot.transform_points(points)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points_ori = train_transforms(points_ori)
            # print(points_ori[0])
            points = train_transforms(points)
            # loss = base_model(points, points_ori)
            targets, output = base_model(points, cutmix=False)
            mse_loss = mse(targets, output)
            loss = mse_loss

            try:
                loss.backward()
                base_model.module.ema_step()
                if config.clip_gradients:
                    norm = builder.clip_gradients(base_model, config.clip_grad)
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")




            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss.item()*1000])
                # losses.update([loss.item()])
            else:
                losses.update([loss.item()*1000])
                # losses.update([loss.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            if idx % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        ## Prepare svm train and RISurConv_attn data

        feats_train_aligned = []
        labels_train_aligned = []
        base_model.eval()

        for i, (data, label) in enumerate(train_dataloader_svm):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            # print(feats.shape)
            for feat in feats:
                feats_train_aligned.append(feat)
            labels_train_aligned += labels

        feats_train_so3 = []
        labels_train_so3 = []

        for i, (data, label) in enumerate(train_dataloader_svm):
            batch_size = data.shape[0]
            # print('batch_size_train', batch_size)
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            trot = Rotate(R=random_rotations(batch_size)).to('cuda')
            data = trot.transform_points(data)
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train_so3.append(feat)
            labels_train_so3 += labels

        feats_train_z = []
        labels_train_z = []
        base_model.eval()

        for i, (data, label) in enumerate(train_dataloader_svm):
            batch_size = data.shape[0]
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            data = trot.transform_points(data)
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train_z.append(feat)
            labels_train_z += labels
            
        feats_test_aligned = []
        labels_test_aligned = []

        for i, (data, label) in enumerate(test_dataloader_svm):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test_aligned.append(feat)
            labels_test_aligned += labels

        feats_test_z = []
        labels_test_z = []

        for i, (data, label) in enumerate(test_dataloader_svm):
            batch_size = data.shape[0]
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            data = trot.transform_points(data)
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test_z.append(feat)
            labels_test_z += labels

        feats_test_so3 = []
        labels_test_so3 = []

        for i, (data, label) in enumerate(test_dataloader_svm):
            batch_size = data.shape[0]
            # print('batch_size_test', batch_size)
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            trot = Rotate(R=random_rotations(batch_size)).to('cuda')
            data = trot.transform_points(data)
            with torch.no_grad():
                feats = base_model(data, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test_so3.append(feat)
            labels_test_so3 += labels

        ### (svm_train: aligned)eval with SVM ###
        feats_train = np.array(feats_train_aligned)
        labels_train = np.array(labels_train_aligned)

        feats_test_aligned = np.array(feats_test_aligned)
        labels_test_aligned = np.array(labels_test_aligned)
        feats_test_z = np.array(feats_test_z)
        labels_test_z = np.array(labels_test_z)
        feats_test_so3 = np.array(feats_test_so3)
        labels_test_so3 = np.array(labels_test_so3)

        model_tl = SVC(C=0.01, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy_aa = model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_az = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_as = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (svm_train: aligned svm_test: aligned): {test_accuracy_aa}",
                  logger=logger)
        print_log(f"Linear Accuracy (svm_train: aligned svm_test: z): {test_accuracy_az}",
                  logger=logger)
        print_log(f"Linear Accuracy (svm_train: aligned svm_test: so3): {test_accuracy_as}",
                  logger=logger)

        ### (svm_train: z)eval with SVM ###
        feats_train = np.array(feats_train_z)
        labels_train = np.array(labels_train_z)

        model_tl = SVC(C=0.01, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy_za = model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_zz = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_zs = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (svm_train: z svm_test: aligned): {test_accuracy_za}", logger=logger)
        print_log(f"Linear Accuracy (svm_train: z svm_test: z): {test_accuracy_zz}",
                  logger=logger)
        print_log(f"Linear Accuracy (svm_train: z svm_test: so3): {test_accuracy_zs}",

                  
        ### (svm_train: so3)eval with SVM ###

        feats_train = np.array(feats_train_so3)
        labels_train = np.array(labels_train_so3)

        model_tl = SVC(C=0.01, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy_sa= model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_sz = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_ss = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (svm_train: so3 svm_test: aligned): {test_accuracy_sa}",
                  logger=logger)
        print_log(f"Linear Accuracy (svm_train: so3 svm_test: z): {test_accuracy_sz}",
                  logger=logger)
        print_log(f"Linear Accuracy (svm_train: so3 svm_test: so3): {test_accuracy_ss}",
                  logger=logger)

        ave_acc = (test_accuracy_sa + test_accuracy_sz + test_accuracy_ss + test_accuracy_aa + test_accuracy_az + test_accuracy_as + test_accuracy_za + test_accuracy_zz + test_accuracy_zs)/9
        if ave_acc > best_acc:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                    logger=logger)
            best_acc = ave_acc
            best_epoch = epoch
        print_log(f"Best ave_acc: {best_acc} at Epoch {best_epoch} ",
                      logger=logger)


    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, noaug=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass
