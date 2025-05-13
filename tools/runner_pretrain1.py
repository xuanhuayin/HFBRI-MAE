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
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudRotatePerturbation(),
        # data_transforms.PointcloudTranslate(),
        # data_transforms.PointcloudJitter(),
        # data_transforms.PointcloudRandomInputDropout(),
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

        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
                norm = compute_LRA(points)
                batch_size = data.shape[0]
                # trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
                trot = Rotate(R=random_rotations(batch_size)).to('cuda')
                points = trot.transform_points(points)
                norm = trot.transform_points(norm)
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                points = misc.fps(points, npoints)   
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            assert points.size(1) == npoints
            points = train_transforms(points)
            norm = train_transforms(norm)
            loss = base_model(points, norm)
            try:
                loss.backward()
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
            else:
                losses.update([loss.item()*1000])


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

        # if epoch % args.val_freq == 0 and epoch != 0:
        #     # Validate the current model
        #     metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)
        #
        #     # Save ckeckpoints
        #     if metrics.better_than(best_metrics):
        #         best_metrics = metrics
        #         builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        if epoch % 25 ==0 and epoch >=250:
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
                                    logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
        ## Prepare svm train and RISurConv_attn data

        feats_train_aligned = []
        labels_train_aligned = []
        base_model.eval()

        for i, (data, label) in enumerate(train_dataloader_svm):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            norm = compute_LRA(data)

            # if args.train_svm_rot == 'z':
            #     trot = RotateAxisAngle(angle=torch.rand(n_batches) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.train_svm_rot == 'so3':
            #     trot = Rotate(R=random_rotations(n_batches)).to('cuda')
            # data = trot.transform_points(data)
            with torch.no_grad():
                feats = base_model(data, norm, eval=True)
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
            norm = compute_LRA(data)
            trot = Rotate(R=random_rotations(batch_size)).to('cuda')
            data = trot.transform_points(data)
            norm = trot.transform_points(norm)
            with torch.no_grad():
                feats = base_model(data, norm, eval=True)
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
            norm = compute_LRA(data)
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            data = trot.transform_points(data)
            norm = trot.transform_points(norm)
            with torch.no_grad():
                feats = base_model(data, norm, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train_z.append(feat)
            labels_train_z += labels
        #
        feats_test_aligned = []
        labels_test_aligned = []

        for i, (data, label) in enumerate(test_dataloader_svm):
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data = data.cuda().contiguous()
            norm = compute_LRA(data)
            # if args.test_svm_rot == 'z':
            #     trot = RotateAxisAngle(angle=torch.rand(n_batches) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            #     trot = Rotate(R=random_rotations(n_batches)).to('cuda')
            # data = trot.transform_points(data)
            with torch.no_grad():
                feats = base_model(data, norm, eval=True)
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
            norm = compute_LRA(data)
            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            data = trot.transform_points(data)
            norm = trot.transform_points(norm)
            with torch.no_grad():
                feats = base_model(data, norm, eval=True)
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
            norm = compute_LRA(data)
            trot = Rotate(R=random_rotations(batch_size)).to('cuda')
            data = trot.transform_points(data)
            norm = trot.transform_points(norm)
            with torch.no_grad():
                feats = base_model(data, norm, eval=True)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test_so3.append(feat)
            labels_test_so3 += labels

        ### (base: so3 svm_train: aligned svm_test: aligned)eval with SVM ###
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
        test_accuracy_aligned = model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_z = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_so3 = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (base: aligned svm_train: aligned svm_test: aligned): {test_accuracy_aligned}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: aligned svm_test: z): {test_accuracy_z}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: aligned svm_test: so3): {test_accuracy_so3}",
                  logger=logger)
        # if test_accuracy_aligned > best_accuracy:
        #     best_accuracy = test_accuracy_so3
        #     print_log(f"(base: aligned svm_train: aligend svm_test: so3)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: aligned svm_test: so3)ckpt-best', args,
        #                             logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: aligned svm_test: so3)ckpt-last', args, logger=logger)

        #
        ### (base: so3 svm_train: aligned svm_test: z)eval with SVM ###

        # feats_test = np.array(feats_test_z)
        # labels_test = np.array(labels_test_z)
        feats_train = np.array(feats_train_z)
        labels_train = np.array(labels_train_z)

        model_tl = SVC(C=0.01, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy_aligned = model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_z = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_so3 = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: aligned): {test_accuracy_aligned}", logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: z): {test_accuracy_z}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: so3): {test_accuracy_so3}",
                  logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: aligned svm_test: z)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: aligned svm_test: z)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: aligned svm_test: z)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: aligned svm_test: z)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        ### (base: so3 svm_train: aligned svm_test: so3)eval with SVM ###

        feats_train = np.array(feats_train_so3)
        labels_train = np.array(labels_train_so3)

        model_tl = SVC(C=0.01, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy_aligned = model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_z = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_so3 = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (base: aligned svm_train: so3 svm_test: aligned): {test_accuracy_aligned}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: so3 svm_test: z): {test_accuracy_z}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: so3 svm_test: so3): {test_accuracy_so3}",
                  logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: aligned svm_test: so3)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: aligned svm_test: z)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: aligned svm_test: so3)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: aligned svm_test: so3)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        # ### (base: so3 svm_train: z svm_test: aligned) eval with SVM ###
        #
        # feats_train = np.array(feats_train_z)
        # labels_train = np.array(labels_train_z)
        #
        # feats_test = np.array(feats_test_aligned)
        # labels_test = np.array(labels_test_aligned)
        #
        # model_tl = SVC(C=0.01, kernel='linear')
        # model_tl.fit(feats_train, labels_train)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        #
        # print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: aligned): {test_accuracy}", logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: aligned svm_test: aligned)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: z svm_test: aligned)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: z svm_test: aligned)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: z svm_test: aligned)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        # ### (base: z svm_train: z svm_test: z) eval with SVM ###
        #
        # feats_test = np.array(feats_test_z)
        # labels_test = np.array(labels_test_z)
        #
        # # model_tl = SVC(C=0.01, kernel='linear')
        # # model_tl.fit(feats_train, labels_train)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        #
        # print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: z): {test_accuracy}", logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: z svm_test: z)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: z svm_test: z)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: z svm_test: z)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: z svm_test: z)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        # ### (base: so3 svm_train: z svm_test: so3) eval with SVM ###
        #
        # feats_test = np.array(feats_test_so3)
        # labels_test = np.array(labels_test_so3)
        #
        # # model_tl = SVC(C=0.01, kernel='linear')
        # # model_tl.fit(feats_train, labels_train)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        #
        # print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: so3): {test_accuracy}", logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: z svm_test: so3)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: z svm_test: so3)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: z svm_test: so3)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: z svm_test: so3)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        # ### (base: so3 svm_train: so3 svm_test: aligned) eval with SVM ###
        # feats_train = np.array(feats_train_so3)
        # labels_train = np.array(labels_train_so3)
        #
        # feats_test = np.array(feats_test_aligned)
        # labels_test = np.array(labels_test_aligned)
        #
        # model_tl = SVC(C=0.01, kernel='linear')
        # model_tl.fit(feats_train, labels_train)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        #
        # print_log(f"Linear Accuracy (base: aligned svm_train: so3 svm_test: aligned): {test_accuracy}", logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: so3 svm_test: aligned)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: so3 svm_test: aligned)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: so3 svm_test: aligned)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: so3 svm_test: aligned)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        # ### (base: so3 svm_train: so3 svm_test: z) eval with SVM ###
        #
        # feats_test = np.array(feats_test_z)
        # labels_test = np.array(labels_test_z)
        #
        # # model_tl = SVC(C=0.01, kernel='linear')
        # # model_tl.fit(feats_train, labels_train)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        #
        # print_log(f"Linear Accuracy (base: aligned svm_train: so3 svm_test: z): {test_accuracy}", logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: so3 svm_test: z)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: so3 svm_test: z)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: so3 svm_test: z)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: so3 svm_test: z)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
        #
        # ### (base: so3 svm_train: so3 svm_test: so3) eval with SVM ###
        #
        # feats_test = np.array(feats_test_so3)
        # labels_test = np.array(labels_test_so3)
        #
        # # model_tl = SVC(C=0.01, kernel='linear')
        # # model_tl.fit(feats_train, labels_train)
        # test_accuracy = model_tl.score(feats_test, labels_test)
        #
        # print_log(f"Linear Accuracy (base: aligned svm_train: so3 svm_test: so3): {test_accuracy}", logger=logger)
        # if test_accuracy > best_accuracy:
        #     best_accuracy = test_accuracy
        #     print_log(f"(base: aligned svm_train: so3 svm_test: so3)Saving best...", logger=logger)
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                             '(base: aligned svm_train: so3 svm_test: so3)ckpt-best', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         '(base: aligned svm_train: so3 svm_test: so3)ckpt-last', args, logger=logger)
        #
        # builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics,
        #                         f'(base: aligned svm_train: so3 svm_test: so3)ckpt-epoch-{epoch:03d}', args,
        #                         logger=logger)
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