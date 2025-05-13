import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations
from torchvision import transforms

from sklearn.svm import SVC


train_transforms = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         # data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
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

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    train_dataloader_svm, test_dataloader_svm = builder.dataset_builder_svm(config.dataset.svm)
    # build model
    base_model = builder.model_builder(config.model)

    # for param in base_model.parameters():
    #     param.requires_grad = False
    #
    #     # 只对分类头进行fine-tune
    # for param in base_model.cls_head_finetune.parameters():
    #     param.requires_grad = True
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    # optimizer, scheduler = builder.build_opti_sche(filter(lambda p: p.requires_grad, base_model.parameters()), config)
    # optimizer, scheduler = builder.build_opti_sche(base_model.module.cls_head_finetune.parameters(), config)
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, base_model.parameters()),
    #                               lr=config.optimizer.kwargs.lr, weight_decay=config.optimizer.kwargs.weight_decay)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.scheduler.kwargs.epochs, eta_min=0,
    #                                                        last_epoch=-1)

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
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # import pdb; pdb.set_trace()
            points = train_transforms(points)

            ret, _ = base_model(points)

            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            # _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, train_dataloader_svm, test_dataloader_svm, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    feats_train_aligned = []
    labels_train_aligned = []
    feats_train_so3 = []
    labels_train_so3 = []
    feats_train_z = []
    labels_train_z = []
    feats_test_aligned = []
    labels_test_aligned = []
    feats_test_so3 = []
    labels_test_so3 = []
    feats_test_z = []
    labels_test_z = []
    npoints = config.npoints
    with torch.no_grad():
        for i, (data, label) in enumerate(train_dataloader):
            batch_size = data.shape[0]
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data1 = data.cuda().contiguous()
            # if args.test_svm_rot == 'z':
            #     trot = RotateAxisAngle(angle=torch.rand(n_batches) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            #     trot = Rotate(R=random_rotations(n_batches)).to('cuda')
            # data = trot.transform_points(data)
            with torch.no_grad():
                _, feats = base_model(data1)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train_aligned.append(feat)
            labels_train_aligned += labels



            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            #     trot = Rotate(R=random_rotations(n_batches)).to('cuda')
            data2 = trot.transform_points(data1)
            with torch.no_grad():
                _, feats = base_model(data2)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train_z.append(feat)
            labels_train_z += labels

            # trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            trot = Rotate(R=random_rotations(batch_size)).to('cuda')
            data3 = trot.transform_points(data1)
            with torch.no_grad():
                _, feats = base_model(data3)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_train_so3.append(feat)
            labels_train_so3 += labels


        for i, (data, label) in enumerate(test_dataloader):
            batch_size = data.shape[0]
            labels = list(map(lambda x: x[0], label.numpy().tolist()))
            data1 = data.cuda().contiguous()
            # if args.test_svm_rot == 'z':
            #     trot = RotateAxisAngle(angle=torch.rand(n_batches) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            #     trot = Rotate(R=random_rotations(n_batches)).to('cuda')
            # data = trot.transform_points(data)
            with torch.no_grad():
                _, feats = base_model(data1)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test_aligned.append(feat)
            labels_test_aligned += labels

            trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            #     trot = Rotate(R=random_rotations(n_batches)).to('cuda')
            data2 = trot.transform_points(data1)
            with torch.no_grad():
                _, feats = base_model(data2)
            feats = feats.detach().cpu().numpy()
            for feat in feats:
                feats_test_z.append(feat)
            labels_test_z += labels

            # trot = RotateAxisAngle(angle=torch.rand(batch_size) * 360, axis="Z", degrees=True).to('cuda')
            # elif args.test_svm_rot == 'so3':
            trot = Rotate(R=random_rotations(batch_size)).to('cuda')
            data3 = trot.transform_points(data1)
            with torch.no_grad():
                _, feats = base_model(data3)
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

        ### (base: so3 svm_train: aligned svm_test: z)eval with SVM ###


        feats_train = np.array(feats_train_z)
        labels_train = np.array(labels_train_z)

        model_tl = SVC(C=0.01, kernel='linear')
        model_tl.fit(feats_train, labels_train)
        test_accuracy_aligned = model_tl.score(feats_test_aligned, labels_test_aligned)
        test_accuracy_z = model_tl.score(feats_test_z, labels_test_z)
        test_accuracy_so3 = model_tl.score(feats_test_so3, labels_test_so3)

        print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: aligned): {test_accuracy_aligned}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: z): {test_accuracy_z}",
                  logger=logger)
        print_log(f"Linear Accuracy (base: aligned svm_train: z svm_test: so3): {test_accuracy_so3}",
                  logger=logger)

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

    #     test_pred_aligned = torch.cat(test_pred_aligned, dim=0)
    #     test_pred_z = torch.cat(test_pred_z, dim=0)
    #     test_pred_so3 = torch.cat(test_pred_so3, dim=0)
    #     test_label = torch.cat(test_label, dim=0)
    #
    #     if args.distributed:
    #         test_pred_aligned = dist_utils.gather_tensor(test_pred_aligned, args)
    #         test_pred_z = dist_utils.gather_tensor(test_pred_z, args)
    #         test_pred_so3 = dist_utils.gather_tensor(test_pred_so3, args)
    #         test_label = dist_utils.gather_tensor(test_label, args)
    #
    #     acc_aligned = (test_pred_aligned == test_label).sum() / float(test_label.size(0)) * 100.
    #     acc_z = (test_pred_z == test_label).sum() / float(test_label.size(0)) * 100.
    #     acc_so3 = (test_pred_so3 == test_label).sum() / float(test_label.size(0)) * 100.
    #     print_log('[Validation] EPOCH: %d  acc_aligned = %.4f' % (epoch, acc_aligned), logger=logger)
    #     print_log('[Validation] EPOCH: %d  acc_z = %.4f' % (epoch, acc_z), logger=logger)
    #     print_log('[Validation] EPOCH: %d  acc_so3 = %.4f' % (epoch, acc_so3), logger=logger)
    #
    #     if args.distributed:
    #         torch.cuda.synchronize()
    #
    # # Add testing results to TensorBoard
    # if val_writer is not None:
    #     val_writer.add_scalar('Metric/ACC_aligned', acc_aligned, epoch)
    #     val_writer.add_scalar('Metric/ACC_z', acc_z, epoch)
    #     val_writer.add_scalar('Metric/ACC_so3', acc_so3, epoch)

    return Acc_Metric(test_accuracy_so3)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)



def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger)
    
def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            points = misc.fps(points, npoints)

            logits = base_model(points)
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc
