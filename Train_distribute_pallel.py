
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torchvision
from torchvision import transforms
import datetime
import time
import torch
import torch.nn as nn
import math
import argparse
from torch.cuda import amp
import torch.distributed.optim
from torch.utils.tensorboard import SummaryWriter
import functions.utils as utils
from functions import loss_functions
from functions.lookahead import Lookahead
import data.data_loaders as data_loaders

# model import
from models.MS_ResNet import *
from models.ResNet import *
from models.ResNet19 import ResNet_SB19



def train(train_loader, model, criterion, optimizer, device, epoch, args, scaler=None):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    progress = utils.ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses, top1, top5],
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()
    for i, (image, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.dataset == 'cifardvs':
            # image B, T, C, H, W -> T, B, C, H, W
            image = image.permute(1, 0, 2, 3, 4)
        image, target = image.to(device), target.to(device)

        if scaler is not None:
            with amp.autocast():
                output = model(image)
                loss = criterion(output, target)
        else:
            output = model(image)
            loss = criterion(output, target)


        # only compute the final output's accuracy
        if isinstance(output, list):
            output = output[0]

        if len(output.shape) == 3:
            output = output.mean(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]

        torch.distributed.barrier()

        reduced_loss = utils.reduce_mean(loss, args.nprocs)
        reduced_acc1 = utils.reduce_mean(acc1, args.nprocs)
        reduced_acc5 = utils.reduce_mean(acc5, args.nprocs)

        losses.update(reduced_loss.item(), batch_size)
        top1.update(reduced_acc1.item(), batch_size)
        top5.update(reduced_acc5.item(), batch_size)

        optimizer.zero_grad()

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            for m in model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    if m.weight.grad is None:
                        print(m)
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, top1.avg, top5.avg


def evaluate(val_loader, model, criterion, device, args):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')
    # create num_sb number of utils.AverageMeter
    auxi_top1 = None
    if args.num_sb > 0:
        auxi_top1 = [utils.AverageMeter('Acc@1', ':6.2f') for _ in range(args.num_sb)]
    progress = utils.ProgressMeter(len(val_loader), [batch_time, losses, top1, top5],
                             prefix='Test: ')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (image, target) in enumerate(val_loader):
            if args.dataset == 'cifardvs':
                # image B, T, C, H, W -> T, B, C, H, W
                image = image.permute(1, 0, 2, 3, 4)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            if isinstance(output, list):
                auxi_out = output[1:]
                output_final = output[0]
            else:
                output_final = output

            if len(output_final.shape) == 3:
                output_final = output_final.mean(0)

            acc1, acc5 = utils.accuracy(output_final, target, topk=(1, 5))
            batch_size = image.shape[0]

            torch.distributed.barrier()

            reduced_loss = utils.reduce_mean(loss, args.nprocs)
            reduced_acc1 = utils.reduce_mean(acc1, args.nprocs)
            reduced_acc5 = utils.reduce_mean(acc5, args.nprocs)

            losses.update(reduced_loss.item(), batch_size)
            top1.update(reduced_acc1.item(), batch_size)
            top5.update(reduced_acc5.item(), batch_size)

            # auxi accuracy
            if isinstance(output, list):
                for j in range(len(auxi_out)):
                    auxi_acc1, auxi_acc5 = utils.accuracy(auxi_out[j], target, topk=(1, 5))
                    reduced_auxi_acc1 = utils.reduce_mean(auxi_acc1, args.nprocs)
                    auxi_top1[j].update(reduced_auxi_acc1.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1,
                                                                    top5=top5))
        if isinstance(output, list) and auxi_top1 is not None:
            for j in range(len(auxi_top1)):
                print(' * Auxi_Acc@1 {top1.avg:.3f}'.format(top1=auxi_top1[j]))
    return losses.avg, top1.avg, top5.avg

def load_data_cifar(use_cifar10=True,download=True, distributed=False, cutout=False, autoaug=False):
    train_dataset, test_dataset = data_loaders.build_cifar(cutout=cutout, autoaug=autoaug, 
                          use_cifar10=use_cifar10,download=download)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler

def load_data_imagenet(distributed=False):
    train_dataset, test_dataset = data_loaders.build_imagenet()
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler

def load_data_cifardvs(distributed=False):
    path = '/data_smr/dataset/cifar-dvs/'
    train_dataset, test_dataset = data_loaders.build_dvscifar(path)
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    return train_dataset, test_dataset, train_sampler, test_sampler

def _get_cache_path(filepath):
    import hashlib
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_train from {}".format(cache_path))
        dataset, _ = torch.load(cache_path)
    else:
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_train to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading dataset_test from {}".format(cache_path))
        dataset_test, _ = torch.load(cache_path)
    else:
        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving dataset_test to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler

def main(args, model, criterion):

    args.nprocs = torch.cuda.device_count()

    max_test_acc1 = 0.
    test_acc5_at_max_test_acc1 = 0.


    train_tb_writer = None
    te_tb_writer = None

    utils.init_distributed_mode(args)
    # print(args)


    device = torch.device(args.device)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        dataset_train, dataset_test, train_sampler, test_sampler = load_data_cifar(use_cifar10=args.dataset == 'cifar10',download=True, distributed=args.distributed, cutout=args.cutout, autoaug=args.autoaug)

    elif args.dataset == 'cifardvs':
        dataset_train, dataset_test, train_sampler, test_sampler = load_data_cifardvs(distributed=args.distributed)

    elif args.dataset =='imagenet':
        # dataset_train, dataset_test, train_sampler, test_sampler = load_data_imagenet(distributed=args.distributed)
        train_dir = os.path.join(args.data_path, 'train')
        val_dir = os.path.join(args.data_path, 'val')
        dataset_train, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir,
                                                                             args.cache_dataset, args.distributed)

    print(f'dataset_train:{dataset_train.__len__()}, dataset_test:{dataset_test.__len__()}')
    print(dataset_train.__getitem__(0)[0].shape)
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size,
        sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    print("Creating model")
    model = model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # optimizer = Lookahead(optimizer, k=5, alpha=0.5)


    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None


    if args.cos_lr_T > 0:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_lr_T)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])  # , output_device=args.gpu)
        model_without_ddp = model.module

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        args.start_epoch = checkpoint['epoch'] + 1

        max_test_acc1 = checkpoint['max_test_acc1']
        test_acc5_at_max_test_acc1 = checkpoint['test_acc5_at_max_test_acc1']

    if args.test_only:
        evaluate(data_loader_test, model, criterion, device, args)
        return

    # praper logger
    utils.mkdir(args.output_dir)
    output_dir = os.path.join(args.output_dir, f'{args.model}_{args.dataset}_T{args.T}')
    utils.mkdir(output_dir)
    gpu_num = torch.cuda.device_count()
    output_dir = os.path.join(output_dir,
                              f'b{args.batch_size * gpu_num}_opt{args.optimizer}_lr{args.lr}_wd{args.weight_decay}_epochs{args.epochs}')
    utils.mkdir(output_dir)

    output_dir = os.path.join(output_dir, f'operation_{args.operation}')
    utils.mkdir(output_dir)

    logger = utils.get_logger(output_dir + '/training_log.log')
    logger.parent = None

    if args.tb and utils.is_main_process():
        purge_step_train = args.start_epoch
        purge_step_te = args.start_epoch
        train_tb_writer = SummaryWriter(output_dir + '_logs/train', purge_step=purge_step_train)
        te_tb_writer = SummaryWriter(output_dir + '_logs/te', purge_step=purge_step_te)
        with open(output_dir + '_logs/args.txt', 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print(f'purge_step_train={purge_step_train}, purge_step_te={purge_step_te}')

    if utils.is_main_process():
        logger.info(output_dir)
        logger.info(args)
        logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        save_max = False
        if args.distributed:
            train_sampler.set_epoch(epoch)
            train_loss, train_acc1, train_acc5 = train(data_loader, model, criterion, optimizer, device, epoch, args,scaler=scaler)

            if utils.is_main_process():
                logger.info('Train Epoch:[{}/{}]\t loss={:.5f}\t top1 acc={:.3f}\t top5 acc={:.3f}\t'
                            .format(epoch, args.epochs, train_loss, train_acc1, train_acc5))
                if args.tb:
                    train_tb_writer.add_scalar('train_loss', train_loss, epoch)
                    train_tb_writer.add_scalar('train_acc1', train_acc1, epoch)
                    train_tb_writer.add_scalar('train_acc5', train_acc5, epoch)
            lr_scheduler.step()

            test_loss, test_acc1, test_acc5 = evaluate(data_loader_test, model, criterion, device, args)

            if utils.is_main_process():
                logger.info('Test Epoch:[{}/{}]\t loss={:.5f}\t top1 acc={:.3f}\t top5 acc={:.3f}\t'
                            .format(epoch, args.epochs, test_loss, test_acc1, test_acc5))

            if te_tb_writer is not None:
                if args.tb and utils.is_main_process():
                    te_tb_writer.add_scalar('test_loss', test_loss, epoch)
                    te_tb_writer.add_scalar('test_acc1', test_acc1, epoch)
                    te_tb_writer.add_scalar('test_acc5', test_acc5, epoch)

            if max_test_acc1 < test_acc1:
                max_test_acc1 = test_acc1
                test_acc5_at_max_test_acc1 = test_acc5
                save_max = True

            if output_dir:
                checkpoint = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'max_test_acc1': max_test_acc1,
                    'test_acc5_at_max_test_acc1': test_acc5_at_max_test_acc1,
                }

                utils.save_on_master(
                    checkpoint,
                    os.path.join(output_dir, 'checkpoint_latest.pth'))
                save_flag = False

                if save_flag:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir, f'checkpoint_{epoch}.pth'))

                if save_max:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(output_dir, 'checkpoint_max_test_acc1.pth'))
            # print(args)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # print(output_dir)
            if utils.is_main_process():
                logger.info('Training time {}\t max_test_acc1: {} \t test_acc5_at_max_test_acc1: {}'
                            .format(total_time_str, max_test_acc1, test_acc5_at_max_test_acc1))
                logger.info('\n')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data-dir', default='./data/', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--model', default='resnet18', type=str)
    # dataset has cifar10, cifar100, cifardvs and imagenet
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100', 'cifardvs','imagenet'])
    parser.add_argument('--optimizer', default='adamw',type=str)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--T', default=1, type=int, help='simulation time')
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=0.02, type=float)
    parser.add_argument('--print-freq', default=100, type=int)
    parser.add_argument('--output-dir', default='./outputs/', type=str)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--num_sb', default=0, type=int)
    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--sync-bn', action='store_true')
    parser.add_argument('--tb', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--autoaug', action='store_true')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--cos_lr_T', default=0, type=int, help='cosine annealing lr')
    parser.add_argument('--zero_init_residual', action='store_true')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    '''
    python -m torch.distributed.launch --nproc_per_node=2 --use_env Train_distribute_pallel.py \
    --batch-size 128 --cos_lr_T 300 --epochs 300 \
    --model ResNet_SB18 \
    --num_classes 100 --dataset cifar100 --T 2 \
    --sync-bn --optimizer adamw --lr 0.01 --weight-decay 0.02
    '''
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    utils.seed_all(1000, benchmark=False)
    args = parse_args()

    criterion = nn.CrossEntropyLoss()
    # criterion = loss_functions.TET_loss_with_MMD(criterion, lamb=0.001, means=1.0)

    # if model name contains 'SB', then use surrogate module
    if 'SB' in args.model:
        session_name = 'sml' # 'sdt' 'sml'
    else:
        session_name = 'sdt'

    if session_name == 'sdt':
        args.operation = 'sdt_snn_autoaug'
        model = ResNet18(T=args.T, num_classes=args.num_classes, zero_init_residual=args.zero_init_residual)
        # ## model.change_activation(True) # change the activation function to ReLU
        if args.dataset == 'cifardvs':
            model.conv1=nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    else:

        ### surrogate module parameters

        # surrogate module insert position (e.g. resnet18 with num_blocks [2,2,2,2] has 8 insert places)
        sb_poi = [3, 3] # [3, 3] means insert surrogete module after the 3-rd and 6-th basicblocks; [2, 1, sm, 1, 2, sm, 2]
        # the kernel_size of surrogte module, kernel_size 7 and 5 build a conv2d with stride 2 for downsample.
        # You can add more 3 to add additional conv2d(panel, panel, 3, 1, 1) to increase the surrogate module depth.
        kernels = [[7,5,3], [7,5,3]] 
        args.num_sb = len(sb_poi)
        # surrogate module channel number
        sb_pads = 256

        # surrogate module training loss parameters
        # L_{total} = \frac{(1-\lamb)}{1+N\alpha }\cdot(L_{snn}+\alpha\sum_i^N L_i)
        # +\frac{(\lamb)}{2N}(KL(q_{snn},q_i)+KL(q_i,q_{snn}))
        temperature = 3.0
        alpha = 0.5
        lamb = 0.9

        # add the SB operation parameter here
        pois = ''.join(str(x)+'a' for x in sb_poi)

        name_kernels = ''
        for ker in kernels:
            name_kernels += ''.join(str(x) for x in ker)
            name_kernels += 'a'
        args.operation = 'SB_poiL' + pois[:-1] + '_pads'+str(sb_pads)+'_k'+name_kernels+\
                         '_tp'+str(temperature)+'_alpha'+str(alpha)+'_lamb'+str(lamb)

        args.operation += 'snn_autoaug'  # add your special operation here
        model = ResNet_SB18(num_classes=args.num_classes, sb_kernels=kernels, sb_places=sb_poi, sb_pads=sb_pads, T=args.T)
        model.use_detach = False
        # model.change_activation(True) # change the activation function to ReLU
        if args.dataset == 'cifardvs':
            model.conv1=nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)
        criterion = loss_functions.SBDistillationLoss(criterion=criterion, temperature=temperature, alpha=alpha, lamb=lamb)



    main(args, model, criterion)

