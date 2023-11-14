from data import *
from data.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from models import *
import os
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from utils.utils import *
from dconv import *
from utils.cfgParser import cfgParser

from optimizer import *

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def train():
    if args.dataset == 'COCO':
        cfg = coco
        image_sets = process_names_coco(args.data_use, type='trainval')[0]
        logger.info(f'training datasets {image_sets}')
        dataset = COCODetectionPair(root=args.dataset_root, image_sets=image_sets,augment_folder=args.augment_folder,
                                transform=SSDAugmentation(cfg['min_dim']))
    elif args.dataset == 'VOC':
        cfg = voc
        image_sets=process_names_voc(args.data_use, type='trainval')[0]
        logger.info(f'training datasets {image_sets}')
        dataset = VOCDetectionPair(root=args.dataset_root, image_sets=image_sets, augment_folder=args.augment_folder,
                               transform=SSDAugmentation(cfg['min_dim']))

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], amp=args.amp and args.multi_gpu, backbone=args.backbone)
    net = ssd_net

    if args.cuda:
        if args.multi_gpu:
            net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        logger.info('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    elif args.basenet!='None':
        vgg_weights = torch.load(args.save_folder + args.basenet)
        logger.info('Loading base network...')
        ssd_net.load_state_dict(vgg_weights)  # by lyz ssd_net.vgg.load_state_dict(vgg_weights) load vgg, now load saved

    if args.cuda:
        net = net.cuda()

    if not args.resume and args.basenet=='None':
        logger.info('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)
        ssd_net.vgg.apply(weights_init)

    if args.optimizer == 'DWP':  # by lyz
        optimizer = DWP(net.parameters(), optim.SGD, rho=args.rho, alpha=args.alpha, adaptive=False, lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    elif args.optimizer == 'RDP':
        optimizer = RDP(net.parameters(), optim.SGD, rho=args.rho, adaptive=False, lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    elif args.optimizer == 'RWP':
        optimizer = RWP(net.parameters(), optim.SGD, rho=args.rho, adaptive=True, lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)

    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5, False, args.cuda, use_focal=args.focal)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    logger.info('Loading the dataset...')
    logger.info(f'Training SSD on: {dataset.name}')
    logger.info('Using the specified args:')
    logger.info(repr(args))

    step_index = 0
    scaler = amp.GradScaler(enabled=args.amp)

    if args.visdom:
        vis_title = 'ssd300 with ' + args.optimizer +' on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    if torch.__version__.startswith('2.0'):  # by lyz 1.9 -> 1.10
        data_loader = data.DataLoader(dataset, args.batch_size // 2,
                                      num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True, drop_last=True, generator=torch.Generator(device='cuda'))
    else:
        data_loader = data.DataLoader(dataset, args.batch_size // 2,
                                      num_workers=args.num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=True, drop_last=True)

    # create batch iterator
    t0 = time.time()
    batch_iterator = BatchPairIter(data_loader, args.cuda)

    loss_sum=0
    for iteration in range(args.start_iter, args.max_iter):
        if iteration in cfg['lr_steps']:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # load train data
        images_aug, images_org, targets = batch_iterator.next()
    #    images_comb = torch.concat([images_aug, images_org])

        # forward
        optimizer.zero_grad()
        if args.optimizer != 'SGD':
            enable_running_stats(net)

        with amp.autocast() if args.amp else Empty():
            if args.adv_rdp:
                delta_x = images_aug - images_org
                e_x = torch.rand(images_org.shape) * 0.5 * delta_x
                x_adv = images_org + e_x
                inputs = torch.concat([images_org, x_adv])
                gt = targets + targets
                out = net(inputs)
                loss_l, loss_c = criterion(out, gt)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.second_step(zero_grad=True)
            elif args.mtl:
                delta_x = images_aug - images_org
                e_x = torch.rand(images_org.shape) * delta_x
                x_adv = images_org + e_x
                inputs = torch.concat([images_org, images_aug, x_adv])
                gt = targets + targets + targets

                out = net(images_org)
                loss_l, loss_c = criterion(out, targets)
                loss_l.backward(retain_graph=True)
                optimizer.grad_x0_lossl(zero_grad=True)
                loss_c.backward()
                optimizer.grad_x0_lossc(zero_grad=True)

                out = net(images_aug)
                loss_l, loss_c = criterion(out, targets)
                loss_l.backward(retain_graph=True)
                optimizer.grad_x1_lossl(zero_grad=True)
                loss_c.backward()
                optimizer.grad_x1_lossc(zero_grad=True)

                disable_running_stats(net)
                optimizer.first_step_l(zero_grad=True)
                out = net(inputs)
                loss_l, loss_c = criterion(out, gt)
                loss_d_l = loss_l + loss_c
                loss_d_l.backward()

                optimizer.first_step_c(zero_grad=True)
                out = net(inputs)
                loss_l, loss_c = criterion(out, gt)
                loss_d_c = loss_l + loss_c

                if loss_d_c > loss_d_l:
                    loss = loss_d_c
                    loss_d_c.backward()
                    optimizer.second_step(zero_grad=True)
                else:
                    loss = loss_d_l
                    optimizer.second_step(zero_grad=True)

            else:
                images_org.requires_grad = True
                out = net(images_org)
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.pre_step(zero_grad=True)

                images_aug.requires_grad = True
                out = net(images_aug)
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.first_step(zero_grad=True)

                delta_x = images_aug - images_org
                e_x = torch.rand(images_org.shape) * 0.5 * delta_x
                x_adv = images_org + e_x

                disable_running_stats(net)
            #    out = net(torch.concat([images_org,images_aug,x_adv]))
                out = net(torch.concat([images_org, x_adv]))
                loss_l, loss_c = criterion(out, targets+targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.second_step(zero_grad=True)

        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        loss_sum += loss.item()

        if iteration % 50 == 0:
            t1 = time.time()
            logger.info('iter ' + repr(iteration) + ' || Loss: %.4f || timer: %.4f sec.' % (loss_sum/50, (t1 - t0)/50))
            loss_sum=0
            t0 = time.time()

        if args.visdom:
            update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                            iter_plot, 'append')

        if iteration != 0 and iteration % 5000 == 0:
            logger.info(f'Saving state, iter: {iteration}')
            torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_{args.optimizer}_{iteration}.pth'))

    torch.save(ssd_net.state_dict(), os.path.join(args.save_folder, f'ssd300_{task_name}_{args.optimizer}_final_{args.max_iter}.pth'))

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, DynamicConv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def weights_init_prob(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, DynamicConv2d):
        init.xavier_uniform_(m.weight, gain=1./256)
        if m.bias is not None:
            init.zeros_(m.bias)

def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )

def update_vis_plot(iteration, loc, conf, window, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window,
        update=update_type
    )

if __name__ == '__main__':
    cfgp = cfgParser(base_block=['model', 'data'])
    args=cfgp.load_cfg(['train'])

    if not os.path.exists(args.log_folder):
        os.mkdir(args.log_folder)

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    suffix_dict = {'_ft': args.resume, '_nobase': args.basenet == 'None', '_resnet': args.backbone == 'resnet', '_mtl':args.mtl, '_adv_rdp':args.adv_rdp}
    task_name = f'{args.dataset}_dwp-rho{args.rho}-alpha{args.alpha}' \
                f'_{args.data_use}{"".join(k for k, v in suffix_dict.items() if v)}'
    logger = get_logger(os.path.join(args.log_folder, f'train_{task_name}.log'))

    logger.info(repr(args))

    if args.visdom:
        import visdom

        viz = visdom.Visdom()

    if args.amp:
        from torch.cuda import amp

        logger.info('using amp')

    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            logger.warning("WARNING: It looks like you have a CUDA device, but aren't " +
                           "using CUDA.\nRun with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    train()
