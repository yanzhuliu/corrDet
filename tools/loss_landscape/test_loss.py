import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo
import numpy as np
import torch
import h5py
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='plotting loss surface')

    # load mmdetection model args  by lyz
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
             'specify, try to auto resume from the latest checkpoint '
             'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    # direction parameters
    parser.add_argument('--xmin', type=float, default=-1)
    parser.add_argument('--xmax', type=float, default=1)
    parser.add_argument('--xnum', type=int, default=51)
    parser.add_argument('--ymin', type=float, default=-1)
    parser.add_argument('--ymax', type=float, default=1)
    parser.add_argument('--ynum', type=int, default=51)

    # plot parameters
    parser.add_argument('--loss_max', default=5, type=float, help='Maximum value to show in 1D plot')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vmin', default=0.1, type=float, help='Minimum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    parser.add_argument('--show', action='store_true', default=False, help='show plotted figures')
    parser.add_argument('--log', action='store_true', default=False, help='use log scale for loss values')
    parser.add_argument('--plot', action='store_true', default=False, help='plot figures after computation')

    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def plot_2d(runner):
    out_dir = runner.work_dir
    x_min = runner.optim_wrapper.optimizer.x_min
    x_max = runner.optim_wrapper.optimizer.x_max
    x_num = runner.optim_wrapper.optimizer.x_num
    y_min = runner.optim_wrapper.optimizer.y_min
    y_max = runner.optim_wrapper.optimizer.y_max
    y_num = runner.optim_wrapper.optimizer.y_num
    fileName = 'landscape_' + str(x_min) + '_' + str(x_max) + '_' \
                    + str(y_min) + '_' + str(y_max) + '_' \
                    + str(x_num) + 'x' + str(y_num)
    path = osp.join(out_dir, fileName + '.h5')
    h5file = h5py.File(path, 'a')
    x_coords = torch.linspace(x_min, x_max, steps=x_num)
    y_coords = torch.linspace(y_min, y_max, steps=y_num)
    coords = torch.meshgrid(x_coords, y_coords)
    X, Y = coords

    for key in ['loss_cls', 'loss_bbox', 'loss_rpn_cls', 'loss_rpn_bbox']:
        Z = np.array(h5file[key][:]).reshape((len(x_coords)),len(y_coords))

        fig = plt.figure()
        CS = plt.contour(X.cpu().numpy(), Y.cpu().numpy(), Z, cmap='summer')
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(osp.join(out_dir, fileName + '_' + key + '_contour.pdf'), dpi=300, bbox_inches='tight',
                    format='pdf')
        plt.show()

    h5file.close()

def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.plot:
        plot_2d(runner)

    # start testing
    runner.test()

if __name__ == '__main__':
    main()