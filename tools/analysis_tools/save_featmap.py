# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from typing import Sequence

import mmcv
from mmdet.apis import inference_detector, init_detector
from mmengine import Config, DictAction
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.registry import VISUALIZERS
from mmdet.utils.misc import auto_arrange_images, get_file_list
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.runner import Runner
import torch
import gzip

def parse_args():
    parser = argparse.ArgumentParser(description='Save feature map')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--target-layers',
        default=['backbone'],
        nargs='+',
        type=str,
        help='The target layers to get feature map, if not set, the tool will '
        'specify the backbone')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--stage',
        type=int,
        default=1,
        help='Select a single stage whose feature map will be saved')
    parser.add_argument(
        '--channel',
        type=int,
        default=0,
        help='Select a single channel whose feature map will be saved')
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
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class ActivationsWrapper:

    def __init__(self, model, target_layers):
        self.model = model
        self.activations = []
        self.handles = []
        self.image = None
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def __call__(self, batch):
        self.activations = []
        with torch.no_grad():
            results = self.model.test_step(batch)
        return results, self.activations[0]

    def release(self):
        for handle in self.handles:
            handle.remove()


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

 #   init_default_scope(cfg.get('default_scope', 'mmyolo'))
    model = init_detector(args.config, args.checkpoint, device=args.device)
    data_loader = Runner.build_dataloader(cfg.val_dataloader)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    target_layers = []
    for target_layer in args.target_layers:
        try:
            target_layers.append(eval(f'model.{target_layer}'))
        except Exception as e:
            print(model)
            raise RuntimeError('layer does not exist', e)

    channel_selected = args.channel
    stage_selected = args.stage
    activations_wrapper = ActivationsWrapper(model, target_layers)

    progress_bar = ProgressBar(len(data_loader))
    for idx, data_batch in enumerate(data_loader):
     #   if idx>=60:
     #       break
        # featmaps [stages][bz, channel, h, w]
        result, featmaps = activations_wrapper(data_batch)
        featmap_selected = featmaps[stage_selected].flatten(start_dim=-2).mean(-1)

        out_file = os.path.join(args.out_dir, 'batch_'+str(idx)+'.pt.gz')
        progress_bar.update()
        torch.save(featmap_selected, gzip.GzipFile(out_file, "wb"))

    print(f'All done!'
              f'\nResults have been saved at {os.path.abspath(args.out_dir)}')


# Please refer to the usage tutorial:
# https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/visualization.md # noqa
if __name__ == '__main__':
    main()
