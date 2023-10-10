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
import umap
#import umap.plot
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import torch
import gzip

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize reduced feature map')
    parser.add_argument(
        '--dir_cls1', default='./output', help='Path to output file')
    parser.add_argument(
        '--dir_cls2', default='./output', help='Path to output file')
    parser.add_argument(
        '--dir_cls3', default='./output', help='Path to output file')
    parser.add_argument(
        '--dir_cls4', default='./output', help='Path to output file')
    parser.add_argument(
        '--stage',
        type=int,
        default=3,
        help='Select a single stage whose feature map will be saved')
    parser.add_argument(
        '--channel',
        type=int,
        default=0,
        help='Select a single channel whose feature map will be saved')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show', action='store_true', help='Show the featmap results')
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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    clean_batch = []
    distorted_batch = []

    files = os.listdir(args.dir_cls1)

    clean_train = [torch.load(gzip.GzipFile(args.dir_cls1 + '/' + f, "rb")) for f in files if os.path.isfile(args.dir_cls1 + '/' + f)]
    clean_train = torch.cat(clean_train,0)
    print(clean_train.shape)

    files = os.listdir(args.dir_cls2)
    clean_test = [torch.load(gzip.GzipFile(args.dir_cls2 + '/' + f, "rb")) for f in files if os.path.isfile(args.dir_cls2 + '/' + f)]
    clean_test = torch.cat(clean_test,0)

    files = os.listdir(args.dir_cls3)
    distorted1_batch = [torch.load(gzip.GzipFile(args.dir_cls3 + '/' + f, "rb")) for f in files if os.path.isfile(args.dir_cls3 + '/' + f)]
    distorted1_batch = torch.cat(distorted1_batch,0)

    files = os.listdir(args.dir_cls4)
    distorted2_batch = [torch.load(gzip.GzipFile(args.dir_cls4 + '/' + f, "rb")) for f in files if os.path.isfile(args.dir_cls4 + '/' + f)]
    distorted2_batch = torch.cat(distorted2_batch,0)

    # Set the position of the map and label on the x-axis
    n_dir_cls = 2

    for j in range(8):
        data = []
        fig, ax = plt.subplots(figsize=(35, 18))
        positions = list(range(0, 6 * n_dir_cls * 32, 12))
        for i in range(32):
            data.append((distorted1_batch[:,8*j+i]-clean_test[:,8*j+i]).tolist())
        #    data.append((distorted2_batch[:, 8 * j + i]-clean_train[:,8*j+i]).tolist())
        violin_parts = plt.violinplot(
            data,
            positions,
            showmeans=True,
            showmedians=True,
            widths=5)
        for vp in violin_parts['bodies']:
            vp.set_facecolor('steelblue')

        data = []
        positions = list(range(6, 6 * n_dir_cls * 32, 12))
        for i in range(32):
            data.append((distorted2_batch[:,8*j+i]-clean_test[:,8*j+i]).tolist())
        #    data.append((distorted2_batch[:, 8 * j + i]-clean_train[:,8*j+i]).tolist())
        violin_parts = plt.violinplot(
            data,
            positions,
            showmeans=True,
            showmedians=True,
            widths=5)
        for vp in violin_parts['bodies']:
            vp.set_facecolor('darkorange')
        plt.axhline(y=0.0, color='r', linestyle='-')

        plt.show()




    data = torch.concat([clean_train, clean_test, distorted_batch],0)
    labels = torch.concat([torch.ones(clean_train.shape[0])*2, torch.ones(clean_test.shape[0]), torch.zeros(distorted_batch.shape[0])],0)

    reducer  = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      n_components=2,
                      metric='correlation').fit(data.cpu())
    embedding = reducer.transform(data.cpu())
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=1)
    plt.show()

    reducer  = umap.UMAP(n_neighbors=15,
                      min_dist=0.3,
                      n_components=3,
                      metric='correlation').fit(data.cpu())
    embedding = reducer.transform(data.cpu())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 1], c=labels, cmap='Spectral', s=1)

 #   plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=1)
 #   plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(sc)
  #  plt.title('UMAP projection of voc feature maps at stage - '+str(args.stage)+' channel - ' +str(args.channel), fontsize=24)
    plt.show()


# Please refer to the usage tutorial:
# https://github.com/open-mmlab/mmyolo/blob/main/docs/zh_cn/user_guides/visualization.md # noqa
if __name__ == '__main__':
    main()
