# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Dict, List
import os
import h5py

from mmengine.hooks import Hook

from mmdet.registry import HOOKS
from matplotlib import pyplot as plt
import numpy as np
import torch
from mmengine.dist import is_main_process

@HOOKS.register_module()
class PlotLossHook(Hook):
    """Check loss together from all batches.
    """

    def __init__(self, out_dir = None, v_min = 0.1, v_max = 10, v_level = 0.5) -> None:
        self.out_dir = out_dir
        self.h5file = None
        self.losses : Dict[str, List[float]] = {}
        self.fileName = ''
        self.v_min = v_min
        self.v_max = v_max
        self.v_level = v_level

    def after_test_epoch(self, runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        if not is_main_process():
            return

        for key in metrics:
            if key in self.losses:
                self.losses[key].append(metrics[key].detach().numpy())
            else:
                self.losses[key] = [metrics[key].detach().numpy()]

    def after_test(self, runner) -> None:
        if not is_main_process():
            return

        if self.out_dir is None:
            self.out_dir = runner.work_dir
        self.x_min = runner.optim_wrapper.optimizer.x_min
        self.x_max = runner.optim_wrapper.optimizer.x_max
        self.x_num = runner.optim_wrapper.optimizer.x_num
        self.y_min = runner.optim_wrapper.optimizer.y_min
        self.y_max = runner.optim_wrapper.optimizer.y_max
        self.y_num = runner.optim_wrapper.optimizer.y_num
        self.fileName = 'landscape_' + str(self.x_min) + '_' + str(self.x_max) + '_'\
                   + str(self.y_min) + '_' + str(self.y_max) + '_' \
                   + str(self.x_num) + 'x' + str(self.y_num)
        path = os.path.join(self.out_dir, self.fileName +'.h5')
        if os.path.isfile(path):
            os.remove(path)
        self.h5file = h5py.File(path, 'a')
        self.x_coords = torch.linspace(self.x_min, self.x_max, steps=self.x_num)
        self.y_coords = torch.linspace(self.y_min, self.y_max, steps=self.y_num)
        self.coords = torch.meshgrid(self.x_coords, self.y_coords)
        self.h5file['x_coords'] = self.x_coords
        self.h5file['y_coords'] = self.y_coords

        X, Y = self.coords

        for key in self.losses:
            Z = np.array(self.losses[key]).reshape((len(self.x_coords),len(self.y_coords)))
            self.h5file[key] = self.losses[key]

            fig = plt.figure()
            CS = plt.contour(X.cpu().numpy(), Y.cpu().numpy(), Z, cmap='summer')
                    #         levels=np.arange(self.v_min, self.v_max, self.v_level))
            plt.clabel(CS, inline=1, fontsize=8)
            fig.savefig(os.path.join(self.out_dir, self.fileName + '_' + key + '_contour.pdf'), dpi=300, bbox_inches='tight', format='pdf')
       #     plt.show()

        self.h5file.close()