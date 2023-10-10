# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls


@METRICS.register_module()
class CollectLossMetric(BaseMetric):
    """Collect loss from each batch and get the final average loss
    """
   # default_prefix: Optional[str] = 'collect_loss_landscape'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        self.results.append(data_samples)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        eval_results = OrderedDict()
        loss_rpn_cls, loss_rpn_bbox, loss_cls, loss_bbox = 0, 0, 0, 0
        for batch in results:
            loss_rpn_cls += batch[0]['loss_rpn_cls']  # results is [[batch][batch] to compatible to evaluator.process()
            loss_rpn_bbox += batch[0]['loss_rpn_bbox']
            loss_cls += batch[0]['loss_cls']
            loss_bbox += batch[0]['loss_bbox']

        eval_results[f'loss_rpn_cls'] = loss_rpn_cls / len(results)
        eval_results[f'loss_rpn_bbox'] = loss_rpn_bbox / len(results)
        eval_results[f'loss_cls'] = loss_cls / len(results)
        eval_results[f'loss_bbox'] = loss_bbox / len(results)

        return eval_results
