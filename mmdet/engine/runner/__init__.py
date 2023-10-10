# Copyright (c) OpenMMLab. All rights reserved.
from .loops import TeacherStudentValLoop
from .test_collectloss_loop import TestLossLoop  # by lyz

__all__ = ['TeacherStudentValLoop', 'TestLossLoop']
