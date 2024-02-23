# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_model, init_model, show_result_pyplot, inference_model_time
from .mmseg_inferencer import MMSegInferencer

__all__ = [
    'init_model', 'inference_model', 'show_result_pyplot', 'MMSegInferencer', 'inference_model_time'
]
