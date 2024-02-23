_base_ = [
    '../_base_/models/fcn_r50_space-d8.py',
    '../_base_/datasets/seasky.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (769, 769)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        use_space=True,
        space_emb='sin_not_normalize',
        space_cal='normal_plus_3',
    ),
    decode_head=dict(align_corners=True, dilation=6, num_classes=6),
    auxiliary_head=dict(align_corners=True, dilation=6, num_classes=6),
    test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513)))
