_base_ = [
    '../../_base_/models/deeplabv3_r50-d8.py', '../../_base_/datasets/mastr.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_80k.py'
]
crop_size = (536, 960)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    decode_head=dict(
        num_classes=3,
    ),
    auxiliary_head=dict(
        num_classes=3,
    )
)
