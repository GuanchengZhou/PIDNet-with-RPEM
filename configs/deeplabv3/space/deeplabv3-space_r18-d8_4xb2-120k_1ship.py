_base_ = './deeplabv3-space_r50-d8_4xb2-120k_1ship.py'
model = dict(
    # pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(
        in_channels=512,
        channels=128,
    ),
    auxiliary_head=dict(in_channels=256, channels=64))
