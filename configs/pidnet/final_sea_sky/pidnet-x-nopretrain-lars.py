_base_ = './pidnet-s-nopretrain-lars.py'
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/pidnet/pidnet-l_imagenet1k_20230306-67889109.pth'  # noqa
model = dict(
    backbone=dict(
        channels=128,
        ppm_channels=112,
        num_stem_blocks=3,
        num_branch_blocks=4,
        # init_cfg=dict(checkpoint=checkpoint_file)
        ),
    decode_head=dict(in_channels=512, channels=512))

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader