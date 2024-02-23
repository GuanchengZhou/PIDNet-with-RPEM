_base_ = [
    '../../_base_/models/deeplabv3-space_r50-d8.py', '../../_base_/datasets/my_city_1ship_real.py',
    '../../_base_/default_runtime.py', '../../_base_/schedules/schedule_40k.py'
]
crop_size = (536, 960)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
