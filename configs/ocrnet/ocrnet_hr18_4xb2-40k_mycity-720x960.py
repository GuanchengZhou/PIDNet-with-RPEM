_base_ = [
    '../_base_/models/ocrnet_hr18.py', '../_base_/datasets/my_city.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]

crop_size = (720, 960)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
