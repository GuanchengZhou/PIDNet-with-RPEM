from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyVOCDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('USV', 'Ship', 'Person', 'Building', 'Buoy', 'Dock'),
        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192],
                 [0, 64, 64], [0, 192, 224], [0, 192, 192], [128, 192, 64],
                 [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224],
                 [0, 0, 64], [0, 160, 192], [128, 0, 96], [128, 0, 192],
                 [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128],
                 [64, 128, 32], [0, 160, 0]])

    def __init__(self, aeg1, arg2):
        pass