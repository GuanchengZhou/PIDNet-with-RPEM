# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LarsDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        # classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        #          'traffic light', 'traffic sign', 'vegetation', 'terrain',
        #          'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        #          'motorcycle', 'bicycle'),
        classes=('obstacle', 'water', 'sky'
            # '__ignore__',
            # 'sea',
            # 'sky',
            # 'land',
            # 'obstacle',
            # 'tower',
            # 'ship',
            # 'USV',
            # 'closeship',
            # 'USV',
        ),
        palette= [[247, 195, 37], [41, 167, 224], [90, 75, 164]],
        # [

         #    [0, 80, 100],
         # [244, 35, 232], [70, 70, 70], [102, 102, 156],
         # [190, 153, 153], [153, 153, 153],
         # [250, 170, 30], [220, 220, 0],
         #    [220, 220, 0],
         #    [107, 142, 35], [152, 251, 152], [70, 130, 180],
         #    [220, 20, 60], [255, 0, 0],
         #    [0, 0, 142],
         #    [250, 170, 30],
         # ],
        # label_map={0: 0,
        #            1: 1,
        #            2: 2,
        #            3: 3,
        #            4: 4,
        #            5: 5,
        #            6: 255,
        #            7: 255,
        #            8: 255,
        #            9: 255,
        #            # 10: 255,
        #            # 11: 255,
        #            },
        # [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        #          [190, 153, 153], [153, 153, 153], [250, 170,
        #                                             30], [220, 220, 0],
        #          [107, 142, 35], [152, 251, 152], [70, 130, 180],
        #          [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        #          [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]]
    )

    def __init__(self,
                 # img_suffix='_leftImg8bit.png',
                 # seg_map_suffix='_gtFine_labelIds.png',
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 # reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, ignore_index=255, reduce_zero_label=False, **kwargs)
