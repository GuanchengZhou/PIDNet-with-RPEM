# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class My_CityscapesDataset_1ship(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        # classes=('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        #          'traffic light', 'traffic sign', 'vegetation', 'terrain',
        #          'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        #          'motorcycle', 'bicycle'),
        classes=(
            '__ignore__',
            'sea',
            'sky',
            'land',
            'obstacle',
            'visionship',
            'USV',
            'closeship',
            # 'USV',
        ),
        palette=
        [
         #    [0, 80, 100],
         # [244, 35, 232], [70, 70, 70], [102, 102, 156],
         # [190, 153, 153], [153, 153, 153],
         # [250, 170, 30], [220, 220, 0],
            [220, 220, 0],
            [107, 142, 35], [152, 251, 152], [70, 130, 180],
                     [220, 20, 60], [255, 0, 0],
            [0, 0, 142],
            [250, 170, 30],
         ],
        # label_map={0: 255,
        #            1: 0,
        #            2: 1,
        #            3: 2,
        #            4: 3,
        #            5: 4,
        #            6: 4,
        #            7: 4,
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
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=True, **kwargs)
