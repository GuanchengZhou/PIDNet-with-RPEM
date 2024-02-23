# Copyright (c) OpenMMLab. All rights reserved.
import os
from argparse import ArgumentParser

from mmengine.model import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model, show_result_pyplot, inference_model_time

import cv2
import numpy as np

def make_heatmap(img):
    drawn_img = None
    drawn_img = cv2.normalize(img, drawn_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    drawn_img[0,0] = 0
    drawn_img[0,1] = 255
    drawn_img = cv2.applyColorMap(drawn_img, cv2.COLORMAP_JET)
    drawn_img = cv2.resize(drawn_img, (960, 720), interpolation=cv2.INTER_NEAREST)
    return drawn_img

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)


    files = os.listdir( args.img)
    files_num = len(files)
    cnt_num, cnt_time = 0, 0
    timess = []
    for i, file in enumerate(files):
        if(file[-3:]=='txt'):
            continue
        # test a single image
        print(file)
        print(i,'/',files_num)
        in_file = os.path.join(args.img, file)
        out_file = args.out_file+file[:-3]+'png'
        out_file_gt = args.out_file+'_gt' + file
        print('out', out_file)
        result, now_time = inference_model_time(model, os.path.join(args.img, file))
        cnt_num += 1
        cnt_time += now_time
        timess.append(now_time)
        print('time', now_time, cnt_time/cnt_num)
        # print(result.shape)
        # print(result['pred_sem_seg'])
        # exit(0)
        # show the results
        import time
        start = time.time()
        show_result_pyplot(
            model,
            in_file,
            result,
            title=args.title,
            opacity=args.opacity,
            draw_gt=False,
            show=False if args.out_file is not None else True,
            out_file=out_file)
        end = time.time()
        print(' ',end-start)
        # feat = result.seg_logits.data.cpu().numpy()

        # heatmaps = [make_heatmap(img) for img in feat]
        # for i, heatmap in enumerate(heatmaps):
        #     tp_out_file = args.out_file + file.split('.')[0] + '_heat_' + str(i) + '.jpg'
        #     cv2.imwrite(tp_out_file, heatmap)
        # show_result_pyplot(
        #     model,
        #     in_file,
        #     result,
        #     title=args.title,
        #     opacity=args.opacity,
        #     draw_gt=True,
        #     draw_pred=False,
        #     show=False if args.out_file is not None else True,
        #     out_file=out_file_gt)
    print(np.mean(timess))



if __name__ == '__main__':
    main()