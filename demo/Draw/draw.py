import os

import matplotlib
import matplotlib.pyplot as plt
import cv2

matplotlib.use('TkAgg')

# plt.rcParams['font.sans-serif'] = 'times'

img_name = 'DJI_0331_000180'
model_name = 'pidnet-s-1ship-space'

images = os.listdir('work_dirs/'+model_name+'/vis/')

for i,image_file in enumerate(images):
    print(img_name, i,'/',len(images))
    img_name = image_file.split('.')[0]
    if os.path.exists('work_dirs/'+model_name+'/vis/trytry/'+img_name+'.png'):
        print('pass')
        continue
    if len(img_name.split('heat')) >1:
        print('pass')
        continue
    if img_name[0]!='D':
        print('pass')
        continue

    crop_size = (48*2,27*2)

    oimg = cv2.imread('data/my_cityscapes3/leftImg8bit/val/'+img_name+'.jpg')
    oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
    # oimg = cv2.resize(oimg, crop_size)
    gtimg = cv2.imread('work_dirs/gt5/vis_data/vis_image/'+img_name+'_gt_0.png')
    gtimg = cv2.cvtColor(gtimg, cv2.COLOR_BGR2RGB)
    # gtimg = cv2.resize(gtimg, crop_size)
    pimg = cv2.imread('work_dirs/'+model_name+'/vis/'+img_name+'.jpg')
    pimg = cv2.cvtColor(pimg, cv2.COLOR_BGR2RGB)
    # pimg = cv2.resize(pimg, crop_size)

    fig, axes = plt.subplots(1, 3,
                             figsize=crop_size
                             )
    #
    # ax11 = axes[0]
    # ax12 = axes[1]
    # ax13 = axes[2]
    #
    axes[0].set_title('Origin Image',  fontsize=80)
    axes[1].set_title('Ground Truth',  fontsize=80)
    axes[2].set_title('Prediction Image',  fontsize=80)
    # axes[3].set_title('Sea')
    # axes[4].set_title('Sky')
    # axes[5].set_title('Land')
    # axes[6].set_title('Obstacle')
    # axes[7].set_title('Visionship')
    # axes[8].set_title('CloseShip')
    # axes[9].set_title('USV')
    #
    for ax in axes:
        ax.axis('off')
    #
    axes[0].imshow(oimg)
    axes[1].imshow(gtimg)
    axes[2].imshow(pimg)

    # for i in range(7):
    #     id = i+3
    #     img = cv2.imread('work_dirs/'+model_name+'/vis/'+img_name+'_heat_'+str(i)+'.jpg')
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img,crop_size)
    #     axes[id].imshow(img)

    # ax11.imshow(img1)
    # ax11.axis('off')
    # ax12.imshow(img2)
    # ax12.axis('off')
    # ax13.imshow(img3)
    # ax13.axis('off')

    # plt.show()
    plt.savefig('work_dirs/'+model_name+'/vis/trytry/'+img_name+'.png')
    plt.close()
