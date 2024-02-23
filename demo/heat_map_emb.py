import torch
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import torch.nn as nn

def get_din_emb(c, input, normalize = False):
    num_pos_feats = c
    temperature = 10000
    # normalize = True
    scale = 2 * math.pi  # 圆周率
    y_embed = input
    if normalize:
        eps = 1e-6
        # b = a[i:j:s]表示：i,j与上面的一样，但s表示步进，缺省为1.
        # 所以a[i:j:1]相当于a[i:j]
        # 当s<0时，i缺省时，默认为-1. j缺省时，默认为-len(a)-1
        # 所以a[::-1]相当于 a[-1:-len(a)-1:-1]，也就是从最后一个元素到第一个元素复制一遍，即倒序。
        # 对于X[:,:,m:n]是取三维矩阵中第m维到第n-1维的所有数据
        # 归一化
        y_embed = y_embed / (y_embed[-1:, :] + eps) * scale  # y_embed[:, -1:, :]代表取三维数据中的最后一行数据
        # x_embed = x_embed / (x_embed[:, -1:] + eps) * scale  # x_embed[:, :, -1:]代表取三维数据中的最后一列数据
        # print(y_embed)
        # print(x_embed)
    dim_t1 = torch.arange(num_pos_feats, dtype=torch.float32).cpu()
    # print(dim_t1)
    dim_t = temperature ** (2 * (dim_t1 // 2) / num_pos_feats)  # i=dim_t1 // 2
    # print(dim_t)
    # pos_x = x_embed[:, :, None] / dim_t
    # print(y_embed)
    pos_y = y_embed[:, :, None] / dim_t
    # print(pos_x)
    # print(pos_y.shape)
    # print(pos_y[:,:, 0])
    # pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # 不降维
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)  # 不降维
    pos_y = pos_y.permute(2,0,1)
    # print(pos_x)
    # print(pos_y)
    # pos = torch.cat((pos_y, pos_x), dim=2)
    # print(pos)
    return pos_y

def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.kaiming_uniform_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(
                module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

add_ori = torch.ones((500, 500)).cpu()
for i in range(500):
    add_ori[i] = torch.ones(500)*(i+1)

# print(add_ori)
# print(add_ori[0,0], add_ori[1,0])
h, w = (50, 50)
input = add_ori[:h,:w]
# print(5, input)
emb = get_din_emb(64,input, normalize=True)
draw_data = emb[:, :, 0].cpu().numpy()
draw_data2 = torch.empty(h,w)
draw_data2 = nn.init.kaiming_normal_(draw_data2, a=0, mode='fan_out', nonlinearity='relu')
draw_data2 = draw_data2.cpu().numpy()
import torch.nn.functional as F
draw_data3 = (input-h//2)/(h//2)
draw_data3 = draw_data3.cpu().numpy()
print(np.max(draw_data2), np.min(draw_data2))
# draw_data3 = bn(draw_data3).view(h,w)
# draw_data3 = draw_data3.detach().numpy()
# print(draw_data3.shape)

from matplotlib import pyplot as plt
print('r2')
import seaborn as sns
print('r3')
import numpy as np
import pandas as pd
print('r4')
#
# # print(input[0])

# draw__ = torch.empty((50,50))
# h, w = draw__.shape
# print(h,w)
# now_id = 2
# draw__ = nn.init.kaiming_normal_(draw__, a=0, mode='fan_out', nonlinearity='relu')
# # for i in range(h):
# #     for j in range(w):
# #         draw__[i,j] = draw_data[now_id, i]
draw__ = torch.empty((50, 50)).cpu()
draw__ = nn.init.kaiming_normal_(draw__, a=0, mode='fan_out', nonlinearity='relu')
draw__ = (draw__*255).int().numpy()

print('running0')
data = pd.DataFrame(draw__)
print('running1')
# # 绘制热度图：
plot = sns.heatmap(data)
print('finished')
plt.show()
