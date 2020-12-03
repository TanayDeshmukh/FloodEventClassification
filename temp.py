import torch
import numpy as np
from tifffile import TiffFile, TiffSequence, imread
import matplotlib.pyplot as plt

# tif = TiffFile('./data/0001/B01_series.tif')
# print(tif.pages[0].shape)
folder = '0073'
# temp = imread('./data/'+folder+'/B01_series.tif').transpose()
# print(temp.shape)

blue = torch.from_numpy(imread('./data/train/'+folder+'/B02_series.tif').transpose().astype(np.float))
print(blue.shape, len(blue))

green = torch.from_numpy(imread('./data/train/'+folder+'/B03_series.tif').transpose().astype(np.float))
print(green.shape)

red = torch.from_numpy(imread('./data/train/'+folder+'/B04_series.tif').transpose().astype(np.float))
print(red.shape)

# temp = imread('./data/'+folder+'/B05_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B06_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B07_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B08_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B8A_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B09_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B11_series.tif').transpose()
# print(temp.shape)

# temp = imread('./data/'+folder+'/B12_series.tif').transpose()
# print(temp.shape)

# a = temp[11]

# for i in range(a.shape[0]):
#     for j in range(a.shape[1]):
#         if(a[i, j] > 0):
#             print(a[i, j])

print(red.shape, green.shape, blue.shape)

# temp = torch.cat((red.unsqueeze(3), green.unsqueeze(3), blue.unsqueeze(3)), 3)
# print(temp.shape)

# out = temp[0].cpu().numpy()
# plt.imshow((out * 255).astype(np.uint8))
# plt.show()