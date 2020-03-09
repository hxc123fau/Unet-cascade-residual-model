import os
import glob
import cv2
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# 第一种数据读取方式
transform = T.Compose([
    # T.Resize(224),
    # T.RandomCrop(256,256),
    # T.RandomHorizontalFlip(),
    # T.RandomSizedCrop(224),
    # T.CenterCrop((256,256)),
    T.ToTensor(),  # 将图片从０－２５５变为０－１
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到［－１，１］
    T.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])  # 标准化到［－１，１］
])


class Train_Data(Dataset):
    def __init__(self, data_root, mask_root, transforms=transform):
        data_image = glob.glob(data_root + '/*.bmp')
        self.data_image = data_image

        mask_image = glob.glob(mask_root + '/*.bmp')
        self.mask_image = mask_image

        self.transforms = transforms
        # print('transforms',self.transforms)

    def __getitem__(self, index):
        data_image_path = self.data_image[index]
        mask_image_path = self.mask_image[index]
        # print('data_image_path',data_image_path,mask_image_path)

        image_data = cv2.imread(data_image_path, 0)
        mask_data = cv2.imread(mask_image_path, 0)
        print('image_data,mask_data', image_data.shape, mask_data.shape)
        cw = int(list(image_data.shape)[0] / 2)
        ch = int(list(image_data.shape)[1] / 2)
        image_data = image_data[0:256, 0:256]
        mask_data = mask_data[0:256, 0:256]
        # if list(image_data.shape)[0] <512 and list(image_data.shape)[1] <512 :
        #     image_data = image_data[cw - 128:cw + 128, ch - 128:ch + 128]
        #     mask_data = mask_data[cw - 128:cw + 128, ch - 128:ch + 128]
        #     print('111')
        # else:
        #     image_data = image_data[0:512,0:512]
        #     mask_data = mask_data[0:512,0:512]
        #     print('222')


        image_data = np.expand_dims(image_data, axis=2)
        mask_data = np.expand_dims(mask_data, axis=2)
        print('image_data11',image_data.shape)
        image_data = np.concatenate((image_data, 255 - image_data), axis=-1)
        mask_data = np.concatenate((mask_data, 255 - mask_data), axis=-1)
        print('image_data22',image_data.shape)
        image_data = Image.fromarray(image_data)
        mask_data = Image.fromarray(mask_data)
        # print('image_data,mask_data',image_data.shape,mask_data.shape)
        if self.transforms:
            image_data = self.transforms(image_data)
            mask_data = self.transforms(mask_data)
            # print('be_transformed', image_data.shape)
        # print('image_data333', mask_data.shape)
        return image_data, mask_data

    def __len__(self):
        return len(self.data_image)

# dataset = Train_Data(data_root='./train',mask_root='./Train_GT')
#
# # 第一种调用，不常用
# for data,mask in dataset:
#      print(data,mask.shape)

# #下面两种应该在训练过程中更加好：
# if __name__ == '__main__':
#     test_data_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, pin_memory=True, shuffle=True,
#                                   drop_last=True)
#     print('test_data_loader', test_data_loader)
# # 第一种
#     for i, data in enumerate(test_data_loader, 0):
#         # print(data[0].shape, '..')
#         # print(data[1].shape, '...')
#         print(' ')

# #第２种
# for data_batch,mask_batch in test_data_loader:
#     print(data_batch.size(),mask_batch.size())
