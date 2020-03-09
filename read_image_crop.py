import os
import glob
import cv2
import numpy as np
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import matplotlib.pyplot as plt
import random

# 第一种数据读取方式
transform = T.Compose([
    # T.Resize(224),
    # T.RandomCrop(256,256),
    # T.RandomHorizontalFlip(),
    # T.RandomSizedCrop(224),
    T.ToTensor(),  # 将图片从０－２５５变为０－１
    # T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化到［－１，１］
    # T.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5])  # 标准化到［－１，１］
])


class Train_Data(Dataset):
    def __init__(self, data_root, mask_root, transforms=transform):
        data_image = glob.glob(data_root + '/*.bmp')
        self.data_image = data_image
        # print('data_image',data_image)

        mask_image = glob.glob(mask_root + '/*.bmp')
        self.mask_image = mask_image
        # mask_image = glob.glob(mask_root + '/*.tiff')

        self.transforms = transforms
        # print('transforms',self.transforms)

    def train_read(self):
        crop_image_data = []
        crop_mask_data = []
        num_img = len(self.data_image)
        sequence = 0
        for index in range(num_img):
            data_image_path = self.data_image[index]
            mask_image_path = self.mask_image[index]
            # print('data_image_path',index,data_image_path)
            image_data = cv2.imread(data_image_path, 0)
            mask_data = cv2.imread(mask_image_path, 0)
            w = list(image_data.shape)[0]
            h = list(image_data.shape)[1]
            num_w = int(np.floor(w / 256))
            num_h = int(np.floor(h / 256))
            for i in range(num_w):
                for j in range(num_h):
                    new_image_data = image_data[256 * i:256 * (i + 1), 256 * j:256 * (j + 1)]
                    new_mask_data = mask_data[256 * i:256 * (i + 1), 256 * j:256 * (j + 1)]
                    pixel_sum = np.sum(new_mask_data == 0.0)  # if it is black
                    # print('pixel_sum',index,pixel_sum)
                    if pixel_sum > 3000:
                        # plt.imshow(new_mask_data,cmap='gray')
                        # plt.pause(0.3)
                        sequence += 1
                        # print('new_image_data',new_image_data.shape,type(new_image_data))
                        # cv2.imwrite('./crop_train/'+ str(index) + '_' + str(i * num_w + j)+'.bmp',new_image_data)
                        # cv2.imwrite('./crop_Train_GT/train_GT_' + str(index) + '_' + str(i * num_w + j)+'.bmp', new_mask_data)
                        # cv2.imwrite('./crop_img_sequence/' +str(sequence)+ '.bmp', new_image_data)
                        # cv2.imwrite('./crop_img_sequence_GT/train_GT_' + str(sequence) + '.bmp', new_mask_data)
                        new_image_data = np.expand_dims(new_image_data, axis=2)
                        new_mask_data = np.expand_dims(new_mask_data, axis=2)
                        new_image_data = np.concatenate((new_image_data, 255 - new_image_data), axis=-1)
                        new_mask_data = np.concatenate((new_mask_data, 255 - new_mask_data), axis=-1)
                        # print('new_image_data',new_image_data.shape)

                        crop_image_data.append(new_image_data)
                        crop_mask_data.append(new_mask_data)

        crop_image_data = np.array(crop_image_data)
        crop_mask_data = np.array(crop_mask_data)
        # print('crop_image_data',crop_image_data.shape,crop_mask_data.shape)
        # print('crop_image_data[112,:,:,0]',crop_image_data[112,:,:,0])
        # plt.imshow(crop_image_data[112,:,:,0],cmap='gray')
        # plt.show()

        b = crop_image_data.shape[0]
        w = crop_image_data.shape[1]
        h = crop_image_data.shape[2]
        c = crop_image_data.shape[3]
        all_batch = b
        print('all_batch', all_batch)
        res_crop_image_data = torch.zeros(b, c, w, h)
        res_crop_mask_data = torch.zeros(b, c, w, h)
        for batch in range(all_batch):
            crop_image_data_batch = Image.fromarray(crop_image_data[batch])
            crop_mask_data_batch = Image.fromarray(crop_mask_data[batch])
            # print('crop_mask_data_batch,crop_mask_data_batch',crop_mask_data_batch.shape,crop_mask_data_batch.shape)
            if self.transforms:
                crop_image_data_batch = self.transforms(crop_image_data_batch)
                crop_mask_data_batch = self.transforms(crop_mask_data_batch)
                # print('crop_mask_data_batch',crop_mask_data_batch.shape)
                print('crop_mask_data_batch',crop_mask_data_batch.shape)
                res_crop_image_data[batch] = crop_image_data_batch
                res_crop_mask_data[batch] = crop_mask_data_batch
                # print('res_crop_image_data',res_crop_image_data.shape,type(res_crop_image_data))
                # print('be_transformed')

        # tt=res_crop_image_data[19,0,:,:].numpy()
        # plt.imshow(tt)
        # plt.show()
        # print('crop_image_data22',crop_image_data[111,:,:,0])
        return res_crop_image_data, res_crop_mask_data


# dataset = Train_Data(data_root='./dataset',mask_root='./Train_GT')
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
#         print(data[0].shape, '..')
#         print(data[1].shape, '...')

# #第２种
# for data_batch,mask_batch in test_data_loader:
#     print(data_batch.size(),mask_batch.size())


class Test_Data(Dataset):
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
        # print('image_data11', image_data.shape, mask_data.shape)
        w = list(image_data.shape)[0]
        h = list(image_data.shape)[1]
        if w * h > 2600000:
        #     # print('1111')
            multi=2600000.0/w/h
            w = int(w * multi)
            h = int(h * multi)
        #     if w * h > 800000:
        #         w = int(w * 0.75)
        #         h = int(h * 0.75)

        new_w = w // 16 * 16
        new_h = h // 16 * 16
        # print('new_w',new_w,new_h)
        image_data = image_data[0:new_w, 0:new_h]
        mask_data = mask_data[0:new_w, 0:new_h]

        image_data = np.expand_dims(image_data, axis=2)
        mask_data = np.expand_dims(mask_data, axis=2)
        # print('image_data11',image_data.shape)
        image_data = np.concatenate((image_data, 255 - image_data), axis=-1)
        mask_data = np.concatenate((mask_data, 255 - mask_data), axis=-1)
        # print('image_data22',image_data.shape)
        image_data = Image.fromarray(image_data)
        mask_data = Image.fromarray(mask_data)
        # print('image_data,mask_data',image_data.shape,mask_data.shape)
        if self.transforms:
            image_data = self.transforms(image_data)
            mask_data = self.transforms(mask_data)
            # print('be_transformed', image_data.shape)
        # print('image_data333', image_data.shape)
        return image_data, mask_data

    def __len__(self):
        return len(self.data_image)
