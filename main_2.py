import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet_residual import Unet_residual
import torchvision
from read_image_crop import *
# from read_image import *
import matplotlib.pyplot as plt
from precision_recall import *
import cv2 as cv
import matplotlib
from matplotlib import cm
import sys

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
# print('device', torch.cuda.is_available(), device)

class train_class():

    def train_model(self, model, criterion, optimizer, num_epochs=40):
        # data_path = Train_Data(data_root='./train', mask_root='./Train_GT')
        # train_data_loader = DataLoader(dataset=data_set, num_workers=1, batch_size=1, pin_memory=True, shuffle=False,
        #                               drop_last=True)
        train_data,train_label=Train_Data(data_root='./train', mask_root='./Train_GT').train_read()
        # print('train_data',train_data.shape)
        # td=train_data.numpy()
        # plt.imshow(td[10,0,:,:])
        # plt.show()
        all_batch=len(train_data)
        # print('all_batch',all_batch)
        b = 1
        w = train_data.shape[2]
        h = train_data.shape[3]
        for i in range(0,all_batch,5):
            x = train_data[i].view(1,2,w,h)
            gt = train_label[i].view(1,2,w,h)
            # print('xxx',i,x)
            # print('yyy',gt)
            gt = gt.float()
            outputs = []
            step = 0
            sum_loss=0
            loss_last=0.0
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print(i,'-' * 10)
                step += 1
                x=x.float()
                gt=gt.float()
                inputs = x.cuda()
                labels = gt.cuda()
                optimizer.zero_grad()
                outputs = model(inputs).cuda()
                labels = labels.permute(0, 2, 3, 1)
                labels=labels.contiguous().view(-1, 2)
                loss = criterion(outputs, labels).cuda()
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()
                sum_loss += epoch_loss
                print("epoch %d loss:%0.3f" % (epoch, sum_loss / step))
                print('loss_now',loss)
                loss_now=loss
                # if epoch==0  or epoch==39:
                #     outputs22 = outputs.view(b, w, h, 2)
                #     outputs22 = outputs22.permute(0, 3, 1, 2)
                #     outputs22 = outputs22 * 255.0
                #     outputs22 = outputs22.cpu().detach().numpy()
                #     # print('shape',outputs22[0,0].shape)
                #     # print('max_outputs22',np.amax(outputs22[0,0]))
                #     # cv.imwrite('./detail/'+str(i)+'_'+str(epoch)+'.png',outputs22[0,0])
                #     plt.imshow(outputs22[0, 0], cmap='gray')
                #     plt.title(str(i)+'_'+str(epoch))
                #     plt.pause(1)

                    # if loss_now ==loss_last:
                    #     for name, param in model.named_parameters():
                    #         print('model_parameter', name, '      ', param.size(), param)
                    #         print('xxx',x[0,0])
                    #         np.save('input.npy',x[0,0].cpu().detach().numpy())
                            # print('outputs22',outputs22[0,0])
                            # print('gt',gt[0,0])
                            # print('unet_p1')
                    # loss_last = loss
                # gt_numpy=gt.cpu().detach().numpy()
                # print('outputs22',outputs22.shape)
                # plt.subplot(2,1,1)
                # plt.subplot(2,1,2)
                # plt.imshow(yy[0,0],cmap='gray')
                # plt.pause(1)
            # plt.show()
            # plt.close()

            outputs = outputs.view(b, w, h, 2)
            outputs = outputs.permute(0, 3, 1, 2)
            # print('outputs22',outputs)
            # print('labels',labels)
            output_numpy = outputs[0,0].cpu().detach().numpy()
            max_value=np.amax(output_numpy)
            min_value=np.amin(output_numpy)
            output_numpy = (output_numpy - min_value) / (max_value - min_value) * 255.0
            # plt.imshow(output_numpy, cmap='gray')
            # plt.pause(1)
            # print('output_numpy',output_numpy[0,0])
            gt_numpy=gt[0,0].cpu().detach().numpy()*255.0
            output_numpy=output_numpy.astype('uint8')
            th, res = cv.threshold(output_numpy, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            res=res
            # plt.subplot(2,1,1)
            # plt.imshow(x[0,0],cmap='gray')
            # plt.subplot(2,1,2)
            # plt.imshow(res,cmap='gray')
            # plt.pause(1)
            p, r, f1 = calculation_precison_recall().calculation(res, gt_numpy)
            print('precision,recall,F1', p, r, f1)

            # np.save('./predict_result_2/' + str(i) + '.npy', res)
            cv.imwrite('./predict_result_2/' + str(i) + '.png', res)
            cv.imwrite('./predict_result_2/GT_' + str(i) + '.png', gt_numpy)
            cv.imwrite('./predict_result_2/input_' + str(i) + '.png', 255.0*x[0,0].cpu().detach().numpy())
            # np.save('./predict_result_1/' + 'input' + str(i) + '.npy', x[0,0])
            # np.save('./predict_result_1/' + 'label' + str(i) + '.npy', y_numpy[0,0])
            # torchvision.utils.save_image(outputs[0,0], './predict_result_1/' + str(i) + '.png')
        # plt.show()
        # plt.close()
        torch.save(model.state_dict(), './predict_result_2/' + 'weights_2.pth' )

        return model

    # 训练模型
    def train(self):
        model = Unet_residual(in_ch=2, out_ch=2).to(device)
        # batch_size = 1
        # criterion = nn.BCEWithLogitsLoss()
        criterion = nn.BCELoss()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(),lr = 0.01)
        optimizer= optim.SGD(model.parameters(), lr = 0.005, momentum=0.9)
        # data_set = Train_Data(data_root='./train', mask_root='./Train_GT')
        self.train_model(model, criterion, optimizer)


# 显示模型的输出结果
class test_class():
    def test(args):
        model = Unet_residual(2, 2)
        model.load_state_dict(torch.load('./predict_result_2/weights_2.pth',map_location="cuda:0"))
        data_set = Test_Data(data_root='./test', mask_root='./Test_GT')
        test_data_loader = DataLoader(dataset=data_set, num_workers=1, batch_size=1, pin_memory=True, shuffle=False,
                                       drop_last=True)
        model.eval()
        # model.cuda()
        # print('optimizer',model)
        with torch.no_grad():
            for i, data in enumerate(test_data_loader, 0):
                # x = data[0].cuda()
                # gt=data[1].cuda()
                x = data[0]
                gt = data[1]
                # print('xxx,gt',x.shape,gt.shape)
                y = model(x)
                # y=x
                b=x.shape[0]
                w=x.shape[2]
                h=x.shape[3]
                y=y.view(b,w,h, 2)
                # print('yy1',y)
                y=y.permute(0, 3, 1, 2)
                # print('yy2', y.shape)
                # img_y = torch.squeeze(torch.sigmoid(y[0,0])).cpu().detach().numpy()
                img_y = torch.squeeze(y[0, 0]).cpu().detach().numpy()
                label=torch.squeeze(gt[0,0]).cpu().detach().numpy()*255.0
                min_value=np.amin(img_y)
                max_value=np.amax(img_y)
                # print('min_max',min_value,max_value)
                img_y_map = (img_y - min_value)/(max_value - min_value) * 255.0
                img_y_map = img_y_map.astype('uint8')
                # img_y_map=(img_y*255.).astype('uint8')
                th, res = cv.threshold(img_y_map, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                # res3=255.0-res
                # print('res_test',res)
                # print('label',type(label),np.amin(label),np.amax(label))
                # print('res3',type(res3),np.amin(res3),np.amax(res3))
                # print('th3',th)
                # plt.subplot(2,1,1)
                # plt.imshow(label,cmap='gray')
                # plt.subplot(2,1,2)
                # plt.imshow(res,cmap='gray')
                # matplotlib.image.imsave('./test_result_1/'+ str(i) + '.png', res,cmap="gray")
                # matplotlib.image.imsave('./test_result_1/' + 'GT_' + str(i) + '.png', label, cmap="gray")
                cv.imwrite('./test_result_2/' + str(i) + '.png', res)
                cv.imwrite('./test_result_2/gt_' + str(i) + '.png', label)
                cv.imwrite('./test_result_2/input_' + str(i) + '.png', x[0, 0].cpu().detach().numpy() * 255.0)

                p, r, f1 = calculation_precison_recall().calculation(res, label)
                # print('precision,recall,F1',p,r,f1)
                print("%.4f" % p,',', "%.4f" % r,',', "%.4f" % f1,',')
            #     plt.clf()
            #     plt.subplot(211)
            #     plt.imshow(res3,cmap='gray')
            #     plt.subplot(212)
            #     plt.imshow(label, cmap='gray')
            #     plt.pause(1)
            # plt.show()
            # plt.close()

if __name__ == '__main__':
    # train_class().train()
    # test_class().test()
    if sys.argv[1]=='train':
        train_class().train()
    elif sys.argv[1]=="test":
        test_class().test()
    else:
        raise ValueError("Must run either in train or test mode")
