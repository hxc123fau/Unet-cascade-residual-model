import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet_cascade(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet_cascade, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

        self.conv11 = DoubleConv(in_ch, 64)
        self.pool11 = nn.MaxPool2d(2)
        self.conv12 = DoubleConv(64, 128)
        self.pool12 = nn.MaxPool2d(2)
        self.conv13 = DoubleConv(128, 256)
        self.pool13 = nn.MaxPool2d(2)
        self.conv14 = DoubleConv(256, 512)
        self.pool14 = nn.MaxPool2d(2)
        self.conv15 = DoubleConv(512, 1024)
        self.up16 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv16 = DoubleConv(1024, 512)
        self.up17 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv17 = DoubleConv(512, 256)
        self.up18 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv18 = DoubleConv(256, 128)
        self.up19 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv19 = DoubleConv(128, 64)
        self.conv20 = nn.Conv2d(64,out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        # print('every_size',c5.shape,up_6.shape, c4.shape,p4.shape)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        merge9 = torch.cat([up_9, c1], dim=1)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        # out = nn.Sigmoid()(c10)

        c11 = self.conv11(c10)
        p11 = self.pool11(c11)
        c12 = self.conv12(p11)
        p12 = self.pool12(c12)
        c13 = self.conv13(p12)
        p13 = self.pool13(c13)
        c14 = self.conv14(p13)
        p14 = self.pool14(c14)
        c15 = self.conv15(p14)
        up_16 = self.up16(c15)
        # print('every_size',c15.shape,up_16.shape, c14.shape,p14.shape)
        merge16 = torch.cat([up_16, c14], dim=1)
        c16 = self.conv16(merge16)
        up_17 = self.up17(c16)
        merge17 = torch.cat([up_17, c13], dim=1)
        c17 = self.conv17(merge17)
        up_18 = self.up18(c17)
        merge18 = torch.cat([up_18, c12], dim=1)
        c18 = self.conv18(merge18)
        up_19 = self.up19(c18)
        merge19 = torch.cat([up_19, c11], dim=1)
        c19 = self.conv19(merge19)
        c20 = self.conv20(c19)

        res=c20.permute(0, 2, 3, 1)
        res=res.contiguous().view(-1, 2)
        out = torch.softmax(res,dim=1)

        return out











