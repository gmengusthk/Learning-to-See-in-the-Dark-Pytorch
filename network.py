import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from net_utils import conv2d, deconv_2d, DepthToSpace
import torch.optim as optim


class SeeInDarkNet(nn.Module):
    def __init__(self,input_channels=4,use_bn=False):
        super(SeeInDarkNet,self).__init__()
        self.pool=nn.MaxPool2d(2) 

        self.conv_1_1=conv2d(input_channels,32,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_1_2=conv2d(32,32,kernel_size=3,stride=1,use_bn=use_bn)

        self.conv_2_1=conv2d(32,64,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_2_2=conv2d(64,64,kernel_size=3,stride=1,use_bn=use_bn)

        self.conv_3_1=conv2d(64,128,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_3_2=conv2d(128,128,kernel_size=3,stride=1,use_bn=use_bn) 

        self.conv_4_1=conv2d(128,256,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_4_2=conv2d(256,256,kernel_size=3,stride=1,use_bn=use_bn) 

        self.conv_5_1=conv2d(256,512,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_5_2=conv2d(512,512,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_6=deconv_2d(512,256,use_bn=use_bn)
        self.conv_6_1=conv2d(512,256,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_6_2=conv2d(256,256,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_7=deconv_2d(256,128,use_bn=use_bn)
        self.conv_7_1=conv2d(256,128,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_7_2=conv2d(128,128,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_8=deconv_2d(128,64,use_bn=use_bn)
        self.conv_8_1=conv2d(128,64,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_8_2=conv2d(64,64,kernel_size=3,stride=1,use_bn=use_bn)

        self.deconv_9=deconv_2d(64,32,use_bn=use_bn)
        self.conv_9_1=conv2d(64,32,kernel_size=3,stride=1,use_bn=use_bn)
        self.conv_9_2=conv2d(32,32,kernel_size=3,stride=1,use_bn=use_bn)

        self.conv_10=conv2d(32,12,kernel_size=1,stride=1,use_bn=use_bn)

        self.depth_to_space=nn.PixelShuffle(2)
        # self.depth_to_space=DepthToSpace(2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)


    def forward(self, input_tensor):
        conv_1=self.conv_1_1(input_tensor)
        conv_1=self.conv_1_2(conv_1)
        conv_1_pool=self.pool(conv_1)

        conv_2=self.conv_2_1(conv_1_pool)
        conv_2=self.conv_2_2(conv_2)
        conv_2_pool=self.pool(conv_2)

        conv_3=self.conv_3_1(conv_2_pool)
        conv_3=self.conv_3_2(conv_3)
        conv_3_pool=self.pool(conv_3)

        conv_4=self.conv_4_1(conv_3_pool)
        conv_4=self.conv_4_2(conv_4)
        conv_4_pool=self.pool(conv_4)

        conv_5=self.conv_5_1(conv_4_pool)
        conv_5=self.conv_5_2(conv_5)

        conv_6_up=self.deconv_6(conv_5)
        conv_6=torch.cat((conv_6_up,conv_4),dim=1)
        conv_6=self.conv_6_1(conv_6)
        conv_6=self.conv_6_2(conv_6)

        conv_7_up=self.deconv_7(conv_6)
        conv_7=torch.cat((conv_7_up,conv_3),dim=1)
        conv_7=self.conv_7_1(conv_7)
        conv_7=self.conv_7_2(conv_7)

        conv_8_up=self.deconv_8(conv_7)
        conv_8=torch.cat((conv_8_up,conv_2),dim=1)
        conv_8=self.conv_8_1(conv_8)
        conv_8=self.conv_8_2(conv_8)

        conv_9_up=self.deconv_9(conv_8)
        conv_9=torch.cat((conv_9_up,conv_1),dim=1)
        conv_9=self.conv_9_1(conv_9)
        conv_9=self.conv_9_2(conv_9)

        conv_10=self.conv_10(conv_9)

        out=self.depth_to_space(conv_10)

        return out


if __name__=='__main__':

    model=SeeInDarkNet(use_bn=False).cuda()
    loss_function=nn.L1Loss().cuda()
    optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    for idx in range(32):
        input_tensor=torch.zeros((32,4,256,256),dtype=torch.float32).cuda()
        gt_tensor=torch.zeros((32,3,512,512),dtype=torch.float32).cuda()
        output=model(input_tensor)
        print(output.size())
        loss=loss_function(output,gt_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(loss.item())
