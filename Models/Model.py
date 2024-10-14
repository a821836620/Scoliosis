import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import numpy as np

def get_Model(model_name):
    base_model1 = None
    base_model2 = None
    feature_fusion = None
    decoder = None
    if model_name == 'densenet':
            base_model = models.densenet121(pretrained=True)
            base_model1 = torch.nn.Sequential(*(list(base_model.children())[:-1][0][:-6]))
            base_model2 = torch.nn.Sequential(*(list(base_model.children())[:-1][0][:-6]))
            feature_fusion = nn.Sequential(
              nn.Conv2d(256, 128, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 64, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 32, (3,3), (1,1), (1,1),bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.Conv2d(32, 16, (3,3), (1,1), (1,1),bias=False),
              nn.BatchNorm2d(16),
            )
            feature_size = 22528
            decoder = nn.Sequential(
                  nn.Linear(feature_size,512,bias=True)
            )
    elif model_name == 'AlexNet':
            base_model = models.alexnet(pretrained=True)
            base_model1 = torch.nn.Sequential(*(list(base_model.children())[:-1][0][:-1]))
            base_model2 = torch.nn.Sequential(*(list(base_model.children())[:-1][0][:-1]))
            feature_fusion = nn.Sequential(
              nn.Conv2d(512, 256, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 128, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 64, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 32, (3,3), (1,1), (1,1),bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.Conv2d(32, 16, (3,3), (1,1), (1,1),bias=False),
              nn.BatchNorm2d(16),
            )
            feature_size = 1536
            decoder = nn.Sequential(
                  nn.Linear(feature_size,512,bias=True)
            )
    elif model_name == 'vgg':
            base_model = models.vgg16(pretrained=True)
            base_model1 = torch.nn.Sequential(*(list(base_model.children())[:-1][0][:-1])) # 只取前几层，层数太深特征图太小
            base_model2 = torch.nn.Sequential(*(list(base_model.children())[:-1][0][:-1])) # 只取前几层，层数太深特征图太小
            feature_fusion = nn.Sequential(
              nn.Conv2d(1024, 512, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(512),
              nn.ReLU(inplace=True),
              nn.Conv2d(512, 256, (3,3), (1,1),(1,1),bias=False),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 128, (3,3), (1,1),(1,1),bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 64, (3,3), (1,1),(1,1),bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 32, (3,3), (1,1), (1,1),bias=False),
              nn.BatchNorm2d(32),
              nn.ReLU(inplace=True),
              nn.Conv2d(32, 16, (3,3), (1,1), (1,1),bias=False),
              nn.BatchNorm2d(16),
            )
            feature_size = 20832
            decoder = nn.Sequential(
                  nn.Linear(feature_size,512,bias=True)
            )
    elif model_name == 'ResNet':
            base_model = models.resnet18(pretrained=True)
            base_model1 = torch.nn.Sequential(*(list(base_model.children())[:-4])) # 只取前几层，层数太深特征图太小
            base_model2 = torch.nn.Sequential(*(list(base_model.children())[:-4])) # 只取前几层，层数太深特征图太小
            feature_fusion = nn.Sequential(
              nn.Conv2d(256, 128, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 64, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 32, (3,3), (2,2), (1,1),bias=False),
              nn.BatchNorm2d(32),
            )
            feature_size = 11264
            decoder = nn.Sequential(
                  nn.Linear(feature_size,512,bias=True)
            )
    elif model_name == 'ResNet34':
            base_model = models.resnet34(pretrained=True)
            base_model1 = torch.nn.Sequential(*(list(base_model.children())[:-4])) # 只取前几层，层数太深特征图太小
            base_model2 = torch.nn.Sequential(*(list(base_model.children())[:-4])) # 只取前几层，层数太深特征图太小
            feature_fusion = nn.Sequential(
              nn.Conv2d(256, 128, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True),
              nn.Conv2d(128, 64, (3,3), (2,2),(1,1),bias=False),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True),
              nn.Conv2d(64, 32, (3,3), (2,2), (1,1),bias=False),
              nn.BatchNorm2d(32),
            )
            feature_size = 11264
            decoder = nn.Sequential(
                  nn.Linear(feature_size,512,bias=True)
            )
    else:
        assert True
    return base_model1, base_model2, feature_fusion, decoder

class MultiInput(nn.Module):
    def __init__(self, args, out_channel) -> None:
        super().__init__() 
        # 两个输入，两个encoder层 不共享权重
        self.encoder1, self.encoder2, self.fusion, self.decoder = get_Model(args.model_name)
        self.out_layer = nn.Linear(512,out_channel,bias=True)

    def forward(self, inputs):
        f1 = self.encoder1(inputs[0])
        f2 = self.encoder2(inputs[1])
        f = torch.cat([f1, f2], dim=1) # 特征拼接
        x = self.fusion(f)
        batch_size, channels, height, width = x.size()
        dim = channels * height * width
        x = x.view(batch_size, dim)
        x = self.decoder(x)
        out = self.out_layer(x)
        return out

if __name__ == '__main__':
    get_Model('densenet')
    # input = torch.zeros(3,512,512)
    # output = base_model(input)
    # print(output)
