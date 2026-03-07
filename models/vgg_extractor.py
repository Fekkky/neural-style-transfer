import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()

        # relu1_1
        for x in range(2):
            self.slice1.add_module(str(x), vgg[x])
        # relu2_1
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg[x])
        # relu3_1
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg[x])
        # relu4_1
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg[x])
        # conv4_2 ← 内容损失
        for x in range(21, 23):
            self.slice5.add_module(str(x), vgg[x])
        # relu5_1 ← 风格损失第五层
        for x in range(23, 30):
            self.slice6.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.slice1(x)
        relu2_1 = self.slice2(relu1_1)
        relu3_1 = self.slice3(relu2_1)
        relu4_1 = self.slice4(relu3_1)
        conv4_2 = self.slice5(relu4_1)
        relu5_1 = self.slice6(conv4_2)
        # 风格：五层，内容：conv4_2
        return relu1_1, relu2_1, relu3_1, relu4_1, conv4_2, relu5_1