# PyTorch lib
import torch.nn as nn
import torchvision


# Tools lib
# import cv2

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable


# Initialize VGG16 with pretrained weight on ImageNet
def vgg_init(device):
    vgg_model = torchvision.models.vgg16(pretrained=True).to(device)
    trainable(vgg_model, False)
    return vgg_model


# Extract features from internal layers for perceptual loss
class vgg(nn.Module):
    def __init__(self, vgg_model):
        super(vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output
