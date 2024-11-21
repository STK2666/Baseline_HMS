import torch
import torch.nn as nn
from torchvision.models import resnet50

# resnet50 = torch.hub.load('pytorch/vision:v0.11.0', 'resnet50', pretrained=True)
resnet50 = resnet50(pretrained=True)

# ResNet的结构可以分为`conv1`、`bn1`、`relu`、`maxpool`、和若干`layerX`模块

class PartialResNet50(nn.Module):
    def __init__(self):
        super(PartialResNet50, self).__init__()
        self.features = nn.Sequential(
            resnet50.conv1,  # 第一层卷积
            resnet50.bn1,    # 批量归一化
            resnet50.relu,   # ReLU激活
            resnet50.maxpool,  # 最大池化
            resnet50.layer1,  # 第一块ResNet层
            resnet50.layer2,  # 第二块ResNet层
            # resnet50.layer3   # 第三块ResNet层 (截断至此时，确保输出大小为 512×512×28)
        )

    def forward(self, x):
        return self.features(x)

# partial_model = PartialResNet50()

# dummy_input = torch.randn(1, 3, 256, 256)  # 假设输入大小为 512×512 的 RGB 图像
# output = partial_model(dummy_input)
# print("输出大小：", output.shape)  # 应输出 torch.Size([1, 512, 512, 28])
