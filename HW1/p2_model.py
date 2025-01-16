import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
# from torchsummary import summary


class VGG16_FCN32s(nn.Module):
    def __init__(self, n_class=7) -> None:
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features[0].padding = (100, 100)

        # replace fc by conv
        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.fc6[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[0].weight.data.view(
            self.fc6[0].weight.size())
        self.fc6[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[0].bias.data.view(self.fc6[0].bias.size())

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.fc7[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[3].weight.data.view(
            self.fc7[0].weight.size())
        self.fc7[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[3].bias.data.view(self.fc7[0].bias.size())

        self.score_fr = nn.Sequential(
            nn.Conv2d(4096, n_class, 1),
            nn.ReLU()
        )

        self.upscore = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 64, stride=32)
        )

    def forward(self, x) -> torch.Tensor:
        raw_h, raw_w = x.shape[2], x.shape[3]
        x = self.features(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.score_fr(x)
        x = self.upscore(x)
        # crop to original size
        x = x[..., x.shape[2] - raw_h:, x.shape[3] - raw_w:]
        return x


class VGG16_FCN8s(nn.Module):
    def __init__(self, n_class=7) -> None:
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features[0].padding = (100, 100)

        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.fc6[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[0].weight.data.view(
            self.fc6[0].weight.size())
        self.fc6[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[0].bias.data.view(self.fc6[0].bias.size())

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.fc7[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[3].weight.data.view(
            self.fc7[0].weight.size())
        self.fc7[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[3].bias.data.view(self.fc7[0].bias.size())

        # classification layer
        self.score_fr = nn.Conv2d(4096, n_class, 1)

        # upsample layers
        self.upscore2 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(n_class, n_class, 16, stride=8, padding=4)

        # 1x1 convolution layers to reduce channels for pool4 and pool3
        self.score_pool4 = nn.Conv2d(512, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)

    def forward(self, x) -> torch.Tensor:
        raw_h, raw_w = x.shape[2], x.shape[3]

        # Feature extraction
        pool3 = self.features[:17](x)  # until pool3
        pool4 = self.features[17:24](pool3)  # until pool4
        x = self.features[24:](pool4)  # rest of the features

        # Fully connected layers (fc6 and fc7)
        x = self.fc6(x)
        x = self.fc7(x)

        # Score for final layer
        x = self.score_fr(x)

        # Upsample by 2x and combine with pool4
        x = self.upscore2(x)

        if pool4.shape[2] > x.shape[2]:
            pool4 = pool4[:, :, :x.shape[2], :]
        if pool4.shape[3] > x.shape[3]:
            pool4 = pool4[:, :, :, :x.shape[3]]

        pool4 = self.score_pool4(pool4)
        
        x = x + pool4  # element-wise sum

        # Upsample by another 2x and combine with pool3
        x = self.upscore_pool4(x)

        if pool3.shape[2] > x.shape[2]:
            pool3 = pool3[:, :, :x.shape[2], :]
        if pool3.shape[3] > x.shape[3]:
            pool3 = pool3[:, :, :, :x.shape[3]]

        pool3 = self.score_pool3(pool3)
        x = x + pool3  # element-wise sum

        # Final 8x upsample to get original size
        x = self.upscore8(x)

        # Crop to original size
        x = x[..., x.shape[2] - raw_h:, x.shape[3] - raw_w:]
        return x


class DeepLabV3_ResNet50(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        
        super(DeepLabV3_ResNet50, self).__init__()

        """
        Initializes the DeepLabV3_ResNet50 model.

        Args:
        num_classes (int): Number of output classes for segmentation.
        pretrained (bool): Whether to load the model pre-trained on COCO dataset.
        """

        # Load the pre-trained DeepLabV3 model
        self.model = torch.hub.load('pytorch/vision:v0.10.0','deeplabv3_resnet50', pretrained=pretrained)

        self.model.classifier = DeepLabHead(2048, num_classes)
        self.model.aux_classifier = FCNHead(1024, num_classes)

    
    def forward(self, x):
        """
        Forward pass for the model.

        Args:
        x (torch.Tensor): Input tensor representing the image.

        Returns:
        dict: Model output with keys 'out' and 'aux'.
        """
        return self.model(x)



# if __name__ == '__main__':
    
#     # net = VGG16_FCN32s() #.cuda()
#     # net = VGG16_FCN8s()
#     # print(net)

#     # net = DeepLabV3_ResNet50(num_classes=7)
#     # print(net)
#     # print(net(torch.rand(3, 3, 512, 512))['aux'].shape)



#     # pytorch_total_params = sum(p.numel() for p in net.parameters())
#     # print(pytorch_total_params)
