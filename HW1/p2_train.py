import os

import numpy as np
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from p2_dataloader import ImageMaskDataset
from p2_model import VGG16_FCN32s, VGG16_FCN8s, DeepLabV3_ResNet50
from p2_loss import MulticlassDiceLoss

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from mean_iou_evaluate import mean_iou_score

device = torch.device("cuda:1")
device_ids = [1, 2]


def compute_iou(pred, target):
    ious = []
    pred = pred.view(-1)  # flatten to 1D
    target = target.view(-1)  # flatten to 1D

    for cls in range(6): # except unknown

        pred_inds = pred == cls
        target_inds = target == cls

        intersection = (pred_inds[target_inds]).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - intersection

        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in IoU
        else:
            ious.append(intersection / union)

    return ious


# load data
mean = [0.485, 0.456, 0.406]  # imagenet
std = [0.229, 0.224, 0.225]
num_classes = 7

train_dataset = ImageMaskDataset(
    'hw1_data/p2_data/train',
    transform=transforms.Compose([
        transforms.
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    train=True, augmentation=True,
)

valid_dataset = ImageMaskDataset(
    'hw1_data/p2_data/validation',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    train=True,
)

batch_size = 8

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

epochs = 100
best_miou = 0.0
ckpt_path = f'p2_results/B_ckpt_7_aug_dice'

# model

model_id = 'DeepLabV3_ResNet50'

if model_id == 'VGG16_FCN32s':
    net = VGG16_FCN32s()
if model_id == 'DeepLabV3_ResNet50':
    net = DeepLabV3_ResNet50(num_classes=7)

# net = VGG16_FCN8s()
# net.copy_params_from_vgg16(models.vgg16(pretrained=True))

net = net.to(device)
net = nn.DataParallel(net, device_ids=device_ids)


net.train()
# loss_fn = nn.CrossEntropyLoss()
loss_fn = MulticlassDiceLoss(num_classes=7, softmax_dim=1)


optim = torch.optim.SGD(net.parameters(), lr=0.004, momentum=0.9)

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

for epoch in range(1, epochs + 1):

    # training
    net.train()
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optim.zero_grad()

        if model_id == 'VGG16_FCN32s':
            logits = net(x)  # no need to calculate soft-max
            loss = loss_fn(logits, y)

        if model_id == 'DeepLabV3_ResNet50':
            out = net(x)
            logits, aux_logits = out['out'], out['aux']
            loss = loss_fn(logits, y) + loss_fn(aux_logits, y)


        loss.backward()
        optim.step()

    # validation
    net.eval()
    with torch.no_grad():

        all_labels = []
        all_preds = []
        
        va_loss = 0
        miou = 0.0

        
        for x, y in tqdm(valid_loader):
            x, y = x.to(device), y.to(device)

            if model_id == 'VGG16_FCN32s':
                out = net(x)
            if model_id == 'DeepLabV3_ResNet50':
                out = net(x)['out']
        
            pred = out.argmax(dim=1)
            # va_loss += nn.functional.cross_entropy(out,
            #                                        y, ignore_index=6).item()
            va_loss += loss_fn(out, y).item()

            pred = pred.detach().cpu().numpy().astype(np.int64)
            y = y.detach().cpu().numpy().astype(np.int64)

            all_preds.append(pred)
            all_labels.append(y)

            
        va_loss /= len(valid_loader)

        # concat all predictions and labels, then calculate mIoU
        miou = mean_iou_score(np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0))


    net.train()

    print(f"epoch {epoch}, mIoU = {round(miou, 3)}, va_loss = {round(va_loss,3)}")
    if miou >= best_miou:
        best_miou = miou
        torch.save(optim.state_dict(), os.path.join(
            ckpt_path, 'best_optimizer.pth'))
        torch.save(net.state_dict(), os.path.join(ckpt_path, f'model_ep{epoch}_miou{round(miou, 3)}.pth'))

    # if (epoch % 10) == 0 or epoch == 1:
    #     torch.save(net.state_dict(), os.path.join(ckpt_path, f'model_ep{epoch}_miou{miou}.pth'))
