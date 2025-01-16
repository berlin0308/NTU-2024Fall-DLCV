import os
import sys
import argparse
import pathlib
import imageio
import numpy as np
import torch
from torchvision import transforms

from p2_dataloader import ImageMaskDataset

from p2_model import DeepLabV3_ResNet50


@torch.no_grad()
def main(args):
    def pred2image(batch_preds, batch_names, out_path):
        # batch_preds = (b, H, W)
        for pred, name in zip(batch_preds, batch_names):
            pred = pred.detach().cpu().numpy()
            pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
            pred_img[np.where(pred == 0)] = [0, 255, 255]
            pred_img[np.where(pred == 1)] = [255, 255, 0]
            pred_img[np.where(pred == 2)] = [255, 0, 255]
            pred_img[np.where(pred == 3)] = [0, 255, 0]
            pred_img[np.where(pred == 4)] = [0, 0, 255]
            pred_img[np.where(pred == 5)] = [255, 255, 255]
            pred_img[np.where(pred == 6)] = [0, 0, 0]
            imageio.imwrite(os.path.join(
                out_path, name.replace('_sat.jpg', '_mask.png')), pred_img)


    device = torch.device(args.device)

    net = DeepLabV3_ResNet50(num_classes=7)
    pretrained_dict = torch.load(args.seg_model_path, map_location=device)
    new_pretrained_dict = {}
    for key, value in pretrained_dict.items():
        new_key = key.replace("module.", "")  
        new_pretrained_dict[new_key] = value

    net.load_state_dict(new_pretrained_dict)
    net = net.to(device)
    net.eval()


    mean = [0.485, 0.456, 0.406]  # imagenet
    std = [0.229, 0.224, 0.225]

    test_dataset = ImageMaskDataset(
        args.input_folder,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        train=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=4, shuffle=False, num_workers=6)

    try:
        os.makedirs(args.output_folder, exist_ok=True)
    except:
        pass

    for x, filenames in test_loader:
        with torch.no_grad():
            x = x.to(device)
            out = net(x)['out']
        pred = out.argmax(dim=1)
        pred2image(pred, filenames, args.output_folder)



def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_folder', type=pathlib.Path, required=False, default='hw1_data/p2_data/validation')
    parser.add_argument('--output_folder', type=pathlib.Path, required=False, default='p2_results/output_ep1')


    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--seg_model_path",
                        type=pathlib.Path, default="model_ep86_miou0.747.pth")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse()
    main(args)