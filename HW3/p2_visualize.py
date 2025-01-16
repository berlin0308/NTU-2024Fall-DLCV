import json, os
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
import torch.nn.functional as F
from tokenizer import BPETokenizer
import timm
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import math
import collections
from p2_model import ImageCaptioningModel
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

device = "cuda:2" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256


class ImageCaptionDataset(Dataset):
    def __init__(self, img_dir, transform):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.Transform = transform
        self.Image_names = [p for p in self.img_dir.glob("*")]

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx]).convert("RGB")
        ori_trans = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        ori_img = ori_trans(img)
        img = self.Transform(img)
        return (
            ori_img,
            img,
            os.path.splitext(os.path.basename(self.Image_names[idx]))[0],
        )

    def __len__(self):
        return len(self.Image_names)



def visualize_attention(img, querys, keys, output_ids, img_name):
    tokenizer = BPETokenizer()
    
    img = img.squeeze(0).permute(1, 2, 0).cpu()
    img = (img - img.min()) / (img.max() - img.min())
    # img = torch.pow(img, 2) 
    img_resized = F.interpolate(img.unsqueeze(0).permute(0, 3, 1, 2), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0)

    num_cols = 5
    num_plots = len(querys)
    num_rows = math.ceil(num_plots / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))  # 调整图像尺寸
    plt.subplots_adjust(hspace=0.4, wspace=0.2)

    for i in range(len(querys)):
        ax = axes[i // num_cols, i % num_cols]
        title = tokenizer.decode([output_ids[i]])
        
        att = querys[i] @ keys[i].permute(1, 0) * (1.0 / math.sqrt(keys[i].size(-1)))
        att = att[-1, 1:].view(1, 16, 16)
        attention_resized = F.interpolate(
            att.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )

        ax.imshow(img_resized.cpu())
        ax.set_title(f"{title}")
        ax.axis('off')

        if i != 0:
            ax.imshow(attention_resized.squeeze().cpu().numpy(), cmap="jet", alpha=0.5)

    for i in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[i])

    plt.savefig(f"{img_name[0]}.png")


def main():
    # args parameters
    trainable_path = "p2/adapter/model_P2_3.pt"
    print("Load trainable weights: ", trainable_path)
    trainable_weights = torch.load(trainable_path)

    # Dataloader setting
    transform = create_transform(
        **resolve_data_config(
            {}, model="vit_large_patch14_clip_224.openai_ft_in12k_in1k"
        )
    )

    # val_dir = "hw3_data/p3_data/images"
    val_dir = "p3_best_last"

    val_dataset = ImageCaptionDataset(
        img_dir=val_dir,
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    print(f"Load decoder weights and build ImageCaptioningModel")
    decoder_path = "hw3_data/p2_data/decoder_model.bin"
    model = ImageCaptioningModel(decoder_path).to(device)
    model.load_state_dict(trainable_weights, strict=False)

    # hook
    hook_q = []
    hook_k = []

    def fetch_q(module, input, output):
        query = output[1].detach()
        hook_q.append(query.squeeze(0))

    def fetch_k(module, input, output):
        key = output[2].detach()
        hook_k.append(key.squeeze(0))

    # validation part
    model.eval()
    for val_data in tqdm(val_loader):
        
        # features, feature_sizes = [], []

        # to_rm_l = register_attention_hook(Model, features, feature_sizes)
        # attention_matrix = features[-1]

        hook_q = []
        hook_k = []
        for block in model.decoder.transformer.h:
            block.crossattn.register_forward_hook(fetch_q)
            block.crossattn.register_forward_hook(fetch_k)

        ori_img, img, filename = val_data
        img = img.to(device)

        with torch.autocast(device_type="cuda"):
            output_ids = model.greedy_search(img)

        output_ids.insert(0, EOS_TOKEN)
        output_ids.insert(len(output_ids), EOS_TOKEN)
        tokenizer = BPETokenizer()
        print(tokenizer.decode(output_ids))

        querys = []
        keys = []
        querys = [hook_q[i] for i in range(1, len(hook_q)) if i % 11 == 0]
        keys = [hook_k[i] for i in range(1, len(hook_k)) if i % 11 == 0]

        visualize_attention(
            ori_img,
            querys[: len(output_ids)],
            keys[: len(output_ids)],
            output_ids,
            filename,
        )


if __name__ == "__main__":
    main()