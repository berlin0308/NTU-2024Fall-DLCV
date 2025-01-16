import argparse
import csv
import json
import os
import pathlib
from argparse import ArgumentParser

import clip
import torch
from torch import Tensor, nn
from PIL import Image
from tqdm.auto import tqdm
import logging
import torch.nn.functional as F
from tokenizer import BPETokenizer
import timm
import torchvision.transforms as trns
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import loralib as lora
import math
import collections
from decoder_lora import DecoderLoRA

# logging.basicConfig(
#     filename="my_log.log",
#     level=logging.INFO,
#     format="%(message)s",
# )

device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
PAD_TOKEN = -100
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256


def pad_sequences(sequences, pad_token_id=0):
    max_length = max([len(seq) for seq in sequences])
    padded_sequences = [
        seq + [pad_token_id] * (max_length - len(seq)) for seq in sequences
    ]
    return padded_sequences


class getDataset(Dataset):
    def __init__(self, img_dir, json_file, transform):
        super().__init__()
        print(f"Loading img from {img_dir}")
        print(f"Loading json from {json_file}")
        with open(json_file, "r") as file:
            info = json.load(file)
        self.tokenizer = BPETokenizer(encoder_file="encoder.json", vocab_file="vocab.bpe")
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        self.id2img = {}

        # notation
        for data in info["annotations"]:
            entry = {"caption": data["caption"], "image_id": data["image_id"]}
            self.data.append(entry)

        # img file
        for data in info["images"]:
            self.id2img[data["id"]] = data["file_name"]

    def __getitem__(self, index):
        info = self.data[index]  # {"caption":xxx , "image_id":xxx}
        imgname = self.id2img[info["image_id"]]
        img = Image.open(self.img_dir + "/" + imgname).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "caption": info["caption"],
            "filename": os.path.splitext(imgname)[0],
        }

    def __len__(self):
        return len(self.data)

    # retrun 一整個batch的dict
    def collate_fn(self, samples):
        captions2id = list()
        filenames = list()
        images = list()
        Start_token = 50256

        for sample in samples:
            id = self.tokenizer.encode(sample["caption"])
            if id[0] != Start_token:
                id.insert(0, Start_token)
            if id[-1] != Start_token:
                id.insert(len(id), Start_token)
            images.append(sample["image"])
            captions2id.append(id)
            filenames.append(sample["filename"])

        pad_captions2id = pad_sequences(captions2id, -1)
        attention_masks = [[float(i != -1) for i in seq] for seq in pad_captions2id]

        pad_captions2id = [
            [PAD_TOKEN if x == -1 else x for x in seq] for seq in pad_captions2id
        ]

        captions = torch.tensor(pad_captions2id)
        attention_mask_tensors = torch.tensor(attention_masks)
        images = torch.stack(images, dim=0)
        return {
            "images": images,
            "captions": captions,
            "filenames": filenames,
            "attmask": attention_mask_tensors,
        }


class Config:
    def __init__(self, checkpoint=None, rank=16):
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        self.vocab_size = 50257
        self.block_size = 1024
        self.checkpoint = checkpoint
        self.rank = rank  # Rank for LoRA


class ImgCaptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Tokenizer
        self.tokenizer = BPETokenizer(encoder_file="encoder.json", vocab_file="vocab.bpe")

        # Encoder (pretrained ViT model)
        self.encoder = timm.create_model(
            "vit_large_patch14_clip_224", pretrained=True, num_classes=0
        ).to(device)
        self.feature_resize = nn.Linear(1024, 768)

        # Decoder
        self.cfg = Config("hw3_data/p2_data/decoder_model.bin")
        self.decoder = DecoderLoRA(self.cfg).to(device)

        self.test = ""
        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.loss = 0

    def forward(self, batch_imgs, batch_captions, batch_attmask):

        """
        ground_truth = torch.concat(
            (batch_captions[:, 1:], batch_captions[:, :1]), dim=1
        )
        batch_attmask = torch.concat(
            (
                batch_attmask[:, 1:],
                torch.zeros(
                    (batch_attmask.shape[0], 1),
                    dtype=batch_attmask.dtype,
                    device=batch_attmask.device,
                ),
            ),
            dim=1,
        )
        feature = self.encoder.forward_features(batch_imgs)  # feature [64, 197, 768]
        feature = self.feature_resize(feature)
        decoder_output = self.decoder(batch_captions, feature)

        # setting ground truth
        for i, attmask in enumerate(batch_attmask):
            for j, element in enumerate(attmask):
                if element == 0:
                    ground_truth[i][j] = -100

        # # test block
        # _, output_id = torch.max(decoder_output[0], dim=-1)
        # self.test = self.tokenizer.decode(output_id.tolist())
        # print(self.test)

        decoder_output = torch.swapaxes(decoder_output, 1, 2)
        """

        ground_truth = batch_captions[:, 1:].tolist() 
        batch_attmask = batch_attmask[:, 1:].tolist()   

        feature = self.encoder.forward_features(batch_imgs)
        feature = self.feature_resize(feature)          

        decoder_output = self.decoder(batch_captions, feature)
        max_len = decoder_output.size(1)

        # pad the sequences
        ground_truth = pad_sequences(ground_truth, pad_token_id=-100)
        ground_truth = [seq[:max_len] + [-100] * (max_len - len(seq)) for seq in ground_truth]
        batch_attmask = pad_sequences(batch_attmask, pad_token_id=0)
        batch_attmask = [seq[:max_len] + [0] * (max_len - len(seq)) for seq in batch_attmask]

        ground_truth = torch.tensor(ground_truth, device=device)
        batch_attmask = torch.tensor(batch_attmask, device=device)

        ground_truth = ground_truth.masked_fill(batch_attmask == 0, -100)

        decoder_output = decoder_output.permute(0, 2, 1)

        # print(f"decoder_output shape: {decoder_output.shape}")
        # print(f"ground_truth shape: {ground_truth.shape}")
        # print(f"batch_attmask shape: {batch_attmask.shape}")

        """
        decoder_output shape: torch.Size([32, 50257, 279])
        ground_truth shape: torch.Size([32, 279])
        batch_attmask shape: torch.Size([32, 279])
        """

        self.loss = self.criterion(decoder_output, ground_truth)
        return self.loss

    def beam_search(self, img, beams=3, max_length=30):
        self.eval()

        def forward_prob(x: Tensor, encoder_feature: Tensor):
            
            x = torch.narrow(x, 1, 0, min(x.size(1), self.decoder.block_size))
            pos = torch.arange(
                x.size()[1], dtype=torch.long, device=x.device
            ).unsqueeze(0)

            x = self.decoder.transformer.wte(x) + self.decoder.transformer.wpe(pos)
            x = torch.cat((encoder_feature, x), dim=1)

            for block in self.decoder.transformer.h:
                x = block(x)

            x = self.decoder.lm_head(self.decoder.transformer.ln_f(x[:, -1, :]))
            return x

        if img.dim() < 4:
            img = img.unsqueeze(0)
        encoder_feature = self.encoder.forward_features(img)
        encoder_feature = self.feature_resize(encoder_feature)
        cur_state = torch.tensor([BOS_TOKEN]).to(device).unsqueeze(1)
        ### Beam Search Start ###
        # get top k words
        next_probs = forward_prob(cur_state, encoder_feature)

        vocab_size = next_probs.shape[-1]
        # 選擇概率最高的beams個單詞作為初始候選序列

        # probs, pred id
        cur_probs, next_chars = next_probs.log_softmax(-1).topk(k=beams, axis=-1)
        cur_probs = cur_probs.reshape(beams)
        next_chars = next_chars.reshape(beams, 1)
        # gen first k beams
        cur_state = cur_state.repeat((beams, 1))  # 複製 beams 次
        cur_state = torch.cat((cur_state, next_chars), axis=1)

        ans_ids = []
        ans_probs = []
        for i in range(max_length - 1):
            # get top k beams for beam*beam candidates
            # print("current state: ", cur_state)
            next_probs = forward_prob(
                cur_state, encoder_feature.repeat((beams, 1, 1))
            ).log_softmax(-1)
            cur_probs = cur_probs.unsqueeze(-1) + next_probs
            cur_probs = cur_probs.flatten()  # (beams*vocab) 攤平成1D

            # length normalization
            # cur_probs / (len(cur_state[0]) + 1) -> nomalized
            _, idx = (cur_probs / (len(cur_state[0]) + 1)).topk(k=beams, dim=-1)
            cur_probs = cur_probs[idx]

            # get corresponding next char
            next_chars = torch.remainder(idx, vocab_size)
            next_chars = next_chars.unsqueeze(-1)
            # print("next char: ",next_chars)

            # get corresponding original beams
            top_candidates = (idx / vocab_size).long()  # 找回屬於哪個beam
            cur_state = cur_state[top_candidates]
            cur_state = torch.cat((cur_state, next_chars), dim=1)

            # concat next_char to beams
            to_rm_idx = set()
            for idx, ch in enumerate(next_chars):
                if i == (max_length - 2) or ch.item() == EOS_TOKEN:
                    ans_ids.append(cur_state[idx].cpu().tolist())
                    # print(cur_probs[idx].item()," / ",len(ans_ids[-1]))
                    ans_probs.append(cur_probs[idx].item() / len(ans_ids[-1]))
                    to_rm_idx.add(idx)
                    beams -= 1

            to_keep_idx = [i for i in range(len(cur_state)) if i not in to_rm_idx]
            if len(to_keep_idx) == 0:
                break
            cur_state = cur_state[to_keep_idx]
            cur_probs = cur_probs[to_keep_idx]

        max_idx = torch.argmax(torch.tensor(ans_probs)).item()

        # 把50256抽離
        ans_ids[max_idx] = [x for x in ans_ids[max_idx] if x != EOS_TOKEN]
        # print(ans_ids)
        return ans_ids[max_idx]


def norm_long(x):
    x /= x.norm(dim=-1, keepdim=True)
    return x.long()


def save_json(json_path, filename, pred_caption):
    pred_caption_list = []
    pred_caption_list.append(pred_caption)
    new_data = dict(zip(filename, pred_caption_list))
    # json_file = sys.argv[2]
    # 嘗試讀取現有數據
    try:
        with open(f"{json_path}", "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(f"{json_path}", "w") as file:
            data = {}

    data.update(new_data)
    with open(f"{json_path}", "w") as f:
        json.dump(data, f)

def main(args):

    device = f"cuda:{args.c}" if torch.cuda.is_available() else "cpu"

    # args parameters
    EPOCHS = 5
    tokenizer = BPETokenizer(encoder_file="encoder.json", vocab_file="vocab.bpe")
    
    # Dataloader setting
    transform = create_transform(
        **resolve_data_config({}, model="vit_large_patch14_clip_224")
    )
    train_dir = "hw3_data/p2_data/images/train"
    train_json = "hw3_data/p2_data/train.json"
    val_dir = "hw3_data/p2_data/images/val"
    val_json = "hw3_data/p2_data/val.json"
    
    train_dataset = getDataset(
        img_dir=train_dir,
        json_file=train_json,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.b,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_dataset = getDataset(
        img_dir=val_dir,
        json_file=val_json,
        transform=transform,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # Model
    model = ImgCaptionModel().to(device)
    lora.mark_only_lora_as_trainable(model)

    for param in model.feature_resize.parameters():
        param.requires_grad = True

    # for i in range(len(model.decoder.transformer.h)):
    #     for param in model.decoder.transformer.h[i].crossattn.parameters():
    #         param.requires_grad = True

    print(
        f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M"
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}")

    trainable_weights = [
        name for name, param in model.named_parameters() if param.requires_grad == True
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(EPOCHS):

        loss = 0
        pbar = tqdm(train_loader)
        for data in pbar:
            model.train()
            # Prepare data
            optimizer.zero_grad()
            data["images"] = data["images"].to(device)
            data["captions"] = data["captions"].to(device)

            # model input: img
            # with torch.autocast(device_type="cuda"):
            loss = model(
                batch_imgs=data["images"],
                batch_captions=data["captions"],
                batch_attmask=data["attmask"],
            )

            # Update
            loss.backward()
            optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.3f}")

        save_weights = {
            k: v for k, v in model.state_dict().items() if k in trainable_weights
        }
        torch.save(save_weights, f"p2/model_lora_r16_lr{args.lr}_{epoch}.pt")

        model.eval()
        for val_data in tqdm(val_loader):
            val_data["image"] = val_data["image"].to(device)
            
            with torch.autocast(device_type=device):
                output_ids = model.beam_search(val_data["image"])
                # output_ids = model.greedy_search(val_data["image"])

            sentence = tokenizer.decode(output_ids)
            save_json(f"p2/model_lora_r16_lr{args.lr}_{epoch}_g.json", val_data["filename"], sentence)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--tag", type=str, default="_", help="Tag of an experiment"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "-c", type=int, default=1, help="cuda"
    )
    parser.add_argument(
        "-b", type=int, default=32, help="batch size"
    )

    args = parser.parse_args()
    main(args)
