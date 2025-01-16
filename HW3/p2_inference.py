import argparse
import json
import os
import pathlib

import torch
from PIL import Image
from tokenizer import BPETokenizer
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from p2_model import ImageCaptioningModel

PAD_TOKEN = 50256
UNK_TOKEN = 1
BOS_TOKEN = 50256
EOS_TOKEN = 50256

class Image_dataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        self.Transform = transform
        self.Image_names = [p for p in root.glob("*")]

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx]).convert('RGB')
        img = self.Transform(img)

        return img, os.path.splitext(os.path.basename(self.Image_names[idx]))[0]

    def __len__(self):
        return len(self.Image_names)


def main(args):
    # config = json.load((args.ckpt_dir / "model_config.json").open(mode='r'))
    tokenizer = BPETokenizer(encoder_file="encoder.json", vocab_file="vocab.bpe")
    transform = transforms.Compose([
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[
                             0.2290, 0.2240, 0.2250])
    ])
    valid_set = Image_dataset(
        root=args.image_dir,
        transform=transform,
    )

    trainable_weights_path = args.checkpoint
    print("Load trainable weights: ", trainable_weights_path)
    trainable_weights = torch.load(trainable_weights_path)

    # decoder_path = "hw3_data/p2_data/decoder_model.bin"
    model = ImageCaptioningModel(args.decoder_path).to(args.device)
    model.load_state_dict(trainable_weights, strict=False)

    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
        
    model.to(args.device)
    model.eval()

    amp_enable = True
    amp_dtype = torch.float16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(f"Enable AMP training using dtype={amp_dtype} on {amp_device_type}")


    preds = dict()
    for data, name in tqdm(valid_set):
        img = data.to(args.device)
        with torch.autocast(device_type="cuda"):
        # with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
            if args.search==1:
                output_ids = model.greedy_search(img)
            else:
                output_ids = model.beam_search(img, beams=args.search, max_length=30)
        
        sentence = tokenizer.decode(output_ids)
        if len(sentence) > 2 and sentence[-1] == '.':
            sentence = sentence[:-1] + '.'  # remove white space before '.'
        preds[name] = sentence
        # print(sentence)

        """
        Post-processing
        """
        # output_ids.insert(0, EOS_TOKEN)
        # output_ids.insert(len(output_ids), EOS_TOKEN)
        # print(tokenizer.decode(output_ids))

    json.dump(preds, args.output_json.open(mode='w'), indent=4)


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument('--info_json', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')

    parser.add_argument('--search', type=int,
                        default=1) # 1: greedy search, >1: beam search

    parser.add_argument("--checkpoint", type=pathlib.Path, default=" ")

    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    parser.add_argument('--output_json', type=pathlib.Path,
                        default='pred.json')
    parser.add_argument('--decoder_path', type=pathlib.Path,
                        default=' ')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.output_json.parent.mkdir(exist_ok=True, parents=True)
    main(args)