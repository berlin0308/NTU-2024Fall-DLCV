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

from p2_model import ImageCaptioningTransformerLoRA

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
    config = json.load((args.ckpt_dir / "model_config.json").open(mode='r'))
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
    # config['pretrained'] = False


    # model = ImageCaptioningTransformerLoRA(**config)
    model = ImageCaptioningTransformerLoRA(
            vocab_size=50257,
            encoder=config['encoder'],
            num_layers=config['num_layers'],
            nhead=config['nhead'],
            d_model=config['d_model'],
            dropout=0.0,
            rank=config['rank'],
            decoder_path="hw3_data/p2_data/decoder_model.bin"
        )
    
    model.load_state_dict(torch.load(
        args.ckpt_dir / "Best_model.pth", map_location=args.device))

    trainable_weights = [
        name for name, param in model.named_parameters() if param.requires_grad == True
    ]
    save_weights = {
        k: v for k, v in model.state_dict().items() if k in trainable_weights
    }
    print(f"Save lora weights {len(save_weights)}")
    torch.save(save_weights, f"p2/lora_module_weights.pt")
        

    model.to(args.device)
    model.eval()

    amp_enable = True
    amp_dtype = torch.float16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(f"Enable AMP training using dtype={amp_dtype} on {amp_device_type}")


    preds = dict()
    for data, name in tqdm(valid_set):
        
        with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
            output_ids = model.beam_search(data.to(args.device), beams=3, max_length=30, device=args.device)

        sentence = tokenizer.decode(output_ids)
        if len(sentence) > 2 and sentence[-1] == '.':
            sentence = sentence[:-2] + '.'  # remove white space before '.'
        preds[name] = sentence

        """
        Visualization
        """
        output_ids.insert(0, EOS_TOKEN)
        output_ids.insert(len(output_ids), EOS_TOKEN)
        print(tokenizer.decode(output_ids))


    json.dump(preds, args.output_json.open(mode='w'), indent=4)

    if args.do_eval:
        from collections import defaultdict

        import clip
        import language_evaluation

        # CLIP score
        model, image_process = clip.load('ViT-B/32', device=args.device)
        clip_scores = []
        for image_name, text in tqdm(preds.items()):
            image = Image.open(
                args.image_dir / f"{image_name}.jpg").convert('RGB')
            image = image_process(image).unsqueeze(0).to(args.device)
            text = clip.tokenize(text).to(args.device)

            with torch.no_grad():
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

            sim = image_features @ text_features.T
            score = 2.5 * max(sim.item(), 0)
            clip_scores.append(score)

        clip_score = sum(clip_scores) / len(clip_scores)
        print(f'clip score={clip_score}')

        # CIDEr score
        evaluator = language_evaluation.CocoEvaluator(coco_types=['CIDEr'])
        info = json.load(args.info_json.open(mode='r'))
        annotations = defaultdict(list)
        for data in info['annotations']:
            annotations[data['image_id']].append(data['caption'])
        img2id = {os.path.splitext(data['file_name'])[0]: data['id']
                  for data in info['images']}

        all_preds = []
        all_ans = []
        for image_name, text in preds.items():
            all_ans.append(annotations[img2id[image_name]])
            all_preds.append(text)
        CIDEr_score = evaluator.run_evaluation(
            all_preds, all_ans)['CIDEr']
        print(f'CIDEr score={CIDEr_score}')




def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")


    # Validation args
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument('--info_json', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')


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