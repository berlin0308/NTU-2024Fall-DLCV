import argparse
import csv
import json
import pathlib

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from p1_dataloader import ImageDatasetFromFolders
from p1_model import Classifier


@torch.no_grad()
def main(args):
    print('Loading dataset')
    image_size = 128
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    dataset = ImageDatasetFromFolders(
        args.input_image_dir,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=12, pin_memory=True
    )
    
    with (pathlib.Path("label2idx.json")).open('r') as f:
        idx2label = {v: k for k, v in json.load(f).items()}


    print('Constructing model')
    model = Classifier(
        backbone=models.resnet50(weights=None),
        in_features=1000,
        n_class=65,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
    ).to(args.device)


    print('Loading model...')
    model.load_state_dict(torch.load(
    args.classifier_model_path, map_location=args.device))


    model.eval()

    print('Start Predicting...')
    predicts = dict() # k: filename, v: prediction
    features = dict() # k: filename, v: feature vectors

    for data in dataloader:
        img = data['img'].to(args.device)
        logits = model(img)
        y_pred = torch.argmax(logits, dim=1)
        for filename, pred in zip(data['filename'], y_pred):
            predicts[filename] = idx2label[pred.item()]

            if args.output_features:
                feature_vector = model.get_features()
                features[filename] = feature_vector.cpu().numpy().tolist()



    # print(predicts)

    print('Writing output...')
    # write output
    with args.input_csv.open('r') as in_f:
        with args.output_csv.open('w') as out_f:
            reader = csv.reader(in_f)
            next(iter(reader))  # skip header
            writer = csv.writer(out_f)

            if not args.output_features:
                writer.writerow(('id', 'filename', 'label'))
            else:
                writer.writerow(('id', 'filename', 'label', 'features'))


            for id, filename, label in reader:

                if not args.output_features:
                    writer.writerow((
                        id,
                        filename,
                        predicts[filename]
                    ))
                else:
                    feature_str = ','.join(map(str, features[filename]))
                    # print(feature_str)
                    writer.writerow((
                        id,
                        filename,
                        label,
                        feature_str
                    ))
    print('Done')


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_image_dir',
                        type=pathlib.Path, required=False, default='hw1_data/p1_data/office/train')
    parser.add_argument("--input_csv", type=pathlib.Path, required=False, default='hw1_data/p1_data/office/train.csv')
    parser.add_argument('--output_csv', type=pathlib.Path, required=False, default='p1_results_2/C_ep200_train.csv')
    parser.add_argument('--output_features', type=bool, default=False)


    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--classifier_model_path",
                        type=pathlib.Path, default="classifier_ep200_acc0.63.pth")

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    try:
        args.output_csv.parent.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print('Error:', e)
    main(args)