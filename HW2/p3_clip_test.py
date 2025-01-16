import os
import json
import clip
import torch
from PIL import Image

device_type = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_img = clip.load("ViT-B/32", device=device_type)

with open("./hw2_data/clip_zeroshot/id2label.json", "r") as file:
    label_mapping = json.load(file)

text_prompts = [f"A photo of {label}." for label in label_mapping.values()]
encoded_text = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(device_type)

img_dir = "./hw2_data/clip_zeroshot/val"

predictions = []

def get_accuracy(data):
    correct = sum(1 for item in data if item["actual_label"] == item["predicted_label"])
    accuracy = (correct / len(data)) * 100
    return accuracy

for img_name in os.listdir(img_dir):
    if img_name.endswith(".png"):

        actual_class = img_name.split("_")[0]
        actual_label = label_mapping[actual_class]

        img_path = os.path.join(img_dir, img_name)
        processed_img = preprocess_img(Image.open(img_path)).unsqueeze(0).to(device_type)

        with torch.no_grad():
            img_features = clip_model.encode_image(processed_img)
            txt_features = clip_model.encode_text(encoded_text)

            img_features /= img_features.norm(dim=-1, keepdim=True)
            txt_features /= txt_features.norm(dim=-1, keepdim=True)

            sim_score = (100.0 * img_features @ txt_features.T).softmax(dim=-1)
            confidence, best_idx = sim_score[0].topk(1)
            predicted_label = list(label_mapping.values())[best_idx.item()]

            predictions.append({
                "image_name": img_name,
                "actual_label": actual_label,
                "predicted_label": predicted_label,
                "confidence_score": 100 * confidence.item()
            })

for item in predictions:
    print(f"Image: {item['image_name']}, Actual Label: {item['actual_label']}, "
          f"Predicted Label: {item['predicted_label']}, Confidence: {item['confidence_score']:.2f}%")

overall_accuracy = get_accuracy(predictions)
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
