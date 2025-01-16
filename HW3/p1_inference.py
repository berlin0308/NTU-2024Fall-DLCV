import json
import os
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration #, BitsAndBytesConfig, pipeline
from PIL import Image
import torch
from p1_visualize import visualize_word_attention
import argparse
import pathlib

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )


def gen_captions(image_folder, output_json):
    # Load model and processor
    model_id = "llava-hf/llava-1.5-7b-hf"
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
    ).to(0)
    
    processor = AutoProcessor.from_pretrained(model_id)


    # prompt = "USER: <image>\nWrite a description for the photo in one sentence describing: the exact subject, its accurate action, and precise visual context. For example, A busy intersection with an ice cream truck driving by. ASSISTANT:"

    prompt = "USER: <image>\nBriefly describe what you see in one sentence. ASSISTANT:"

    config = {
                "num_beams": 5, 
                "max_new_tokens": 40,
                "do_sample": True,
                "temperature": 0.27,
                "top_p": 0.85,
                "no_repeat_ngram_size": 4,
                "repetition_penalty": 1.35,
                "length_penalty": 0.91,
                "early_stopping": True
            }
        

    # prompt = "USER: <image>\nSummarize what you see in one sentence. ASSISTANT:"

    # Get list of image files
    image_files = [
        os.path.join(image_folder, f) for f in os.listdir(image_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]
    captions = {}
    
    batch_size = 4  # Adjust based on your GPU memory capacity
    for i in range(0, len(image_files), batch_size):
        print(f'{i}/{len(image_files)/batch_size}')
        batch_files = image_files[i:i+batch_size]
        images = []
        valid_files = []
        for image_file in batch_files:
            try:
                image = Image.open(image_file).convert('RGB')
                images.append(image)
                valid_files.append(image_file)
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
        
        if not images:
            continue
        
        prompts = [prompt] * len(images)
        
        inputs = processor(
            images=images, text=prompts, return_tensors='pt', padding=True
        ).to(0, torch.float16)
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)

        
        for idx, output in enumerate(outputs):
            caption = processor.decode(output[3:], skip_special_tokens=True)
            image_file = valid_files[idx]
            captions[os.path.splitext(os.path.basename(image_file))[0]] = caption.split('ASSISTANT: ')[-1]
            print(f"{os.path.basename(image_file)}: {caption.split('ASSISTANT: ')[-1]}")

    # Save captions to JSON file
    with open(output_json, 'w') as f:
        json.dump(captions, f, indent=2)



def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    parser.add_argument('--output_json', type=pathlib.Path,
                        default='pred.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse()

    image_directory = args.image_dir
    output_json_path = args.output_json

    gen_captions(image_directory, output_json_path)