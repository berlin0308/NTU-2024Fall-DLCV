### Installation

You can run the following commands to install all the packages listed in the `environment.yaml`:

```sh
cd stable-diffusion
conda env create -f environment.yaml
conda activate ldm
```

To install the stable-diffusion v1.4. You can download the sd-v1-4.ckpt file from [hugging face link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/tree/main) or use wget

```sh
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

place the model checkpoint in the following path:

```sh
stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
```

You can run the following command to test if you successfully build the environment
```sh
python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse"
```

(Optional)
You can also install the environment with pip and requirements.txt, note that the version of pytorch should match your CUDA version. [pytorch CUDA version](https://pytorch.org/get-started/previous-versions/)
```sh
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
cd stable-diffusion
pip install -e .
```

> :warning: **Note**: Using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.

> :warning: **Important**: You can **NOT** use **diffuser** for easy implementation!

## Evaluation
We provide the code for evaluation of hw2_3
    
    python evaluation/grade_hw2_3.py --json_path hw2_data/textual_inversion/input.json --input_dir hw2_data/textual_inversion --output_dir output_folder_example 

## Directory Structure

The directory structure is organized as follows:
```
.
└── output folder/
    ├── 0/
    │   ├── 0/
    │   │   ├── source0_prompt0_0.png
    │   │   ├── ...
    │   │   └── source0_prompt0_25.png
    │   └── 1/
    │       ├── source0_prompt1_0.png
    │       ├── ...
    │       └── source0_prompt0_25.png
    └── 1/
        ├── 0/
        │   ├── source1_prompt0_0.png
        │   └── ...
        └── 1/
            ├── ...
            └── source1_prompt1_25.png
```

- **First Level**: Indices of concepts.
- **Second Level**: Indices of prompts.

In the private set, we will replace the `prompt` and `prompt_4_clip_eval` in `input.json`.
