import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize_word_attention(attentions, output_words, output_path, image_path):
    """
    Visualizes the attention map for each word in the generated caption over the input image.

    Parameters:
    - attentions (tuple): Tuple of torch.FloatTensor with attention weights (one for each layer).
    - output_words (list): List of generated words to visualize attention for.
    - output_path (str): Path to save the attention overlay output.
    - image_path (str): Path to the original image.
    - image_token_count (int): Number of tokens corresponding to the image in the attention map.
    """
    
    # Check attentions structure
    print(f"Total layers in attentions: {len(attentions)}")
    print(f"Shape of each layer's attention on CPU: {attentions[0].shape} (expected: [batch_size, num_heads, seq_len, seq_len])")
    
    # Move attentions to CPU and stack them for aggregation
    attentions_cpu = [att.cpu() for att in attentions]
    
    # Average over all layers and heads to get a single (seq_len, seq_len) attention map
    avg_attention = torch.mean(torch.stack(attentions_cpu), dim=(0, 1)).squeeze().detach().numpy()  # Shape should be (seq_len, seq_len)
    print(f"Shape after averaging over layers and heads: {avg_attention.shape}")

    seq_length = avg_attention.shape[0]
    text_token_count = len(output_words)
    image_token_count = seq_length - text_token_count
    print(f"Calculated image token count: {image_token_count}")

    # Ensure avg_attention is now (seq_len, seq_len)
    if avg_attention.ndim != 2:
        print(f"Unexpected avg_attention shape after averaging: {avg_attention.shape}")
        return  # Early exit if shape is not as expected
    
    # Load and prepare the image for overlay
    original_image = Image.open(image_path).convert("RGB")
    
    # Visualize attention for each word
    for idx, word in enumerate(output_words):
        token_idx = idx + image_token_count
        if token_idx >= avg_attention.shape[0]:
            print(f"Skipping word '{word}' due to index out of bounds.")
            continue
        
        # Extract attention map for the specific word token
        word_attention = avg_attention[token_idx, :image_token_count]
        print(f"Attention shape for word '{word}' (token index {token_idx}): {word_attention.shape}")
        
        # Reshape attention map to approximate image shape
        height = int(np.sqrt(len(word_attention)))
        width = len(word_attention) // height
        if height * width != len(word_attention):
            width += 1
        word_attention_rescaled = np.uint8(word_attention / word_attention.max() * 255).reshape((height, width))
        
        # Display debugging info
        print(f"Reshaped word attention for '{word}' to: {word_attention_rescaled.shape}")
        
        # Plot the attention overlay on the image
        plt.figure(figsize=(8, 8))
        plt.imshow(original_image, alpha=0.6)
        plt.imshow(word_attention_rescaled, cmap="viridis", alpha=0.4)
        plt.colorbar()
        plt.title(f"Attention for '{word}'")
        plt.axis("off")
        
        # Save the visualization for each word
        word_output_path = f"{output_path}_{word}.png"
        plt.savefig(word_output_path)
        plt.show()
        print(f"Saved attention visualization for '{word}' at {word_output_path}")
