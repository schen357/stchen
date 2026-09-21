import os
import pickle
import torch

from pathlib import Path
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

"""Locate and load the FlowChart flowchart images for Mermaid conversion.

Paths are resolved relative to this file, not the process working directory,
so imports and `python flowchart_to_mermaid.py` both work from anywhere.
"""


# Path(__file__) is this source file; .resolve() makes it absolute and follows
# symlinks. .parent then drops the filename, leaving .../rag/rag/
PACKAGE_DIR = Path(__file__).resolve().parent

DATASET_DIR = PACKAGE_DIR / "flowcharts" / "FlowChart.v3i.multiclass"
TRAIN_DIR = DATASET_DIR / "train"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE = "mps"

def load_model(model_id:str = MODEL_ID):
    
    return model, processor


def image_paths(split_dir: Path = TRAIN_DIR) -> list[Path]:
    """Absolute paths to every image in a split, sorted by filename."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    return sorted(
        p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def image_to_instructions(model, processor, image_filepaths: str, prompt_text: str):
    # 1. Open the image file
    images = []
    for image_path in image_filepaths:
        try:
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_path}")
            return

    # 2. Format the chat prompt template expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # 3. Apply standard chat formatting template
    text_prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )


    # 4. Preprocess image and text inputs for the CPU
    print("Preparing visual and text tokens")
    inputs = processor(
        text=[text_prompt] * len(images), 
        images=images, 
        padding=True, 
        return_tensors="pt"
    )

    # Ensure inputs are on correct device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 5. Generate text output
    print("Running inference")
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=512,
            do_sample=False
        )

    # 6. Trim input tokens away to isolate only the newly generated response
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
        
    # Decode the numerical token outputs back into readable text string
    print("Decoding output to text.")
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )

    return output_text
    

# --- Demonstration Usage ---
if __name__ == "__main__":
    print(f"dataset:  {DATASET_DIR}")
    print(f"exists:   {DATASET_DIR.is_dir()}")

    images = image_paths()
    print(f"images:   {len(images)}")

    image_filepaths = images[:5]
    print("Loading processor and model")
    # Note: passed max_pixels to the processor instantiation correctly
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=256*256)

    # Load model weights onto CPU. We use float16 or bfloat16 to save RAM if supported, 
    # but torch.float32 is the safest baseline for all standard CPUs.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, 
        dtype = torch.float16 ,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    print("Model loaded successfully!")# Run the model

    prompt_text = """Convert the flowchart in the image to easy to follow, specific,
                step by step instructions. If there are decision points where steps differ depending on
                some criteria, make the different procedures and the criteria for following
                such procedures clear."""

    mermaid_output = image_to_instructions(model, processor, image_filepaths, prompt_text)
    
    print("Model Output:")
    print(mermaid_output)

    with open("flowchart_instructions.pkl", "wb") as f:
        pickle.dump(mermaid_output, f, protocol=pickle.HIGHEST_PROTOCOL)

    