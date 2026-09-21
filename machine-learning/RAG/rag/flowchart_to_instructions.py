import os
import ollama
import csv
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

"""Locate and load the FlowChart flowchart images for Mermaid conversion.

Paths are resolved relative to this file, not the process working directory,
so imports and `python flowchart_to_mermaid.py` both work from anywhere.
"""


# Path(__file__) is this source file; .resolve() makes it absolute and follows
# symlinks. .parent then drops the filename, leaving .../rag/rag/
PACKAGE_DIR = Path(__file__).resolve().parent

DATASET_DIR = PACKAGE_DIR / "flowcharts" / "FlowChart.v3i.multiclass"
TRAIN_DIR = DATASET_DIR / "train"
CLASSES_CSV = TRAIN_DIR / "_classes.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

def image_paths(split_dir: Path = TRAIN_DIR) -> list[Path]:
    """Absolute paths to every image in a split, sorted by filename."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    return sorted(
        p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )


def image_to_instructions(image_path: str, prompt_text: str):
    # 1. Load the model and processor optimized for CPU
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print("🔄 Loading processor and model (this may take a moment)...")
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load model weights onto CPU. We use float16 or bfloat16 to save RAM if supported, 
    # but torch.float32 is the safest baseline for all standard CPUs.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float32, 
        device_map="cpu"
    )

    # 2. Open the image file
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ Error: Image file not found at {image_path}")
        return

    # 3. Format the chat prompt template expected by Qwen2.5-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    # Apply standard chat formatting template
    text_prompt = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )


    # 4. Preprocess image and text inputs for the CPU
    print("🔄 Preparing visual and text tokens...")
    inputs = processor(
        text=[text_prompt], 
        images=[image], 
        padding=True, 
        return_tensors="pt"
    )

    # Ensure inputs are on the CPU
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    # 5. Generate text output
    print("🚀 Running inference on CPU (generating response)...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=False
        )

    # 6. Trim input tokens away to isolate only the newly generated response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        
        # Decode the numerical token outputs back into readable text string
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        return output_text
    

# --- Demonstration Usage ---
if __name__ == "__main__":
    print(f"dataset:  {DATASET_DIR}")
    print(f"exists:   {DATASET_DIR.is_dir()}")

    images = image_paths()
    print(f"images:   {len(images)}")

    # for path in images[:3]:
    #     print(f"{path.name}")

    IMAGE_FILE = images[0]
    
    # Run the model
    prompt_text = """Convert the flowchart in the image to easy to read
                step by step instructions"""
    mermaid_output = image_to_instructions(IMAGE_FILE, prompt_text)
    
    print("\n🚀 Model Output:")
    print(mermaid_output)