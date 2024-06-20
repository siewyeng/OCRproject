import argparse
import os

import clip
import torch
import torch.nn.functional as nnf
from PIL import Image
from src.clip.caption_generation import generate2, generate_beam
from src.clip.model import ClipCaptionModel
from src.utils import *
from tqdm import trange
from transformers import GPT2Tokenizer


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f"cuda:{device_id}")


CUDA = get_device

model_path = "pretrained_models/model_weights.pt"
device = CUDA(0) if is_gpu else "cpu"


def load_models():
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = ClipCaptionModel(prefix_len)

    model.load_state_dict(
        torch.load(model_path, map_location=CPU), strict=False
    )  # already downloaded weights into folder

    model = model.eval()  # do clip caption in eval mode
    model = model.to(device)
    return model, tokenizer, clip_model, preprocess


def process_image(image_path, preprocess, device):
    pil_image = Image.open(image_path)
    pil_image.show()
    image = preprocess(pil_image).unsqueeze(0).to(device)
    return image


# used by server.py
def infer_caption_from_image(pil_image: Image.Image) -> str:
    model, tokenizer, clip_model, preprocess = load_models()
    # Preprocess the image
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_len, -1)

        # Generate caption (you can use beam search or other methods as needed)
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    return generated_text_prefix


def main(args):
    #     image_path = "memes/210513.png"
    # # @title Inference
    use_beam_search = False  # @param {type:"boolean"}
    model, tokenizer, clip_model, preprocess = load_models()
    image = process_image(args.image_path, preprocess, device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_len, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    print("\n")
    print(generated_text_prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CLIP + GPT-2 Inference")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file"
    )
    # parser.add_argument('--prompt', type=str, default='A photo of', help='Text prompt to prepend to the generated text')
    # parser.add_argument(
    #     "--use_beam_search", action="store_true", help="Use beam search for generation"
    # )

    args = parser.parse_args()
    main(args)
