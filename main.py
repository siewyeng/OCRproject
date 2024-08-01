"""main.py is how this whole project can be called from the cmd and also contains main function for server"""

import argparse

import clip
import torch

# import torch.nn.functional as nnf
from PIL import Image
from src.clip.caption_generation import generate2, generate_beam
from src.clip.model import ClipCaptionModel
from src.utils import CPU, IS_GPU, PREFIX_LEN, D

# from tqdm import trange
from transformers import GPT2Tokenizer

# import os


def get_device(device_id: int) -> D:
    """To retrieve device type and id if GPU"""
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f"cuda:{device_id}")


CUDA = get_device

MODEL_PATH = "pretrained_models/model_weights.pt"
DEVICE = CUDA(0) if IS_GPU else "cpu"


def load_models():
    """Loads models to be used in main function

    Returns
    -------
    loaded pretrained model, gpt2's tokenizer,
    clip model and the image preprocessing for the clip model
    """
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = ClipCaptionModel(PREFIX_LEN)

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=CPU), strict=False
    )  # already downloaded weights into folder

    model = model.eval()  # do clip caption in eval mode
    model = model.to(DEVICE)
    return model, tokenizer, clip_model, preprocess


def process_image(image_path, preprocess, device):
    """Preprocesses image based on model

    Parameters
    ----------
    image_path : str
    preprocess
    device : D

    Returns
    -------
       preprocessed image
    """
    pil_image = Image.open(image_path)
    pil_image.show()
    image = preprocess(pil_image).unsqueeze(0).to(device)
    return image


# used by server.py
def infer_caption_from_image(pil_image: Image.Image) -> str:
    """Infer function used by server"""
    model, tokenizer, clip_model, preprocess = load_models()
    # Preprocess the image
    image = preprocess(pil_image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(DEVICE, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, PREFIX_LEN, -1)

        # Generate caption (you can use beam search or other methods as needed)
        generated_text_prefix = generate2(
            model, tokenizer, embed=prefix_embed, prompt=CUSTOM_PROMPT
        )

    return generated_text_prefix


def main(args):
    """main function to be carried out
    preprocesses image and generates caption

    Parameters
    ----------
    args : arguments specified in command
    """
    #     image_path = "memes/210513.png"
    # # @title Inference
    use_beam_search = False  # @param {type:"boolean"}
    model, tokenizer, clip_model, preprocess = load_models()
    image = process_image(args.image_path, preprocess, DEVICE)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(DEVICE, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(
            1, PREFIX_LEN, -1
        )  # projects the CLIP embedding into a dimension suitable for GPT2
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
