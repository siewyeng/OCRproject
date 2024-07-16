import argparse
import os

import clip
import torch
import torch.nn.functional as nnf
from PIL import Image
from src.clip.caption_generation import generate2, generate_beam
from src.clip.model import ClipCaptionModel, ClipCaptionPrefix
from src.utils import *
from tqdm import trange
from transformers import GPT2Tokenizer


def get_device(device_id: int) -> D:
    """Function used to determind if GPU can be used

    Parameters
    ----------
    device_id : int

    Returns
    -------
    torch device
    """
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f"cuda:{device_id}")


CUDA = get_device

device = CUDA(0) if is_gpu else "cpu"


def load_models(mode="transformer"):
    """Loads the appropriate pretrained model from the model weights
    in the directory

    Parameters
    ----------
    mode : str, optional
        the type of model which can also be "MLP", by default "transformer"

    Returns
    -------
    model: the pretrained model
    tokenizer: gpt2 tokenizer
    clip_model: clip model (in this case RN50x4)
    preprocess: the method used by clip to preprocess the image before
        sending it to the model
    """
    clip_model, preprocess = clip.load("RN50x4", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    if mode == "MLP":
        model_path = "pretrained_models/model_weights.pt"
        model = ClipCaptionModel(prefix_len)

        model.load_state_dict(
            torch.load(model_path, map_location=CPU), strict=False
        )  # already downloaded weights into folder
    else:
        model_path = "pretrained_models/transformer_weights.pt"
        model = ClipCaptionPrefix(
            prefix_length=40,
            clip_length=40,
            prefix_size=640,
            num_layers=8,
            mapping_type="transformer",
        )
        model.load_state_dict(torch.load(model_path, map_location=CPU))
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
    """Generates text caption of image based on the components generated
    by load_models. This is used by the upload_image in server.py

    Parameters
    ----------
    pil_image : Image.Image
        the image to generate caption for

    Returns
    -------
    str
        caption
    """
    model, tokenizer, clip_model, preprocess = load_models()
    # Preprocess the image
    image = preprocess(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_len, -1)

        # Generate caption (you can use beam search or other methods as needed)
        generated_text_prefix = generate2(
            model, tokenizer, embed=prefix_embed, prompt=custom_prompt
        )

    return generated_text_prefix


def main(args):
    """Carries out the image caption generationg in the command line

    Parameters
    ----------
    args : arguments include image_path
    """
    #     image_path = "memes/210513.png"
    # # @title Inference
    use_beam_search = False  # @param {type:"boolean"}
    model, tokenizer, clip_model, preprocess = load_models()
    image = process_image(args.image_path, preprocess, device)

    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(
            1, prefix_len, -1
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
