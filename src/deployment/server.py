"""server.py contains the functions used in the fastapi server"""

import io

# import shutil
from pathlib import Path
from typing import Dict

import fastapi
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from main import infer_caption_from_image, load_models
from PIL import Image
from src.utils import API_HOST, API_PORT, IS_GPU

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
app.mount(
    "/static", StaticFiles(directory=str(Path(BASE_DIR, "static"))), name="static"
)

device = "cuda" if IS_GPU else "cpu"
model_path = "../../pretrained_models/model_weights.pt"  # Update with your model path
model, tokenizer, clip_model, preprocess = load_models()

templates = Jinja2Templates(directory=str(Path(BASE_DIR, "templates")))


@app.get("/", response_class=HTMLResponse)
async def read_root(request: fastapi.Request) -> HTMLResponse:
    """
    Read the HTML home page.

    Args:
        request (fastapi.Request): The incoming FastAPI request.

    Returns:
        HTMLResponse: The HTML response for the home page.
    """
    return templates.TemplateResponse(
        "upload.html", {"request": request, "port": API_PORT, "host": API_HOST}
    )
    # with open("templates/upload.html") as f:
    #     return HTMLResponse(content=f.read(), status_code=200)


@app.post("/upload-image", status_code=fastapi.status.HTTP_200_OK)
async def upload_image(
    image: UploadFile = File(...),
    # model_path: str = Form(...),
    # prompt: str = Form("A photo of"),
    # use_beam_search: bool = Form(False),
) -> Dict[str, str]:
    """Post method when image is uploaded

    Parameters
    ----------
    image : UploadFile

    Returns
    -------
    Dict[str, str]
        a dictionary containing caption of the image
    """
    image_bytes = await image.read()
    # Open the image with PIL
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert the PIL image to the format needed for your model
    caption = infer_caption_from_image(pil_image)

    return {"caption": caption}


if __name__ == "__main__":

    uvicorn.run(app, host=API_HOST, port=API_PORT)
