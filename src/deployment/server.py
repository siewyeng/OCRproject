import io
import shutil
from pathlib import Path
from typing import Dict

import fastapi
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from main import infer_caption_from_image, load_models
from PIL import Image
from src.utils import api_host, api_port, is_gpu

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
app.mount(
    "/static", StaticFiles(directory=str(Path(BASE_DIR, "static"))), name="static"
)

device = "cuda" if is_gpu else "cpu"
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
        "upload.html", {"request": request, "port": api_port, "host": api_host}
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
    image_bytes = await image.read()
    # Open the image with PIL
    pil_image = Image.open(io.BytesIO(image_bytes))

    # Convert the PIL image to the format needed for your model
    caption = infer_caption_from_image(pil_image)

    return {"caption": caption}


if __name__ == "__main__":

    uvicorn.run(app, host=api_host, port=api_port)
