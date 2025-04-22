from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

# Serve static files (images)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/list-images", response_class=HTMLResponse)
def list_images(request: Request, page: int = 1, per_page: int = 10):
    image_dir = "static/images"
    all_images = sorted(os.listdir(image_dir))
    total_pages = (len(all_images) + per_page - 1) // per_page

    # Pagination logic
    start = (page - 1) * per_page
    end = start + per_page
    page_images = all_images[start:end]
    image_paths = [f"/static/images/{file}" for file in page_images]

    return templates.TemplateResponse("image_list.html", {
        "request": request,
        "images": image_paths,
        "page": page,
        "total_pages": total_pages
    })
