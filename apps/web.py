import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# Import MediaMTX API client

import mediamtx_client
from mediamtx_client.rest import ApiException
 # Defining the host is optional and defaults to http://localhost:9997
# See configuration.py for a list of all supported configuration parameters.
mediamtx_conf = mediamtx_client.Configuration(
    host = "http://localhost:9997"
)


app = FastAPI()

# Serve static files (images)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

api_client = mediamtx_client.ApiClient(mediamtx_conf)

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/list-streams", response_class=HTMLResponse)
async def list_streams(request: Request):
    try:
        api_instance = mediamtx_client.PathsApi(api_client)
        streams = api_instance.paths_list()  # Assuming there's a method to list streams
        stream_info = [{"id": "Test", "name": "Name"}]
        return templates.TemplateResponse("streams_list.html", {"request": request, "streams": stream_info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching streams: {str(e)}")

@app.get("/list-config", response_class=HTMLResponse)
async def list_config(request: Request):
    try:
        config_data = api_instance.config_global_get()
        config_info = {
            "version": config_data.version,
            "hostname": config_data.hostname,
            "port": config_data.port
        }
        return templates.TemplateResponse("config_list.html", {"request": request, "config": config_info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching configuration: {str(e)}")

# Models for API interaction (just for illustration)
class Stream(BaseModel):
    id: str
    name: str
    status: str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)