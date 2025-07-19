from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from PIL import Image
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import io
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request


from model_1 import TinyVGG

class_names = ["kabob", "osh", "manti"]

model_path = "TinyVGG_uzfood.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()
app = FastAPI()
templates = Jinja2Templates(directory="templates")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html",{"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(Ellipsis)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        result = class_names[predicted.item()]
    
    return templates.TemplateResponse("index.html",{
        "request": request,
        "result": result
    })