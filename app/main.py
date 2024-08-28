from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import cv2

import torch
import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    Resize,
)
from torch.utils.data import DataLoader
app = FastAPI()

model_path = r'C:\Users\acer\OneDrive\Desktop\CCTV_Surveillance\app\model3.pt'
model = torch.load(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resize_to = (224, 224)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
val_transform = Compose(
    [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(32),
                    Lambda(lambda x: x / 255.0),
                    Normalize(mean, std),
                    Resize(resize_to),
                ]
            ),
        ),
    ]
)

def get_video(video_path):
    labeled_video_paths = [(video_path, {'label': 2})]

    data = pytorchvideo.data.LabeledVideoDataset(
        labeled_video_paths =labeled_video_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", 8),
        decode_audio=False,
        transform=val_transform,
    )

    loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=True)
    video_data = next(iter(loader))
    
    video = video_data['video'].squeeze(0).to(device)
    return video

def run_inference(model, video_path):
    video = get_video(video_path)
    perumuted_sample_test_video = video.permute(1, 0, 2, 3)

    inputs = {
        "pixel_values": perumuted_sample_test_video.unsqueeze(0),
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    return logits

class VideoPath(BaseModel):
    video_path: str

@app.post("/predict/")
async def predict(video_path: VideoPath):
    if not os.path.exists(video_path.video_path):
        raise HTTPException(status_code=404, detail="Video file not found.")
    
    video_path = video_path.video_path

    resized_video_path = os.path.join('resized_uploaded_videos', video_path.split('\\')[-1].split('.')[-2]) + '.avi'
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(resized_video_path,fourcc, fps, (640,360))
    while True:
        ret, frame = cap.read()
        if ret == True:
            b = cv2.resize(frame,(640,360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            out.write(b)
        else:
            break 
    cap.release()
    out.release()
    cv2.destroyAllWindows()                                         


    logits = run_inference(model, resized_video_path)
    label = model.config.id2label[logits.argmax().item()]
    return {"prediction": label}