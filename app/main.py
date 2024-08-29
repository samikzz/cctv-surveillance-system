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
        labeled_video_paths=labeled_video_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", 8),
        decode_audio=False,
        transform=val_transform,
    )

    loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=True)
    video_data = next(iter(loader))
    
    video = video_data['video'].to(device)
    return video

def run_inference(model, video_clip):
    perumuted_sample_test_video = video_clip.permute(0, 2, 1, 3, 4)

    inputs = {
        "pixel_values": perumuted_sample_test_video,
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
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    segment_duration = 5  # 5-second segments

    predictions = []
    for start_time in range(0, int(duration), segment_duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        frames = []
        for _ in range(int(fps * segment_duration)):
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_CUBIC)
            frames.append(resized_frame)
        
        if len(frames) == 0:
            continue

        # Write frames to a temporary video file
        temp_video_path = os.path.join('temp', f'clip_{start_time}.avi')
        os.makedirs('temp', exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (640, 360))
        for frame in frames:
            out.write(frame)
        out.release()

        # Run inference on the video clip
        video = get_video(temp_video_path)
        logits = run_inference(model, video)
        label = model.config.id2label[logits.argmax().item()]
        predictions.append({"start_time": start_time, "end_time": start_time + segment_duration, "prediction": label})
        os.remove(temp_video_path)  # Clean up the temporary video file

    print(predictions)
    cap.release()
    cv2.destroyAllWindows()

    return {"predictions": predictions}
