import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from utils import CUE_LIST

# Prepare data
def prepare_data(video_folder, annotation_file):
    X = []
    y = []
    groups = []

    # Load annotation file
    excel_file = pd.ExcelFile(annotation_file)
    # Load each sheet as df
    for sheet_name in excel_file.sheet_names:
        if sheet_name == "Overview": # skip this sheet
            continue
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        video_file = os.path.join(video_folder, f"{sheet_name}.mp4")

        # loop over instances in df
        for idx, row in df.iterrows():
            frame = extract_frame(video_file, row['Selected_Frame_Time'])
            label_names = [item.strip() for item in row["Total_Cues"].split(",")]
            label_vector = create_multilabel(label_names)
            X.append(frame)
            y.append(label_vector)
            print(label_vector)
            groups.append(sheet_name)

    # Convert to numpy arrays
    X = np.array(X, dtype=object)    # dtype=object because frames might have different shapes
    y = np.array(y, dtype=np.int32)  # or np.int32 depending on your labels
    groups = np.array(groups)

    return X, y, groups

def extract_frame(video_file: str, frame_time: float) -> Image.Image:
    # Read video, Find frame, Convert to PIL, Extract
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Frame extraction failed at {frame_time}s in {video_file}")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return  frame_rgb

def create_multilabel(labels):
    vec = np.zeros(len(CUE_LIST))
    for label in labels:
        if label in CUE_LIST:
            vec[CUE_LIST.index(label)] = 1
    return vec