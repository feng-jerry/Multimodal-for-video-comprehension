import torch
from transformers import AutoImageProcessor, VideoMAEModel
from torchvision.io import read_video
import torchvision
import numpy as np
import av
from tqdm import tqdm
import os

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
else:
    print ("MPS device not found.")

def extract_and_pad_frames(video_path, window_size=16):
    container = av.open(video_path)
    frames = [frame.to_rgb().to_ndarray() for frame in container.decode(video=0)]
    # Convert list of frames to a single NumPy array for better performance
    frames_array = np.array(frames)
    # Create padding (using np.zeros_like to match the dimensions and type of frames)
    padded_frames = np.zeros((window_size-1, *frames_array.shape[1:]), dtype=frames_array.dtype)
    # Concatenate padding and frames
    padded_frames = np.concatenate([padded_frames, frames_array], axis=0)
    container.close()
    return padded_frames

def create_overlapping_windows(frames, window_size=16):
    # This should slide the window one frame at a time for overlapping sequences
    stride = frames.strides[0]
    return np.lib.stride_tricks.as_strided(
        frames, 
        shape=(frames.shape[0] - window_size + 1, window_size, frames.shape[1], frames.shape[2], frames.shape[3]),
        strides=(stride, stride) + frames.strides[1:]
    )

processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
model.eval()
model.to('cpu')

def process_windows(windows, processor, model):
    all_features = []
    # Process each window - assuming windows is a 4D NumPy array [num_windows, window_size, H, W, C]
    for i in tqdm(range(windows.shape[0]), desc='Processing windows'):
#         print(i)
        window = list(windows[i])
        inputs = processor(window, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            features = outputs.last_hidden_state.squeeze(0)  # Assume last_hidden_state is what you need
        all_features.append(features)  # Append features of the last frame in the window
    return torch.stack(all_features)

folder_path = '/Volumes/T7T/HW/CSCI535/project/Aff-Wild2/video'
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i, file in enumerate(file_names):
    print('Starting generating feature for the first video: ', file)
    frames = extract_and_pad_frames(folder_path + '/' + file)
    windows = create_overlapping_windows(frames)
    feature = process_windows(windows, processor, model)
    torch.save(feature, f'features/{file}.pt')


