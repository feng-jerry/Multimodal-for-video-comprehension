import cv2
import numpy as np
import os
import torch
import timm

if torch.backends.mps.is_available():
    device = torch.device("mps")
    x = torch.ones(1, device=device)
    print (x)
else:
    print ("MPS device not found.")

model = timm.create_model('vit_base_patch16_224', pretrained=True)
model = model.to(device)
model.eval()

def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.tensor(frame / 255.0, dtype=torch.float32)
    frame = (frame - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    frame = frame.permute(2, 0, 1).unsqueeze(0)  # Reshape to [1, 3, 224, 224]
    return frame

def generate_feature(filename):
    cap = cv2.VideoCapture('/Volumes/T7T/HW/CSCI535/project/Aff-Wild2/video/'+filename)
    frame_index = 0
    all_features=[]
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Preprocess the frame
            input_tensor = preprocess_frame(frame)

            # Move tensor to GPU if available
            if torch.backends.mps.is_available():
                input_tensor = input_tensor.to(device)

            # Extract features
            with torch.no_grad():
                features = model.forward_features(input_tensor)
            
            all_features.append(features.cpu().numpy())
            # Save features to a file
            frame_index += 1
        else:
            break
            
    feature_path = os.path.join(output_dir, f'{filename.split(".")[0]}.npy')
    np.save(feature_path, np.array(all_features))    
    cap.release()

output_dir = 'output_features'
os.makedirs(output_dir, exist_ok=True)
# Path to the directory you want to list files from
folder_path = '/Volumes/T7T/HW/CSCI535/project/Aff-Wild2/video'

# List all files in the directory
file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

for i, filename in enumerate(file_names):
    print(f'Starting generating feature for the {i}th video: ', filename)
    generate_feature(filename)




