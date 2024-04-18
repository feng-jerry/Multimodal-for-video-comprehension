import os
import torch
from transformers import Wav2Vec2Processor, HubertModel
import torchaudio
from moviepy.editor import VideoFileClip, vfx

VIDEO_FOLDER_PATH = "./video"
FEATURE_PATH = "./features"
CHUNK_SIZE = 60 * 16000  # 1 second of audio at 16kHz, 60 seconds

video_files = []

for entry in os.listdir(VIDEO_FOLDER_PATH):
    video_files.append(entry)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(device)


def load_audio(video_path):
    video_clip = VideoFileClip(os.path.join(VIDEO_FOLDER_PATH, video_path))
    audio_clip = video_clip.audio
    # print(video_clip.reader.nframes)
    new_audio = audio_clip.fx(vfx.speedx, 1 / (0.02002 * video_clip.fps))
    new_audio.write_audiofile('output.wav')
    return video_clip.reader.nframes


for video in video_files:
    frames = load_audio(video)
    audio_input, sample_rate = torchaudio.load("output.wav")

    # Check if the audio is stereo and convert to mono if necessary
    if audio_input.shape[0] > 1:  # More than one channel
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)

    # Resample the audio file if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio_input = resampler(audio_input)

    num_chunks = len(audio_input[0]) // CHUNK_SIZE

    # Process and extract features for each chunk
    features_list = []
    for i in range(num_chunks + 1):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = audio_input[:, start:end].squeeze()

        if chunk.nelement() == 0:  # Skip empty chunks
            continue

        # Move the chunk to the chosen device
        chunk = chunk.to(device)

        # Move input_values to the chosen device as well
        input_values = processor(chunk, return_tensors="pt", sampling_rate=16000).input_values.to(device)
        # input_values = input_values.squeeze(1)  # Remove the extra dimension

        # Extract features with model in evaluation mode (no gradients)
        with torch.no_grad():
            chunk_features = model(input_values).last_hidden_state
            features_list.append(chunk_features)

        # Concatenate all features from chunks
        features = torch.cat(features_list, dim=1)
        # print(features.shape)
        features = features[:, :frames, :]

        if not os.path.exists(FEATURE_PATH):
            os.makedirs(FEATURE_PATH)

        # The features can now be used for your classification task
        torch.save(features, os.path.join(FEATURE_PATH, os.path.splitext(video)[0] + ".pt"))
