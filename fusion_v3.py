#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import torch
import timm
import torch.nn as nn
from transformers import Wav2Vec2Processor, HubertModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
import math



device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
device


def seq_feature_generation(video_feature, audio_feature, seq_len, pooling = "mean"):
    #video_feature : (771, 1, 197, 768)
    #audio_feature : [1, 773, 1024]
    video_feature = torch.tensor(video_feature, dtype=torch.float32).to(device)
    audio_feature = torch.tensor(audio_feature, dtype=torch.float32).to(device)

    video_feature = video_feature.permute(1,0,2,3)
    
    if pooling == "mean":
        video_feature = torch.mean(video_feature, dim = 2, keepdim=False)
    elif pooling == "max":
        video_feature = torch.max(video_feature, dim = 2, keepdim=False)[0]

    max_seq = min(video_feature.shape[1], audio_feature.shape[1])
    video_feature = video_feature[:, :max_seq, :]
    audio_feature = audio_feature[:, :max_seq, :]
    combined_feature = torch.cat([video_feature, audio_feature], dim = -1)
    #[1, max_seq, 1024 + 768]
    
    if max_seq < seq_len:
        # Pad both features to seq_len along the sequence dimension
        combined_sequences = F.pad(combined_feature, (0, 0, 0, seq_len - max_seq))
    else:
        num_complete_seqs = max_seq // seq_len
        combined_sequences = combined_feature[:,:num_complete_seqs*seq_len, :].view(-1, seq_len, combined_feature.shape[-1])
    #[-1, seq_len, combined_feature_size]
    return combined_sequences


class ViTHuBERTTransformer(nn.Module):
    def __init__(self, vit_base_model,
                 hubert_base_model,
                 num_classes,
                 nhead,
                 num_layers,
                small_dataset = True):
        super().__init__()

        self.vit = timm.create_model(vit_base_model, pretrained=True)

        #self.processor = Wav2Vec2Processor.from_pretrained(hubert_base_model)
        self.hubert = HubertModel.from_pretrained(hubert_base_model)

        if small_dataset:
            for param in self.vit.parameters():
                param.requires_grad = False
        
            for param in self.hubert.parameters():
                param.requires_grad = False
            

        encoder_layer = nn.TransformerEncoderLayer(d_model = self.vit.num_features + self.hubert.config.hidden_size,
                                                  nhead = nhead,
                                                  dim_feedforward = (self.vit.num_features + self.hubert.config.hidden_size)//2,
                                                  batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        # Classifier
        self.classifier = nn.Linear(self.vit.num_features + self.hubert.config.hidden_size, num_classes)
    def forward(self, video_feature_raw, audio_feature_raw):

        vit_feature = self.vit.forward_features(video_feature_raw)
        audio_feature = self.hubert(audio_feature_raw).last_hidden_state
        
        # Combine features
        combined_features = torch.cat((vit_feature, audio_feature), dim=1)

        transformer_output = self.transformer_encoder(combined_features)

        logits = self.classifier(transformer_output.squeeze(1))
        return logits


class ViTHuBERTTransformer_prepossed(nn.Module):
    def __init__(self, vit_base_model,
                 hubert_base_model,
                 num_classes,
                 nhead,
                 num_layers,
                small_dataset = True):
        super().__init__()

        self.vit = timm.create_model(vit_base_model, pretrained=True)

        #self.processor = Wav2Vec2Processor.from_pretrained(hubert_base_model)
        self.hubert = HubertModel.from_pretrained(hubert_base_model)

        if small_dataset:
            for param in self.vit.parameters():
                param.requires_grad = False
        
            for param in self.hubert.parameters():
                param.requires_grad = False
            

        encoder_layer = nn.TransformerEncoderLayer(d_model = self.vit.num_features + self.hubert.config.hidden_size,
                                                  nhead = nhead,
                                                  dim_feedforward = (self.vit.num_features + self.hubert.config.hidden_size)//2,
                                                  batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = num_layers)

        # Classifier
        self.classifier = nn.Linear(self.vit.num_features + self.hubert.config.hidden_size, num_classes)
    def forward(self, combined_feature):

        transformer_output = self.transformer_encoder(combined_feature)

        logits = self.classifier(transformer_output.squeeze(1))
        return logits


model = ViTHuBERTTransformer_prepossed(
    vit_base_model = 'vit_base_patch16_224',
    hubert_base_model = "facebook/hubert-large-ls960-ft",
    num_classes = 12,
    nhead = 8,
    num_layers = 6,
    small_dataset = True
)


model.to(device)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    loss_cumulative = 0.
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.squeeze().to(device), labels.squeeze().to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels).cpu()
            loss_cumulative = loss_cumulative + loss.detach().item()
    return loss_cumulative / len(dataloader)


def loglinspace(rate, step, end=None):
    t = 0
    while end is None or t <= end:
        yield t
        t = int(t + 1 + step * (1 - math.exp(-t * rate / step)))


def train(model, optimizer, dataloader_train, dataloader_valid, loss_fn,
             max_iter=101, scheduler=None, device="cpu"):
    model.to(device = device, dtype=torch.float32)
    print(device)
    checkpoint_generator = loglinspace(0.3, 5)
    checkpoint = next(checkpoint_generator)
    start_time = time.time()
    run_name = "vithubertformer"
    try:
        model.load_state_dict(torch.load(run_name + '.torch')['state'])
    except:
        results = {}
        history = []
        s0 = 0
    else:
        results = torch.load(run_name + '.torch')
        history = results['history']
        s0 = history[-1]['step'] + 1
        
    
    for step in range(max_iter):
        model.train()
        loss_cumulative = 0.
        all_labels = []
        all_preds = []

        for inputs, labels in tqdm(dataloader_train, desc="Training"):
            inputs, labels = inputs.squeeze().to(device), labels.squeeze().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_cumulative += loss.item()
            
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).int()            
            all_labels.append(labels.cpu().numpy().reshape(-1, 12))
            all_preds.append(preds.cpu().numpy().reshape(-1, 12))

        all_labels = np.vstack(all_labels)
        all_preds = np.vstack(all_preds)   

        train_f1_score = f1_score(all_labels, all_preds, average='macro')
        model.eval()
        val_labels = []
        val_preds = []
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader_valid, desc="Validation"):
                # print('valid')
                inputs, labels = inputs.squeeze().to(device), labels.squeeze().to(device)
                print(inputs.shape, labels.shape)
                outputs = model(inputs)
                preds = torch.sigmoid(outputs)
                preds = (outputs > 0.5).int()
                val_labels.append(labels.cpu().numpy().reshape(-1, 12))
                val_preds.append(preds.cpu().numpy().reshape(-1, 12))

        val_labels = np.vstack(val_labels)
        val_preds = np.vstack(val_preds)
        val_f1_score = f1_score(val_labels, val_preds, average='macro')
        print(f'epoch {step+1} train_f1_score:', np.round(train_f1_score,4), 'val_f1_score:', np.round(val_f1_score,4))
        
        wall = time.time() - start_time
        if step == checkpoint:
            checkpoint = next(checkpoint_generator)
            assert checkpoint > step

            valid_avg_loss = evaluate(model, dataloader_valid, loss_fn, device)
            train_avg_loss = evaluate(model, dataloader_train, loss_fn, device)
            history.append({
                'step': s0 + step,
                'wall': wall,
                'batch': {
                    'loss': loss.item(),
                },
                'valid': {
                    'loss': valid_avg_loss,
                },
                'train': {
                    'loss': train_avg_loss,
                },
            })

            results = {
                'history': history,
                'state': model.state_dict()
            }

            print(f"epoch {step + 1:4d}   " +
                  f"abs = {train_avg_loss:8.4f}   " +
                  f"valid loss mse= {valid_avg_loss:8.4f}   " +
                  f"wall = {time.strftime('%H:%M:%S', time.gmtime(wall))}")

            with open(run_name + '.torch', 'wb') as f:
                torch.save(results, f)

        if scheduler is not None:
            scheduler.step()


class AudioVideoDataset(Dataset):
    def __init__(self, video_dir, audio_dir, label_dir):
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.label_dir = label_dir

        # Collect all label files, and construct corresponding video and audio file paths
        self.entries = []
        for label_file in sorted(os.listdir(label_dir)):
            if label_file.endswith('.txt'):
                base_name = os.path.splitext(label_file)[0]
                video_file = os.path.join(video_dir, f"{base_name}.npy")
                audio_file = os.path.join(audio_dir, f"{base_name}.pt")
                label_file_path = os.path.join(label_dir, label_file)
                
                # Add entry only if corresponding video and audio files exist
                if os.path.exists(video_file) and os.path.exists(audio_file):
                    self.entries.append((video_file, audio_file, label_file_path))
                else:
                    print(f"Missing video or audio file for {label_file}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        video_file, audio_file, label_file = self.entries[idx]
        video_feature = np.load(video_file)
        audio_feature = torch.load(audio_file)
        labels = np.loadtxt(label_file, skiprows=1, delimiter=',')


        seq_len = 10  # Define the desired sequence length
        min_len = min(len(labels), video_feature.shape[0], audio_feature.shape[1])

        # Further truncate data to the minimum length across modalities
        labels = labels[:min_len, :]
        video_feature = video_feature[:min_len, :, :, :]
        audio_feature = audio_feature[:, :min_len, :]

        # Find indices where all labels are binary
        binary_indices = np.all(np.isin(labels, [0, 1]), axis=1)
        # Filter out non-binary frames
        labels = labels[binary_indices, :]
        video_feature = video_feature[binary_indices, :, :, :]
        audio_feature = audio_feature[:, binary_indices, :]  # Adjust this if necessary
        
        combined_features = seq_feature_generation(video_feature, audio_feature, seq_len)  # Adjust device as needed

        label_sequences = labels[:combined_features.shape[0] * seq_len].reshape(-1, seq_len, 12)

        return combined_features, label_sequences


train_label_dir = '/project/msoleyma_1026/Aff-Wild2/labels/AU_Detection_Challenge/Train_Set'
audio_feature_dir = '/project/msoleyma_1026/Aff-Wild2/features4'
video_feature_dir = '/project/msoleyma_1026/Aff-Wild2/video_feature'
dataset_train = AudioVideoDataset(video_feature_dir, audio_feature_dir, train_label_dir)
dataloader_train = DataLoader(dataset_train, batch_size=1, shuffle=True)


val_label_dir = '/project/msoleyma_1026/Aff-Wild2/labels/AU_Detection_Challenge/Validation_Set'
dataset_val = AudioVideoDataset(video_feature_dir, audio_feature_dir, val_label_dir)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True)



loss_function = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
train(model, opt, dataloader_train, dataloader_val, loss_function,
             max_iter=10, scheduler=scheduler, device=device)


