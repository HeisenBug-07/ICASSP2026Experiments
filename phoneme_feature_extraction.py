import glob
import random

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch.nn.functional as F
import torchaudio
import torch
import numpy as np
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"  
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name, use_safetensors=True)
model.to(device)

def audio_to_phone(path):
    waveform, sample_rate = torchaudio.load(path)
    
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    inputs = processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt")
    
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    phonemes = processor.batch_decode(predicted_ids, group_tokens=True)

    logits_squeezed = logits.squeeze(0).cpu()
    logit_trans = logits_squeezed.transpose(0, 1)
    numpy_array = logit_trans.numpy()
    return numpy_array

def extract_and_save_features(wav_files, output_folder, feature_list_filename, name):
    os.makedirs(output_folder, exist_ok=True)

    label_map = {
        'asm': '0', 'ben': '1', 'eng': '2', 'guj': '3',
        'hin': '4', 'kan': '5', 'mal': '6', 'mar': '7',
        'odi': '8', 'pun': '9', 'tam': '10', 'tel': '11'
    }
    label = label_map.get(name)
    
    if wav_files.lower().endswith('.wav'):
        base_name = os.path.basename(wav_files)
        feature_filename = os.path.splitext(base_name)[0] + ".npy"
        feature_path = os.path.join(output_folder, feature_filename)

        # Skip if feature already exists
        if os.path.exists(feature_path):
            #print(f"Skipping {wav_files}, feature already exists.")
            return

        feature_np = audio_to_phone(wav_files)
        np.save(feature_path, feature_np)

        with open(feature_list_filename, "a") as list_file:
            list_file.write(f"{feature_path} {label}\n")

# Combine everything if needed
all_train_files = glob.glob("../audio/ekstep_seen/**/**/*wav")
print("Total files:", len(all_train_files))

for wav_file in tqdm(all_train_files):
    name = wav_file.split('/')[4]
    if name not in ['asm', 'ben', 'eng', 'guj','hin', 'kan', 'mal', 'mar', 'odi', 'pun', 'tam', 'tel'] :
        print("Unexpected name:", name)
    extract_and_save_features(wav_file,"../feature/seen_set_ekstep_phoneme", "seen_set_ekstep_phoneme.txt", name)
