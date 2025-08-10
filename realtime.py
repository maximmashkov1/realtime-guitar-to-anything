import numpy as np
import sounddevice as sd
import torch
from model import GuitarToneCloning
import torch.nn.functional as F
from utils import mel_spectrogram, NoiseReducer
import os
import psutil

p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)


class DummyClass:
    a=1

model = torch.load('backup_24k',weights_only=False).eval().cuda()
model.vocoder.turn_on_caching()
model.flow.turn_on_caching()



sample_rate = 24000
buffer_size = 1024
frame_size = 256
nr=NoiseReducer(sample_rate)
print("devices:")
for i, dev in enumerate(sd.query_devices()):
    print(f"{i}: {dev['name']}")

input_device = int(input("Select input device: "))
output_device = int(input("Select output device: "))

input_info = sd.query_devices(input_device)
output_info = sd.query_devices(output_device)

buffer = torch.tensor([],device='cuda')
def audio_callback(indata, outdata, frames, time, status):
    global buffer
    if status:
        print(status)


    mono_input = indata[:, 0] * 100
    mono_input = torch.tensor(mono_input,device='cuda')
    mono_input = torch.clip(mono_input, -1.0, 1.0)
    buffer = torch.cat((buffer,mono_input))[-buffer_size:]

    if buffer.shape[-1] == buffer_size:
        mel_chunk = mel_spectrogram(buffer.unsqueeze(0),buffer_size,win_size=buffer_size)
        with torch.inference_mode():
            reconstructed = model(mel_chunk.flatten()).cpu().numpy().flatten()

        outdata[:, 0] = reconstructed


try:
    with sd.Stream(callback=audio_callback,
                  samplerate=sample_rate,
                  blocksize=frame_size,
                  dtype='float32',
                  channels=1,
                  device=(input_device, output_device),
                  latency=2e-2):
        while True:
            sd.sleep(1000)
except Exception as e:
    print("Error:", e)