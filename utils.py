import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn
import numpy as np

class NoiseReducer:
    def __init__(self, sample_rate, chunk_size=256, noise_estimate_frames=50, device='cuda'):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_estimate_frames = noise_estimate_frames
        self.device = device

        self.noise_spectra = []
        self.noise_spectrum = None

    def process_sequence(self, sequence: torch.Tensor, chunk_size=256) -> torch.Tensor:
        result = []
        position = 0
        while position + chunk_size < sequence.shape[-1] - 1:
            chunk = sequence[..., position:position + chunk_size]
            result.append(self.process_chunk(chunk))
            position += chunk_size
        return torch.cat(result, dim=-1)

    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        chunk = chunk.to(self.device)
        spectrum = torch.fft.rfft(chunk)
        mag = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        if len(self.noise_spectra) < self.noise_estimate_frames:
            self.noise_spectra.append(mag)
            self.noise_spectrum = torch.stack(self.noise_spectra).mean(dim=0)
            return chunk

        cleaned_mag = mag - self.noise_spectrum
        cleaned_mag = torch.clamp(cleaned_mag, min=0)

        cleaned_spectrum = cleaned_mag * torch.exp(1j * phase)
        cleaned_chunk = torch.fft.irfft(cleaned_spectrum, n=self.chunk_size)

        return cleaned_chunk
    
mel_basis = {}
hann_window = {}

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def mel_spectrogram(y, n_fft=1024, num_mels=80, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=8000, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    if y.shape[-1] != win_size:
        y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
        y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec.to(torch.float32))
    spec = spectral_normalize_torch(spec)

    return spec


def wave_delta(x):
    return x[1:]-x[:1]

def yin_pitch_sequence(x, sample_rate, min_freq = 65.0, max_freq = 2000.0, threshold = 0.0, window=1024, stride=256):

    pitch_list = []
    current_pos = window
    while current_pos+stride < x.shape[-1] - 1:
        pitch = yin_pitch(x[current_pos-window:current_pos], sample_rate, min_freq, max_freq, threshold)
        pitch_list.append(pitch.item())
        current_pos+=stride
    return torch.tensor(pitch_list,device=x.device), current_pos

def amplitude_sequence(x, window=256, stride=256, start_pos=4096):
    ampl_list = []
    current_pos = start_pos
    while current_pos+stride < x.shape[-1] - 1:
        ampl = amplitude(x[current_pos-window:current_pos])
        ampl_list.append(ampl)
        current_pos+=stride
    return torch.tensor(ampl_list,device=x.device), current_pos

def amplitude(x):
    return torch.mean(torch.abs(x))

def yin_pitch(x, sample_rate, min_freq = 65, max_freq = 2000, threshold = 0.01):

    device = x.device
    frame_size = x.size(-1)
    x = x - x.mean(dim=-1, keepdim=True)
    window = torch.hann_window(frame_size, device=device)
    x = x * window.unsqueeze(0)

    n_fft = 2 ** (int(torch.log2(torch.tensor(frame_size * 2 - 1))) + 1)
    
    X = torch.fft.rfft(x, n=n_fft, dim=-1)
    autocorr = torch.fft.irfft(X * X.conj(), n=n_fft, dim=-1)[..., :frame_size]
    
    energy = torch.cumsum(x**2, dim=-1)
    E1 = energy[..., frame_size-1 - torch.arange(frame_size, device=device)]
    energy_padded = F.pad(energy, (1, 0), value=0.0)
    E2 = energy[..., -1].unsqueeze(-1) - energy_padded[..., :frame_size]
    d = E1 + E2 - 2 * autocorr
    d[..., 0] = 1e-12
    
    cumulative_sum = torch.cumsum(d, dim=-1)
    d_prime = d / (cumulative_sum / (torch.arange(1, frame_size+1, device=device)))
    
    max_lag = min(frame_size-1, int(sample_rate / min_freq))
    min_lag = max(1, int(sample_rate / max_freq))
    valid_range = torch.zeros(frame_size, dtype=torch.bool, device=device)
    valid_range[min_lag:max_lag+1] = True
    
    d_prime_masked = torch.where(valid_range, d_prime, torch.finfo(d_prime.dtype).max)
    min_values, min_indices = torch.min(d_prime_masked, dim=-1)
    
    candidate_mask = (d_prime < threshold) & valid_range
    candidate_indices = torch.argmax(candidate_mask.int(), dim=-1)
    found = candidate_mask.any(dim=-1)
    periods = torch.where(found, candidate_indices, min_indices).float()
    
    batch_indices = torch.arange(x.size(0), device=device)
    valid_interp = (periods > 1) & (periods < frame_size-1)
    tau = periods[valid_interp].long()
    
    d0 = d_prime[batch_indices[valid_interp], tau - 1]
    d1 = d_prime[batch_indices[valid_interp], tau]
    d2 = d_prime[batch_indices[valid_interp], tau + 1]
    
    offset = (d0 - d2) / (2 * (d0 + d2 - 2 * d1 + 1e-12))
    periods[valid_interp] = float(tau + offset.clamp(-0.5, 0.5))
    
    f0 = sample_rate / (periods + 1e-12)
    return f0
