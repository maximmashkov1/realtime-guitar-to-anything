import torch
import torch.nn as nn
import torch.nn.functional as F
from causal_decoder import CausalVocoder, CausalMelMorpher2d
from discriminator import DiscriminatorS
import torchaudio


    
class GuitarToneCloning(nn.Module):

    def __init__(self):
        super().__init__()

        self.vocoder = CausalVocoder()
        self.vocoder_disc = DiscriminatorS()
        self.flow = CausalMelMorpher2d()
        self.mean_restoration = nn.Sequential(nn.Dropout(0.4),
                                 nn.Linear(10,64),
                                 nn.Dropout(0.1),
                                 nn.LeakyReLU(),
                                 nn.Linear(64,10))
        self.norm_params = {'x':{},'y':{}}

    def mel_spectral_loss(self, x, y, sample_rate=24000, n_mels=80):
        mel_loss = 0
        
        for n_fft in [512, 1024, 2048]:
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=n_fft // 4,
                n_mels=n_mels,
                f_max=8000,
                window_fn=torch.hann_window
            ).to(y.device)

            mel_y = mel_transform(y)
            mel_x = mel_transform(x)

            log_mel_y = torch.log(mel_y + 1e-9)
            log_mel_x = torch.log(mel_x + 1e-9)

            mel_loss += F.l1_loss(log_mel_x, log_mel_y)

        return mel_loss
    
    def mel_normalize(self, x):
        return (x-self.norm_params['mean'])/self.norm_params['std']

    def mel_denormalize(self, x):
        return x*self.norm_params['std'] + self.norm_params['mean']


    def train_vocoder(self, mel, true, optimizer, loss_weight):
        """
        mel
        true: true waveform
        """
        optimizer.zero_grad()
        generated = self.vocoder(mel)
        print('shapes: ',generated.shape,true.shape)
        print(generated.std().item())
        print(generated.shape,true.shape)
        loss_spectral = self.mel_spectral_loss(generated.squeeze(1),true.squeeze(1))*loss_weight['spec']

        ones = 0.9 * torch.ones(true.shape[0], device=true.device)
        zeros = 0.1 * torch.ones(true.shape[0], device=true.device)
        

        original_requires_grad = []
        for param in self.vocoder_disc.parameters():
            original_requires_grad.append(param.requires_grad)
            param.requires_grad_(False)

        loss_adv = F.binary_cross_entropy_with_logits(
            self.vocoder_disc(generated), zeros*0
        ) * loss_weight['adv']

        (loss_spectral+loss_adv).backward()


        for param, rg in zip(self.vocoder_disc.parameters(), original_requires_grad):
            param.requires_grad_(rg)

        disc_input = torch.cat((generated.detach(), true), dim=0)
        disc_pred = self.vocoder_disc(disc_input)
        targets = torch.cat((ones, zeros))
        loss_disc = F.binary_cross_entropy_with_logits(disc_pred, targets) * loss_weight['disc']
        (loss_disc + self.vocoder_disc.spectral_norm_loss()*loss_weight['sn']).backward(retain_graph=True)

        optimizer.step()

        return loss_spectral.item(), loss_adv.item(), loss_disc.item()

    def forward(self,chunk,coeff=0.3):
        chunk = self.mel_normalize(chunk.unsqueeze(0).unsqueeze(1))
        mean = F.interpolate(chunk, (10))
        mean_corrected = self.mean_restoration(mean)
        displacement = F.interpolate(mean_corrected - mean, (80),mode='linear')
        chunk += displacement
        chunk = chunk.squeeze(0).unsqueeze(2)
        noise = torch.randn_like(chunk)
        chunk = chunk*coeff+noise*(1-coeff)
        chunk = self.mel_denormalize(self.flow(chunk,torch.tensor([coeff]).cuda()) + noise)
        out=self.vocoder(chunk).flatten()
        return out

def gradient_penalty(discriminator, real, fake):
    batch_size = real.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, requires_grad=True).to(real.device)
    interpolated = epsilon * real + (1 - epsilon) * fake
    interpolated.requires_grad_(True)

    pred_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=pred_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(pred_interpolated),
        create_graph=True,
        retain_graph=True,
    )[0] 
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty
