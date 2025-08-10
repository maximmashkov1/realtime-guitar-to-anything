import torch
from torch import nn
import torch.nn.functional as F
def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

class DiscriminatorS(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([nn.utils.spectral_norm(c) for c in [
            nn.Conv1d(1, 128, 15, 1, padding=7),
            nn.Conv1d(128, 128, 41, 2, groups=4, padding=20),
            nn.Conv1d(128, 256, 41, 2, groups=16, padding=20),
            nn.Conv1d(256, 512, 41, 4, groups=16, padding=20),
            nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20),
            nn.Conv1d(1024, 1024, 5, 1, padding=2),
        ]])
        self.conv_post = nn.utils.spectral_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def spectral_norm_loss(self):

        loss = 0.0
        for module in self.modules():
            if hasattr(module, 'weight_orig'):
                weight = module.weight_orig
                if isinstance(module, nn.Conv1d):
                    weight = weight.view(weight.size(0), -1)
                
                u = module.weight_u
                v = module.weight_v
                
                with torch.no_grad():
                    Wv = torch.mv(weight, v)
                    sigma = torch.dot(u, Wv)
                
                loss += (sigma - 1) ** 2
        
        return loss
    def forward(self, x):
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)

        return x.mean(dim=1)
