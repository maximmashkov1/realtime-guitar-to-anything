import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim, max_period=10000, scale=1000):
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) / (half_dim - 1)
    freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent).to(timesteps.device)
    timesteps = timesteps * scale
    args = timesteps[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb
    
class FrequencyConvTranspose2d(nn.Module):

    def __init__(self, in_channels, out_channels, factor):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(factor, 1),
            stride=(factor, 1),
            output_padding=(0, 0)
        )


    def forward(self, x,_):

        return self.conv(x)
    
class FrequencyDown(nn.Module):
    def __init__(self, in_channels, out_channels, factor=2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(factor, 1),
            stride=(factor, 1),
        )
        self.factor=factor

    def forward(self,x,_):
        return self.conv(x)

class CausalConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size-1,0,kernel_size//2,kernel_size//2)
        self.top_bottom_padding = (0,0,kernel_size//2,kernel_size//2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0)
        self.cache = None

    def forward(self, x):

        if self.cache is None:
            x = F.pad(x, self.padding)
        else:
            if self.cache.shape[-1] == 0:
                self.cache = torch.zeros(1, x.shape[1], x.shape[2], self.padding[0]).to(x.device)
            x = torch.cat((self.cache, x), dim=-1)
            self.cache = x[:, :, :, -self.padding[0]:]
            x = F.pad(x, self.top_bottom_padding)

        out = self.conv(x)
        return out

class ResBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        
        self.in_channels=in_channels
        mid = max(out_channels,in_channels)
        self.mid_channels = mid
        self.time_encoder = nn.Sequential(nn.Linear(128,mid))
        
        self.norm1 = nn.BatchNorm2d(in_channels)#nn.LayerNorm(rows*in_channels)
        self.conv1 = CausalConv2d(in_channels,mid,kernel_size)
        self.norm2 = nn.BatchNorm2d(mid)#nn.LayerNorm(rows*mid)
        self.conv2 = CausalConv2d(mid,out_channels,kernel_size)
        self.skip = nn.Conv2d(in_channels,out_channels,1) if in_channels != out_channels else nn.Identity()
        #self.dropout = nn.Dropout(0.08)
    def forward(self, x, t):
        
        N,C,R,W=x.shape
        xt = xt=self.norm1(x)#self.norm1(x.reshape(N,self.in_channels*R,W).transpose(-1,-2)).transpose(-1,-2).reshape(N,self.in_channels,R,W)
        xt = self.conv1(xt)
        tim=self.time_encoder(t).transpose(-1,-2).unsqueeze(-1)

        xt = xt+tim
        xt = F.leaky_relu(xt, 0.01)

        xt = xt=self.norm2(xt)#self.norm2(xt.reshape(N,self.mid_channels*R,W).transpose(-1,-2)).transpose(-1,-2).reshape(N,self.mid_channels,R,W)
        xt = self.conv2(xt)
        xt = F.leaky_relu(xt, 0.01)

        return xt + self.skip(x)
    
class CausalMelMorpher2d(nn.Module):
    
    def __init__(self):
        super().__init__()
        dim=32
        self.time_encoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        self.encoder = nn.ModuleList([
                                      ResBlock2d(1,dim,3),
                                      ResBlock2d(dim,dim,3),
                                      FrequencyDown(dim,dim,2),#40

                                      ResBlock2d(dim,dim,3),
                                      ResBlock2d(dim,dim,3),
                                      FrequencyDown(dim,dim,2),#20
                                      
                                      ResBlock2d(dim,dim,3),
                                      ResBlock2d(dim,dim,3),
                                      ])
        
        
        self.mid_gru = nn.GRU(dim*20,hidden_size=128,num_layers=3,dropout=0.1,batch_first=True)
        self.mid_up = nn.Sequential(nn.Linear(128,dim*10),
                                    nn.LeakyReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(dim*10,dim*10))

        self.decoder = nn.ModuleList([
                                      ResBlock2d(dim+8,dim,3),
                                      ResBlock2d(dim,dim,3),

                                      FrequencyConvTranspose2d(dim,dim,2),#80
                                      ResBlock2d(dim*2,dim,3),
                                      ResBlock2d(dim,1,3),
                                      ])
        
        self.last_h = None
    def forward(self, x, t):
        time_emb = get_timestep_embedding(t.flatten(),128)
        time_emb = self.time_encoder(time_emb).unsqueeze(1)
        skips = []
        x = x.unsqueeze(1)
        for layer in self.encoder:
            if type(layer)==FrequencyDown:
                skips.append(x)
            x = layer(x,time_emb)
        #N, C, 20, S
        N, C, F, S = x.shape
        x = x.reshape(N,32*F,S).transpose(-1,-2)
        x, last_h = self.mid_gru(x,self.last_h)
        if not self.training:
            self.last_h = last_h
        x = self.mid_up(x).transpose(-1,-2).reshape(N,8,40,S)
        #print(x[0],x[0].std())
        x = torch.cat((x,skips.pop()),dim=1)
        for layer in self.decoder:
            x = layer(x,time_emb)
            if type(layer)==FrequencyConvTranspose2d:
                x = torch.cat((x,skips.pop()),dim=1)

        return x.squeeze(1)
    
    def turn_on_caching(self):
        for module in self.modules():
            if type(module) == CausalConv2d:
                module.cache = torch.zeros(module.in_channels,0).cuda()


    

class CausalConvTranspose1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1, 0)

        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0
        )
        self.cache = None

    def forward(self, x):

        if self.cache is None:
            x = F.pad(x, self.padding)
        else:
            if self.cache.shape[-1] == 0:
                self.cache = torch.zeros(1, self.channels, self.padding[0], device=x.device)
            x = torch.cat((self.cache, x), dim=-1)
            self.cache = x[:, :, -self.padding[0]:]

        out = self.conv(x)
        input_length = x.size(-1) - self.padding[0]
        target_length = input_length * self.stride
        exp=self.stride*(self.kernel_size -1)
        out_shape = out.shape[-1]
        start_idx = out_shape-exp//2-target_length
        if (start_idx)<0:
            out= out[:, :, -target_length:]
        else:
            out = out[:, :, start_idx:-exp//2]
        return out


class ResBlock(torch.nn.Module):

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        convs1 = [CausalConv1d(channels,channels,kernel_size,dilation=dilation_) for dilation_ in dilation]
        self.convs1 = nn.ModuleList(convs1)

        convs2 = [CausalConv1d(channels,channels,kernel_size,dilation=1) for _ in dilation]
        self.convs2 = nn.ModuleList(convs2)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

class CausalConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation, groups=1, sn=False):
        super().__init__()
        self.channels = in_channels
        self.dilation = dilation
        self.padding = ((kernel_size-1)*dilation , 0)
        wrapper  = nn.utils.spectral_norm if sn else lambda x: x
        self.conv = wrapper(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation,padding=0,groups=groups))
        self.cache = None
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
            
                
    def forward(self, x):
        """
        Caching is only used in the generator during inference.
        """
        if self.cache == None:
            x = F.pad(x, self.padding)
        else:
            if self.cache.shape[-1] == 0:
                self.cache = torch.zeros(1, self.channels, self.padding[0]).to(x.device)
            x = torch.cat((self.cache, x), dim=-1)
            self.cache = x[:, :, -self.padding[0]:]

        return self.conv(x)



    
class CausalVocoder(nn.Module):

    def __init__(self, upsample_rates=[8,8,4], upsample_kernel_sizes=[16,16,4], upsample_initial_channel=256, resblock_kernel_sizes=[3,5,7], resblock_dilation_sizes=[[1,2], [2,6], [3,12]], mel_bands=80):
        super().__init__()
        #causal hifigan generator https://github.com/jik876/hifi-gan/blob/master/config_v1.json

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.in_conv = nn.Conv1d(mel_bands,upsample_initial_channel,1)
        
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(CausalConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)), k, u))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d))
                
        self.out_conv = CausalConv1d(ch, 1, 7, dilation=1)

    def turn_on_caching(self):
        for module in self.modules():
            if isinstance(module, CausalConv1d) or isinstance(module, CausalConvTranspose1d):
                module.cache = torch.zeros(1,0)

    def forward(self, x):
        
        x = self.in_conv(x)
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            #print('b,',x.shape)
            x = self.ups[i](x)
            #print('a,',x.shape)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.out_conv(x)
        x = torch.tanh(x)
        
        return x

