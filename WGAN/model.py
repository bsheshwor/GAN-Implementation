import torch
import torch.nn as nn

def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_channel = 3, hidden_dim = 64):
        super(Generator,self).__init__()
        self.z_dim = z_dim
        self.gen = nn.Sequential(
            self._generator_block(z_dim, hidden_dim*4),
            self._generator_block(hidden_dim*4, hidden_dim*2, kernel_size = 4, stride=1),
            self._generator_block(hidden_dim*2, hidden_dim),
            self._generator_block(hidden_dim, im_channel, kernel_size = 4, final_layer=True)
        )
    
    def _generator_block(self, input_channels, output_channels, kernel_size = 3, stride = 2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace = True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh()
            )
    
    def unsqueeze_noise(self, noise):
        return noise.view(len(noise),self.z_dim, 1,1)
    
    def forward(self, noise):
        X = self.unsqueeze_noise(noise)
        return self.gen(X)

class Discriminator(nn.Module):
    def __init__(self, im_channel=3, hidden_dim=16):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self._discriminator_block(im_channel, hidden_dim),
            self._discriminator_block(hidden_dim, hidden_dim*2),
            self._discriminator_block(hidden_dim*2, 1, final_layer=True)
        )

    def _discriminator_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.InstanceNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )

    def forward(self, image):
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)



