import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

def get_noise(n_samples, z_dim, device = 'cpu'):
    return torch.randn(n_samples, z_dim, device=device)

class Discriminator(nn.Module):
    def __init__(self, im_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim*4, hidden_dim *2),
            self.get_discriminator_block(hidden_dim*2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_discriminator_block(self, input_dim, output_dim):
        return nn.Sequential(
                    nn.Linear(input_dim, output_dim),
                    nn.LeakyReLU(0.1),
                )

    def forward(self, image):
        return self.disc(image)
    
    def get_disc(self):
        return self.disc

class Generator(nn.Module):
    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(hidden_dim, 2*hidden_dim),
            self.get_generator_block(hidden_dim*2, hidden_dim*4),
            self.get_generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, im_dim),
            nn.Sigmoid()
        )
    
    def get_generator_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace = True)
        )

    def forward(self, noise):
        return self.gen(noise)
    
    def get_gen(self):
        return self.gan

criterion = nn.BCEWithLogitsLoss()
num_epochs = 50
z_dim = 64
display_step = 500
batch_size = 32
lr = 3e-4
# device = 'cuda' if torch.cuda.is_available() else "cpu"
device = 'cpu'

disc = Discriminator().to(device)
gen = Generator(z_dim).to(device)
fixed_noise = get_noise(batch_size, z_dim, device = device)

transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

dataset = datasets.MNIST(root="dataset/", transform = transforms, download = True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
disc_opt = optim.Adam(disc.parameters(), lr=lr)
gen_opt = optim.Adam(gen.parameters(),lr=lr)
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

def get_disc_loss(gen, disc, criterion, real, num_images, z_dim, device):
    fake_noise = get_noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake.detach())
    disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
    disc_real_pred = disc(real)
    disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
    disc_loss = (disc_fake_loss + disc_real_loss)/2
    return disc_loss

def get_gen_loss(gen, disc, criterion, num_images, z_dim, device):
    fake_noise = get_noise(num_images, z_dim)
    fake = gen(fake_noise)
    disc_fake_pred = disc(fake)
    gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
    return gen_loss

cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        cur_batch_size = len(real)
        real = real.view(cur_batch_size, -1)
        disc_opt.zero_grad()
        
        #calculating the disciminator loss
        disc_loss = get_disc_loss(gen, disc, criterion, real, cur_batch_size, z_dim,device)

        #update optimizer
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss = get_gen_loss(gen, disc, criterion, cur_batch_size, z_dim, device)
        gen_loss.backward()
        gen_opt.step()

        mean_discriminator_loss += disc_loss.item()/display_step
        mean_generator_loss +=gen_loss.item()/display_step

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs} DiscriminatorLoss: {disc_loss: .4f}, GeneratorLoss: {gen_loss:.4f}"
            )

        with torch.no_grad():
            fake = gen(fixed_noise).reshape(-1,1,28,28)
            data = real.reshape(-1,1,28,28)
            img_grid_fake = torchvision.utils.make_grid(fake,normalize=True)
            img_grid_real = torchvision.utils.make_grid(data, normalize=True)

            writer_fake.add_image(
                "Mnist Fake Image", img_grid_fake, global_step=step
            )
            writer_fake.add_image(
                "Mnist fake Images", img_grid_fake, global_step=step
            )
            step += 1








