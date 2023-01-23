import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Generator, initialize_weights, get_noise
import os
BASE_DIR = os.getcwd()

writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

z_dim = 100
display_step = 500
batch_size = 10
# A learning rate of 0.0002 works well on DCGAN
lr = 5e-5
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01
# These parameters control the optimizer's momentum, which you can read more about here:
# https://distill.pub/2017/momentum/ but you don’t need to worry about it for this course!
beta_1 = 0.5 
beta_2 = 0.999
device = 'cuda'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

IMAGE_PATH = os.path.join(BASE_DIR, 'data/')
print(IMAGE_PATH)
dataset = datasets.ImageFolder(IMAGE_PATH, transform = transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

gen = Generator(z_dim).to(device)
gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lr)
critic = Discriminator().to(device) 
critic_opt = torch.optim.RMSprop(critic.parameters(), lr=lr)

gen = gen.apply(initialize_weights)
critic = critic.apply(initialize_weights)

n_epochs = 5
cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

for epoch in range(n_epochs):
    # Dataloader returns the batches
    for batch_id, (real, _) in enumerate(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        ## Update discriminator ##
        critic_opt.zero_grad()

        for _ in range(CRITIC_ITERATIONS):
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = -(torch.mean(critic_real)- torch.mean(critic_fake))
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            critic_opt.step()

            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        gen_opt.step()

        ## Visualization code ##
        if cur_step % 100 == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {loss_gen}, discriminator loss: {loss_critic}")
        
            with torch.no_grad():
                fake = gen(fake_noise).reshape(-1,1,28,28)
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
            cur_step += 1

