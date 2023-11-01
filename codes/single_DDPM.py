# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:06:13 2023
@author: Jean-Baptiste Bouvier

Simplest implementation of 1D Denoising Diffusion Probabilistic Models (DDPM).   
Built from Algorithms 1 and 2 of "Denoising Diffusion Probabilistic Models".
"""

import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


#%% Hyperparameters

mu_data = 1 # mean of the initial data distribution
sigma_data = 0.01 # std of the initial data distribution
noise_scales = torch.tensor([0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.999])
test_size = 2000 # dimension of data

### Training
nb_epochs = 20000
batch_size = 64
lr = 1e-3

#%% Denoiser Neural Network

class Denoiser(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.input_size = 1
        self.net = nn.Sequential(nn.Linear(self.input_size+1, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.width), nn.ReLU(),
                                 nn.Linear(self.width, self.input_size) )
        
    def forward(self, x, alpha):
        """Takes noised data and predict noise level."""
        s = alpha*torch.ones((x.shape[0], 1))
        return self.net( torch.cat((x,s), dim=1) )

den = Denoiser(32)


#%% Training loop with iteration on the denoisers

losses = np.zeros(nb_epochs)
optimizer = torch.optim.SGD(den.parameters(), lr)

for epoch in range(nb_epochs):   
    
    id_noise_scale = random.randint(0, len(noise_scales)-1)
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])  
    x0 = torch.randn((batch_size,1))*sigma_data + mu_data # initial unnoised data
    eps = torch.randn_like(x0) # noise
    x_noised = np.sqrt(alpha_bar)*x0 + np.sqrt(1-alpha_bar)*eps # adding noise corresponding to noise_scale
    pred = den(x_noised, 1 - noise_scales[id_noise_scale])
    loss = torch.linalg.vector_norm(eps - pred) # difference between actual noise and prediction
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses[epoch] = loss.detach().item()
    
plt.title("Training loss")
plt.plot(np.arange(nb_epochs), losses)
plt.show()
    

#%% Denoising process

def plot_sample(x, id_noise_scale):
    plt.title(f"Noised scale {id_noise_scale}")
    plt.scatter(np.arange(test_size), x.numpy())
    plt.ylim([-3., 3.])
    plt.show()

### Sample from the most noised distribution
alpha_bar = torch.prod(1 - noise_scales)
x = torch.randn((test_size,1))*torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar)
plot_sample(x, len(noise_scales))

print(f"Sample: mean {x.mean().item():.3f}  std {x.std().item():.3f}")

for id_noise_scale in range(len(noise_scales)-1, -1, -1):
    if id_noise_scale > 1: z = torch.randn_like(x)
    else: z = torch.zeros_like(x)
        
    alpha = 1 - noise_scales[id_noise_scale]
    alpha_bar = torch.prod(1 - noise_scales[:id_noise_scale+1])
    sigma_sq = noise_scales[id_noise_scale] * (1 - alpha_bar/alpha)/(1 - alpha_bar)
    with torch.no_grad():
        x = (x - (1-alpha)*den(x, alpha)/np.sqrt(1-alpha_bar) )/np.sqrt(alpha) + torch.sqrt(sigma_sq)*z
    plot_sample(x, id_noise_scale)
    print(f"Sample: mean {x.mean().item():.3f}  std {x.std().item():.3f}")
    
print(f"Target: mean {mu_data:.3f}  std {sigma_data:.3f}")