import torch
from torch.utils.data import Dataset


class ForwardDiffusionDataset(Dataset):
    """
    Base class for generating samples for forward diffusion. See class definition in
    mnist_diffusion_dataset.py for example usage.
    """
    def __init__(self, data, beta0, betaT, num_timesteps):
        self.data = data
        self.beta0 = beta0
        self.betaT = betaT
        self.T = num_timesteps

        self.betas = torch.linspace(beta0**0.5, betaT**0.5, num_timesteps)**2
        self.alpha_bars = torch.cumprod(1-self.betas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_m_alpha_bars = torch.sqrt(1-self.alpha_bars)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        t = torch.randint(high=self.T, size=()) # Sample timestep t
        eps = torch.randn_like(x)   # Sample noise
        xt = self.sqrt_alpha_bars[t] * x + self.sqrt_one_m_alpha_bars[t] * eps  # Forward diffusion
        return {'x': xt, 't': t, 'eps': eps}