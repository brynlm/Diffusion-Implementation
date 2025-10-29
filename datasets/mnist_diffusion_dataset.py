import torch
import torchvision
import torchvision.transforms.v2 as v2
from .forward_diffusion_dataset import ForwardDiffusionDataset

class MnistDiffusionDataset(ForwardDiffusionDataset):
    def __init__(self, beta0, betaT, num_timesteps):
        transform = v2.Compose([
                        v2.Lambda(lambda x: x.unsqueeze(dim=1)),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.RGB()
                    ])
        data = torchvision.datasets.MNIST(root='./data', train=True, download=False)
        features = transform(data.data)
        features = (2*features) - 1
        self.targets = data.targets
        super().__init__(features, beta0, betaT, num_timesteps)