Diffusion Implementation in PyTorch
===================================
This repository implements a simple DDPM model and config utilities for quick experimentation/iteration, with the purpose of being easily extendable to other datasets, architectures for experimentation. 

This repository implements:
* Basic building blocks for a Unet architecture (or other CNN variations);
* A Base pytorch dataset class for sampling forward diffusion process, which can be extended for arbitrary datasets;
* Pipeline for defining model and training parameters.

## Training
* Run ```python -m train --config config/mnist.yaml``` in your terminal. Refer config/mnist.yaml file as an example of how to customize your own config.
* Can optionally add ```--use-checkpoint path_to_checkpoint.pt``` to resume a model's training from a checkpoint. The .pt file is expected to hold state dictionaries of a torch model and optimizer, and also potentially a scheduler. 
