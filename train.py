import argparse
from importlib import import_module
import yaml
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from utils.utils import train_loop
from models.unet_base import UNet


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)
    print(config)

    diffusion_params = config['diffusion_params']
    dataset_params = config['dataset_params']
    unet_params = config['unet_params']
    training_params = config['training_params']

    # Load dataset and instantiate data loader #
    my_dataset_module = import_module(dataset_params['module_name'])
    MyDatasetClass = getattr(my_dataset_module, dataset_params['name'])
    dataset = MyDatasetClass(diffusion_params['beta_start'], diffusion_params['beta_end'], diffusion_params['num_timesteps'])
    dataloader = DataLoader(dataset, shuffle=True, batch_size=training_params['batch_size'], num_workers=2, prefetch_factor=2)

    # Instantiate model, optimizer and scheduler #
    model = UNet(**unet_params)
    loss_fn = getattr(nn, training_params['loss'])()
    opt = getattr(torch.optim, training_params['optimizer'])(model.parameters(), **training_params['opt_params'])
    sched = getattr(torch.optim.lr_scheduler, training_params['scheduler'])(opt, **training_params['scheduler_params']) \
        if training_params['scheduler'] else None
    
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location=training_params['device'])
        model.load_state_dict(checkpoint['model'], assign=True)
        if sched is not None and hasattr(checkpoint, 'lr_scheduler'):
            sched.load_state_dict(checkpoint['lr_scheduler'])
        opt.load_state_dict(checkpoint['opt'])    

    # Train model #
    for epoch in range(1, training_params['epochs']+1):
        print(f"Starting epoch {epoch}")
        losses = train_loop(dataloader, model, loss_fn, opt, training_params['num_steps'], sched, training_params['device'])
        
    # Save checkpoint #
    if training_params['save_checkpoint']:
        ckpt_dict = {'model': model.state_dict(), 'opt': opt.state_dict()}
        if sched:
            ckpt_dict['lr_scheduler'] = sched.state_dict()
        torch.save(ckpt_dict, training_params['save_checkpoint'])

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for vq vae training')
    parser.add_argument('--config', dest='config_path',
                        default='config/mnist.yaml', type=str)
    parser.add_argument('--use-checkpoint', dest='checkpoint_path',
                        default=None, type=str)
    args = parser.parse_args()
    train(args)