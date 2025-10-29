import torch

def train_loop(dataloader, model, loss_fn, opt, num_steps, sched=None):
    device = 'mps' if torch.mps.is_available() else 'cpu'
    model.to(device)
    model.train()

    losses = []
    for batch_num, batch in enumerate(dataloader):
        batch['x'] = batch['x'].to(device)
        batch['t'] = batch['t'].to(device)
        batch['eps'] = batch['eps'].to(device)
        
        opt.zero_grad()
        pred = model(batch['x'], batch['t'])
        loss = loss_fn(pred, batch['eps'])
        
        loss.backward()
        opt.step()
        if sched:
            sched.step()

        if batch_num % 25 == 0:
            print(f"Batch {batch_num}: tr_loss = {loss.item()}")
            losses.append(loss.item())

        if batch_num >= num_steps:
            break
    
    return losses