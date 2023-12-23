import torch
import numpy as np
from metrics.hessian import Hessian

def grad_norm(model, criterion, optimizer, dataloader, lp=2):
    model.eval()
    total_norm = []
    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        optimizer.zero_grad()
        batch_loss.backward()
        
        parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
        total_norm.append(torch.norm(torch.stack([torch.norm(p.grad.detach(), lp) for p in parameters]), lp).item())
    return np.mean(total_norm)

def eigen_spec(model, criterion, dataloader):
    model.eval()
    hessian_dataloader = []
    for i, (inputs, labels) in enumerate(dataloader):
        hessian_dataloader.append((inputs, labels))
    hessian_comp = Hessian(model, criterion, dataloader=hessian_dataloader)
    top_eigenvalues, _ = hessian_comp.eigenvalues()
    trace = hessian_comp.trace()
    return top_eigenvalues, np.mean(trace)