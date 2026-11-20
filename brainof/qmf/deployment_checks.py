"""Regression checks for deployment fixes, run by the idle-GPU smoke job."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import main_finetuning_qmf as training


def accumulation_tail():
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    steps = []
    scheduler = NS(step=steps.append)
    cfg = NS(train=NS(accumulation_steps=2, clip_grad=5, epochs=2), print_freq=20)
    args = NS(fusion='static', lamb=0, reg_start_epoch=1)
    logger = NS(info=lambda message: None)
    def batch(value, device):
        target = torch.tensor([[1.0, 0.0]])
        return value, None, None, None, target, None
    def loss(model, value, *args, **kwargs):
        return model.weight.sum()*value, torch.tensor(0.0), torch.tensor([[1.0,0.0]]), None, None, None
    def scaler(value, opt, update_grad, **kwargs):
        value.backward()
        if update_grad:
            opt.step()
    with patch.object(training, 'to_device', batch), patch.object(training, 'forward_loss', loss), \
         patch.object(torch.cuda, 'max_memory_allocated', return_value=0):
        for epoch in range(2):
            training.train_one_epoch(model, [1.0, 2.0, 3.0], optimizer, scheduler, scaler,
                                     0.001, epoch, 'cpu', cfg, args, None, None, logger)
    assert abs(model.weight.item()+0.9) < 1e-6, model.weight
    assert steps == [0,1,2,3], steps
    print('partial gradient accumulation and scheduler: OK')

if __name__ == '__main__':
    accumulation_tail()
