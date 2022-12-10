import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from survae.optim.schedulers import LinearWarmupScheduler

optim_choices = {'sgd', 'adam', 'adamax'}

def get_optim(args, model):
    assert args.optimizer in optim_choices

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=args.lr, betas=(args.momentum, args.momentum_sqr))

    if args.warmup > 0:
        scheduler_iter = LinearWarmupScheduler(optimizer, total_epoch=args.warmup)
    else:
        scheduler_iter = None

    scheduler_epoch = ExponentialLR(optimizer, gamma=args.gamma)


    return optimizer, scheduler_iter, scheduler_epoch
