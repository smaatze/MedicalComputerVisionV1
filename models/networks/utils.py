
from torch.optim import lr_scheduler

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    return num_params

def get_scheduler(optimizer, opt):
    print('opt.lr_policy = [{}]'.format(opt.lr_policy))
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch+1+opt.epoch_count-opt.niter)/float(opt.niter_decay+1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule())
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'step2':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        print('schedular=plateau')
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=0.01, patience=5)
    elif opt.lr_policy == 'plateau2':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'step_warmstart':
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 100:
                lr_l = 1
            elif 100 <= epoch < 200:
                lr_l = 0.1
            elif 200 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step_warmstart2':
        def lambda_rule(epoch):
            #print(epoch)
            if epoch < 5:
                lr_l = 0.1
            elif 5 <= epoch < 50:
                lr_l = 1
            elif 50 <= epoch < 100:
                lr_l = 0.1
            elif 100 <= epoch:
                lr_l = 0.01
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler