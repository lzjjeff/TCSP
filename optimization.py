import math
from torch.optim.lr_scheduler import LambdaLR


class WarmUpLRScheduler():
    def __init__(self, optimizer, warmup=0.1, total_epochs=40):
        warm_up_epochs = int(total_epochs * warmup)
        if warm_up_epochs <= 0:
            warm_up_epochs = 1
        warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warm_up_epochs if epoch < warm_up_epochs \
            else 0.5 * (math.cos((epoch - warm_up_epochs) / (total_epochs - warm_up_epochs) * math.pi) + 1)
        self.scheduler = LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)

    def step(self, epoch):
        self.scheduler.step(epoch)
