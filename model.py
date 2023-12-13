import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import cuda
from numbers import Number

class IBNet(nn.Module): 

    def __init__(self, K=256):
        super(IBNet, self).__init__()
        self.K = K

        self.encode = nn.Sequential(
            nn.Conv2d(3, 8, 5, 1, 2),
            nn.Conv2d(8, 8, 5, 1, 2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2*self.K),
            )

        self.decode = nn.Sequential(
            nn.Linear(self.K, 10)
            )
        
    def forward(self, x, num_sample=1):

        statistics = self.encode(x)
        mu  = statistics[:,:self.K]
        std = F.softplus(statistics[:,self.K:]-5,beta=1)

        encoding = self.reparametrize_n(mu,std,num_sample)
        logit = self.decode(encoding)

        return (mu, std), logit

    def reparametrize_n(self, mu, std, n=1):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        def expand(v):
            if isinstance(v, Number): return torch.Tensor([v]).expand(n, 1)
            else: return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(cuda(std.data.new(std.size()).normal_(), std.is_cuda))
        return mu + eps * std

    def weight_init(self):
        for m in self._modules: xavier_init(self._modules[m])

def xavier_init(ms):
    torch.manual_seed(0)
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()
