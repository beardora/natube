import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import pdb

class VAE(nn.Module):
    def __init__(self,encodeconv,lastlayer1,lastlayer2,decode,iscuda=True,loss='BCE',sigma=1):
        super(VAE, self).__init__()
        self.encodeconv = encodeconv
        self.iscuda = iscuda
        self.lastlayer1 = lastlayer1
        self.lastlayer2 = lastlayer2
        self.decode = decode
        self.loss = loss
        self.sigma = sigma
        if self.loss == 'BCE':
            self.reconstruction_function = nn.BCELoss()
        elif self.loss == 'MSE':
            self.reconstruction_function = nn.MSELoss()
        self.reconstruction_function.size_average = False
        
    def encode(self, x):
         h1 = self.encodeconv(x)
         return self.lastlayer1(h1),self.lastlayer2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.iscuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss_function(self,recon_x, x, mu, logvar,sigma):
        BCE = self.reconstruction_function(recon_x, x)
        if self.loss == 'MSE':
            BCE.mul_(0.5/sigma**2)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        
        return BCE + KLD, BCE, KLD

    def loss_function_individual(self,recon_x, x, mu, logvar,sigma):
        nbatch = recon_x.size()[0] 
        t = x.view(nbatch,-1)
        f = recon_x.view(nbatch,-1)
        if self.loss == 'BCE':
            BCE = -torch.sum(t*torch.log(f)+(1-t)*torch.log(1-f),1)
        elif self.loss == 'MSE':
            BCE = 0.5/sigma**2*torch.sum((t-f).pow(2),1)
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element,1).mul_(-0.5)
        return BCE + KLD, BCE, KLD    

