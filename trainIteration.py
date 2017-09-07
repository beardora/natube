#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:29:29 2017

@author: yanhaopku
"""

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io as sio
import os
import model as modelall
import torch.optim as optim
import copy
import pdb
from torchvision import datasets, transforms
import torchvision
import re
import matplotlib.pyplot as plt
from scipy.stats import norm
import pickle

class trainmain():
    def __init__(self,model,optimizer,train_loader,iscuda=True,is1D=True,nplot=8,logfile='',iscontinue=False):
        self.train_loader = train_loader
        self.iscuda = iscuda
        self.model = model
        self.optimizer = optimizer
        self.ndf = model.decode[0].weight.size()[0]
        self.is1D = is1D
        self.nplot = nplot
        
        if self.is1D:
            self.f, self.axes = plt.subplots(nplot,nplot, sharex=True, sharey=False)
            self.f2D,self.axes2D = plt.subplots(nplot,nplot, sharex=True, sharey=False)
        else:
            self.f = plt.subplot(111)
            self.axes = []
        
        
        if self.iscuda:
            self.model = self.model.cuda()
            
        if not iscontinue:    
            self.savedpara = {"mu":np.zeros((len(self.train_loader.dataset),self.ndf)),\
                            "logvar":np.zeros((len(self.train_loader.dataset),self.ndf)),\
                            "loss":np.zeros((len(self.train_loader.dataset))),\
                            "BCE":np.zeros((len(self.train_loader.dataset))),\
                            "KLD":np.zeros((len(self.train_loader.dataset)))}
            
            self.log = {"epoch":[],\
                        "loss":[],\
                        "BCE":[],\
                        "KLD":[]}
            self.logfile = logfile
        else:
            f = open(logfile,'rb')
            self.model, self.log, self.savedpara= pickle.load(f)
            f.close()
            self.logfile = logfile
        self.floss,self.axesloss = plt.subplots(len(self.log)-1)
        self.lines = [[] for i in range(len(self.log)-1)]
        for i,(key,value) in enumerate(self.log.items()):
            if i>0:
                self.lines[i-1],=self.axesloss[i-1].plot([],[])
                self.axesloss[i-1].set_ylabel(key)
                self.axesloss[i-1].set_autoscaley_on(True)
                
    def getmodel(self):
        return self.model
        
    def plotloss(self,pltfile):
        for i,(key,value) in enumerate(self.log.items()):
            if i>0:
                self.lines[i-1].set_xdata(self.log["epoch"])
                self.lines[i-1].set_ydata(value)
                self.axesloss[i-1].relim()
                self.axesloss[i-1].autoscale_view()
        #We need to draw and flush
        self.floss.canvas.draw()
        self.floss.canvas.flush_events()
        self.floss.savefig(pltfile)
    
    def logtofile(self):
        with open(self.logfile,'wb') as f:
            pickle.dump([self.model,self.log,self.savedpara], f)
                
    def plotrecon1D(self,data,recon_batch,pltfile):
        npoint = data.size()[-1]
        nplot = self.nplot
        for iplot in range(nplot*nplot):
            i,j = iplot%nplot,int(iplot/nplot)
# if for tonnage dataset            
#            self.axes[i][j].plot(range(npoint),data.data.numpy()[iplot][0][0],'k',range(npoint),recon_batch.data.numpy()[iplot][0][0],'r')
# if for other dose data
            self.axes[i][j].plot(range(npoint),data.data.numpy()[iplot][0],'k',range(npoint),recon_batch.data.numpy()[iplot][0],'r')
            self.axes[i][j].axis('off')
            
        self.f.savefig(pltfile)
        for iplot in range(nplot*nplot):
            i,j = iplot%nplot,int(iplot/nplot)
            self.axes[i][j].cla()
        
    def plotrecon2D(self,recon_batch,pltfile):       
        npoint = 32
        torchvision.utils.save_image(recon_batch.view(recon_batch.size()[0],1,npoint,npoint).cpu().data,pltfile)
    
        
    
    def plottest1D(self,ichannel,pltfile):
        grid_x = norm.ppf(np.linspace(0.05, 0.95, self.nplot))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, self.nplot))
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):            
                z_sample = Variable(torch.from_numpy(np.array([[xi, yi]])),volatile=True)
# For tonnage
#                x_decoded = self.model.decode(z_sample.float().view(1,2,1,1))
# For dose
                x_decoded = self.model.decode(z_sample.float().view(1,2,1))
# For tonnage                
#                self.axes2D[i][j].plot(x_decoded[0][0][ichannel].data.numpy(),'k')
# For dose
                self.axes2D[i][j].plot(x_decoded[0][0].data.numpy(),'k')
                self.axes2D[i][j].axis('off')
        self.f2D.savefig(pltfile, format='eps', dpi=300, bbox_inches='tight')
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):            
                self.axes2D[i][j].cla()      
                
    def plottest2D(self,pltfile):
        digit_size = 32
        figure = np.zeros((digit_size * self.nplot, digit_size * self.nplot))
        grid_x = norm.ppf(np.linspace(0.05, 0.95, self.nplot))
        grid_y = norm.ppf(np.linspace(0.05, 0.95, self.nplot))
        for i, yi in enumerate(grid_x):
            for j, xi in enumerate(grid_y):            
                z_sample = Variable(torch.from_numpy(np.array([[xi, yi]])),volatile=True)
                x_decoded = self.model.decode(z_sample.float().view(1,2,1,1))
                digit = x_decoded[0].view(digit_size, digit_size).data.cpu().numpy()
                figure[i * digit_size: (i + 1) * digit_size,
                      j * digit_size: (j + 1) * digit_size] = digit

        self.f.imshow(figure)
        self.f.savefig(pltfile, format='eps', dpi=300, bbox_inches='tight')
        
        
        

    def train(self,epoch,isTrain=False,noise_factor=0.5,issave=False,isplot=True,pltfile=[],islog=True):
        istart = 0
        if isTrain:
            self.model.train()
        else:
            self.model.eval()
            noise_factor = 0
            
        train_loss = 0
        train_BCE = 0
        train_KLD = 0
        ntrain = len(self.train_loader.dataset)
        
        for batch_idx, (data, _) in enumerate(self.train_loader):
            nbatch = data.size()[0]
            data = Variable(data)
            data_noise = data + noise_factor * Variable(torch.randn(data.size()))
#            data_noise.data.clamp_(0., 1.)
    
            if self.iscuda:
                data = data.cuda()
                data_noise = data_noise.cuda()
            recon_batch, mu, logvar = self.model(data_noise)
            mu = mu.view(nbatch,-1)
            logvar = logvar.view(nbatch,-1)
            
            loss,BCE,KLD = self.model.loss_function(recon_batch, data, mu, logvar,self.model.sigma)
            if isTrain:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            train_loss += loss.data[0]
            train_BCE += BCE.data[0]
            train_KLD += KLD.data[0]
            
            
            if isplot:
                if self.is1D:
                    self.plotrecon1D(data,recon_batch,pltfile)
                    if self.ndf == 2:
                        pltfilemodel = re.sub('.eps','_model.eps',pltfile)
                        self.plottest1D(0,pltfilemodel)
    
                else:
                    if self.ndf == 2:
                        pltfilemodel = re.sub('.eps','_model.eps',pltfile)
                        self.plottest2D(pltfilemodel)
                    trainmain.plotrecon2D(data,pltfile)
                    pltfilerecon = re.sub('.eps','_recon.eps',pltfile)
                    trainmain.plotrecon2D(recon_batch,pltfilerecon)

            if issave:
                lossindi,BCEindi,KLDindi = self.model.loss_function_individual(recon_batch, data, mu, logvar,self.model.sigma)
                if self.iscuda:
                    mu = mu.cpu()
                    logvar=logvar.cpu()
                    lossindi=lossindi.cpu()
                    BCEindi=BCEindi.cpu()
                    KLDindi=KLDindi.cpu()      
                
                self.savedpara["mu"][istart:istart+nbatch,:] = np.squeeze(mu.data.numpy())
                self.savedpara["logvar"][istart:istart+nbatch,:] = np.squeeze(logvar.data.numpy())
                self.savedpara["loss"][istart:istart+nbatch] =  np.squeeze(lossindi.data.numpy())
                self.savedpara["BCE"][istart:istart+nbatch] =  np.squeeze(BCEindi.data.numpy())
                self.savedpara["KLD"][istart:istart+nbatch] = np.squeeze(KLDindi.data.numpy())
                istart = istart+nbatch    

        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, train_loss / ntrain))
        
        train_loss = train_loss/ntrain
        train_BCE = train_BCE/ntrain
        train_KLD = train_KLD/ntrain
        
        self.log["epoch"].append(epoch)
        self.log["loss"].append(train_loss)
        self.log["BCE"].append(train_BCE)
        self.log["KLD"].append(train_KLD)

        pltlossfile = re.sub('.eps','_loss.eps',pltfile)            
        self.plotloss(pltlossfile)
        if islog: 
            self.logtofile()
            
def generativeDistribution(self,x,z):
    """ Example of a target distribution that could be sampled from using NUTS.  (Doesn't include the normalizing constant.)
    Note: 
    """
    self.model.eval()
    nbatch = 1
    zv = Variable(torch.from_numpy(z).view(1,self.ndf,1,1).float(),requires_grad=True)
    xv = Variable(x,requires_grad=False)
    if self.iscuda:
        zv = zv.cuda()
        xv = xv.cuda()
    
    recon_zx = self.model.decode(zv)
    logpxonz = 0.5/sigma**2*torch.sum((recon_zx-xv).pow(2))
    logpz = torch.sum((zv).pow(2))/2
    
    logp = -logpxonz - logpz
    logp.backward()
    return logp.cpu().data.numpy(), np.squeeze(zv.grad.cpu().data.numpy())

def trueLE(self,x,M=1000,Madapt=1000):
    self.model.eval()
    posterior = lambda z: self.generativeDistribution(x,z)
    theta0 = np.random.randn(self.ndf)
    samples, lnprob, epsilon = nuts6(posterior, M, Madapt, theta0)
    qmu = mu.squeeze().data.numpy()
    xv = Variable(x,requires_grad=False)
    recon_x, mu, logvar = self.model(xv)
    
    qsigmaest = (logvar/2).exp().squeeze().data.numpy()
    
    logqz = -1/2*np.sum(((samples-qmu)/qsigmaest)**2,axis=1)-log(2*pi)/2*2
    logpz = -1/2*np.sum((samples**2),axis=1)-log(2*pi)/2*2
    
    decodemodel = self.model.decode(Variable(torch.from_numpy(samples).float().view(M,2,1,1)))
    deviation = decodemodel.view(M,-1).data.numpy()-x.view(1,-1).numpy()
    logpxz = -np.sum(deviation**2,axis=1)/2/sigma**2
    logeach = logqz-logpz-logpxz
    demeanlogeach = logeach - np.mean(logeach)
    trueliklihood = np.mean(logeach) + np.exp(demeanlogeach).mean()
    
    
    