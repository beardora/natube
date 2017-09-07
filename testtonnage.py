import numpy as np
from model import VAE
#from maxpoolmodel import VAE
from data import dataloader
import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import roc_curve
import pdb
import pickle
import torch.nn.functional as F
import argparse
isBCE=False
from sklearn import metrics 
import os
from scipy.stats import norm
import matplotlib.gridspec as gridspec
from itertools import product
from trainIteration import trainmain
#import sys; sys.argv=['']; del sys


#%%
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many interval to log')
parser.add_argument('--file_root', type=str, default="", metavar='filename',
                    help='file directory of this run')
parser.add_argument('--filename', type=str, default="run", metavar='filename',
                    help='file name of this run')                    
parser.add_argument('--ratio', type=float, default=0.7, metavar='N',
                    help='percentage of Training sample (out of 300)')                    
parser.add_argument('--noplot',action='store_true', default=False, help='no plot')                    
parser.add_argument('--noise', default=0.5,type=float, metavar='N',
                    help='added noise level')                    
parser.add_argument('--continueTrain', action='store_true', default=False, help='Continue Training')                    
parser.add_argument('--nplot', default=5, type=int, metavar='N',
                    help='plot per row * col')                    
parser.add_argument('--logfile', default="", type=str, metavar='filename',
                    help='plot per row * col')     
parser.add_argument('--sigma', default=1, type=float, metavar='N',
                    help='sigma for MSE loss')     

    
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)

iscuda = args.cuda
isplot = not args.noplot


if args.file_root !="" and not os.path.isdir(args.file_root):
    os.mkdir(args.file_root)

thisrunfilename = os.path.join(args.file_root,args.filename)
sigma = args.sigma
nplot = 5

islog = args.logfile != ""
nchannel = [30,10,10,5]
encodeconv = nn.Sequential(
             nn.Conv2d(1,nchannel[0],kernel_size=[1,5],stride=[1,5]),
             nn.ReLU(True),
             nn.Conv2d(nchannel[0],nchannel[1], kernel_size = [1,5], stride = [1,5]),
             nn.ReLU(True),
             nn.Conv2d(nchannel[1],nchannel[2],kernel_size=[4,1]),
            )
lastlayer1 = nn.Conv2d(nchannel[2], nchannel[3], kernel_size=[1,48])
lastlayer2 = nn.Conv2d(nchannel[2], nchannel[3], kernel_size=[1,48])

decode = nn.Sequential(
            nn.ConvTranspose2d(nchannel[3], nchannel[2], kernel_size = [1,48]),
            nn.ReLU(True),
            nn.ConvTranspose2d(nchannel[2], nchannel[1], kernel_size = [4,1]),
            nn.ReLU(True),
            nn.ConvTranspose2d(nchannel[1],nchannel[0],kernel_size=[1,5], stride=[1,5]),
            nn.ReLU(True),
            nn.ConvTranspose2d(nchannel[0],1,kernel_size=[1,5], stride=[1,5]),
)

args.nchannel = nchannel


model = VAE(encodeconv=encodeconv,lastlayer1=lastlayer1,lastlayer2=lastlayer2,decode=decode,iscuda=args.cuda,loss="MSE")



n = 20
optimizer = optim.Adam(model.parameters(), lr=1e-3)


ismode = "Train"
if args.continueTrain:
    ismode = "continue"

loaddata = dataloader(loaddata="Tonnage",format="torch",demean=True,scale=False)
trainData, validataData = dataloader.splitTensorDataset(loaddata.trainData,args.ratio)

trainmean = trainData.data_tensor.mean(0)
trainData.data_tensor.sub_(trainmean.expand_as(trainData.data_tensor))
validataData.data_tensor.sub_(trainmean.expand_as(validataData.data_tensor))
loaddata.testData.data_tensor.sub_(trainmean.expand_as(loaddata.testData.data_tensor))

train_loader = torch.utils.data.DataLoader(trainData,shuffle=True,batch_size=args.batch_size)
validate_loader = torch.utils.data.DataLoader(validataData,shuffle=True,batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(loaddata.testData,shuffle=True,batch_size=args.batch_size)

print("# of samples are {}".format(len(train_loader.dataset)))



logepoch = []

tosavepara = {'T':0,'auc':0,'args':args}
    

if ismode == "Train" or ismode == "continue":
    start_epoch = 1
    npara = 0
    
    for i in model.parameters():
        npara = npara + i.numel()

    if islog:
        logfilename = os.path.join(args.file_root,args.logfile)
        flog = open(logfilename, 'w') 
        flog.write("# of parameters is {}\n".format(npara))
        flog.write(str(args))
        flog.write("\n")
        
    trainfunc = trainmain(model,optimizer,train_loader,iscuda=args.cuda,is1D=True,nplot=args.nplot,logfile=thisrunfilename+'train.log',iscontinue=args.continueTrain)
    validatefunc = trainmain(model,optimizer,validate_loader,iscuda=args.cuda,is1D=True,nplot=args.nplot,logfile=thisrunfilename+'validate.log',iscontinue=args.continueTrain)
    testfunc = trainmain(model,optimizer,test_loader,iscuda=args.cuda,is1D=True,nplot=args.nplot,logfile=thisrunfilename+'test.log',iscontinue=args.continueTrain)

    start_epoch = 0
    if args.continueTrain:
        start_epoch = trainfunc.log["epoch"][-1]
    for epoch in range(start_epoch, start_epoch+args.epochs + 1):
        issave = False
        isplot = False
        if epoch % args.log_interval == 1:
            issave = True
            isplot = not args.noplot

        trainfunc.train(epoch,isTrain=True,noise_factor=args.noise,issave=issave,isplot=isplot,pltfile=thisrunfilename+"train.eps")

        if epoch % args.log_interval == 1:
            validatefunc.train(epoch,isTrain=False,issave=issave,isplot=isplot,pltfile=thisrunfilename+"validate.eps")
            testfunc.train(epoch,isTrain=False,issave=issave,isplot=isplot,pltfile=thisrunfilename+"test.eps")
            
            Tin = validatefunc.savedpara["loss"]
            Tout = testfunc.savedpara["loss"]
            thresh = np.percentile(Tin,99.5)
            nvalidate = np.sum(Tin>thresh)
            ntest = np.sum(Tout>thresh)
            fpr, tpr, thresholds = metrics.roc_curve(np.hstack((Tin*0,Tout*0+1)), np.hstack((Tin,Tout)))
            auc = metrics.auc(fpr, tpr)
            
            T = (np.mean(Tout) - np.mean(Tin))/(np.sqrt(np.std(Tin)**2+np.std(Tout)**2))
            print("Model parameters {}, Incontrol false detection {}, OC Detection {} with AUC {} and T {}".format(npara,nvalidate,ntest,auc,T))
            if islog:
                tosave = [epoch,T,auc, trainfunc.log['loss'][-1],validatefunc.log['loss'][-1],testfunc.log['loss'][-1]]
                flog.write(','.join(map(str, tosave)) +"\n")
        
        torch.save(model.state_dict(), thisrunfilename+'model_epoch_DCVAE_1d.pth')

elif ismode == "Test":
    thisrunfilename = 'tonnage10D/run'
    loaddata = dataloader(loaddata="Tonnage",format="torch",demean=True,scale=False)
    trainData, validataData = dataloader.splitTensorDataset(loaddata.trainData,0.2)
    
    train_loader = torch.utils.data.DataLoader(trainData,shuffle=True,batch_size=128)
    validate_loader = torch.utils.data.DataLoader(validataData,shuffle=True,batch_size=128)
    test_loader = torch.utils.data.DataLoader(loaddata.testData,shuffle=True,batch_size=128)

    trainfunc = trainmain(model,optimizer,train_loader,iscuda=False,is1D=True,nplot=3,logfile=thisrunfilename+'train.log',iscontinue=True)
    validatefunc = trainmain(model,optimizer,validate_loader,iscuda=False,is1D=True,nplot=3,logfile=thisrunfilename+'validate.log',iscontinue=True)
    testfunc = trainmain(model,optimizer,test_loader,iscuda=False,is1D=True,nplot=3,logfile=thisrunfilename+'test.log',iscontinue=True)
    
