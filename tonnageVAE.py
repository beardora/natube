import numpy as np
from model import VAE
#from maxpoolmodel import VAE
from data import dataloader
import torch.nn as nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.optim as optim
#import sys; sys.argv=['']; del sys
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
parser.add_argument('--ratio', type=float, default=0.3, metavar='N',
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
nchannel = [30,10,10,2]
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


model = VAE(encodeconv=encodeconv,lastlayer1=lastlayer1,lastlayer2=lastlayer2,decode=decode,iscuda=False,loss="MSE")

#model = VAE(iscuda=False,loss="MSE")


n = 20
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch,train_loader,isTrain=True,iscuda=True,issave=False,isplot=True,pltaxes=[],pltfile=[]):
    istart = 0
    
    ndf = model(Variable(train_loader.dataset[0][0].unsqueeze(0).unsqueeze(1)))[-1].view(-1).size()[0]
    

    if issave:
        muall = np.zeros((len(train_loader.dataset),ndf))
        logvarall= np.zeros((len(train_loader.dataset),ndf))
        lossall = np.zeros((len(train_loader.dataset)))
        BCEall =  np.zeros((len(train_loader.dataset)))
        KLDall = np.zeros((len(train_loader.dataset)))

    if isTrain:
        model.train()
        noise_factor = args.noise
    else:
        model.eval()
        noise_factor = 0
        
    train_loss = 0
    train_BCE = 0
    train_KLD = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        nbatch = data.size()[0]
        data = Variable(data)
        data_noise = data + noise_factor * Variable(torch.randn(data.size()))

        if iscuda:
            data = data.cuda()
            data_noise = data_noise.cuda()
        recon_batch, mu, logvar = model(data_noise)
        mu = mu.view(nbatch,-1)
        logvar = logvar.view(nbatch,-1)
        
        loss,BCE,KLD = model.loss_function(recon_batch, data, mu, logvar,sigma)
        
        if isplot:
            f,axes = pltaxes
            npoint = data.size()[-1]
            for iplot in range(nplot*nplot):
                i,j = iplot%nplot,int(iplot/nplot)
                
                axes[i][j].plot(range(npoint),data.data.numpy()[iplot][0][0],'k',range(npoint),recon_batch.data.numpy()[iplot][0][0],'r')
                
            f.savefig(pltfile)
            for iplot in range(nplot*nplot):
                i,j = iplot%nplot,int(iplot/nplot)
                axes[i][j].cla()

        
        if issave:
            lossindi,BCEindi,KLDindi = model.loss_function_individual(recon_batch, data, mu, logvar,sigma)
    
            if iscuda:
                mu = mu.cpu()
                logvar=logvar.cpu()
                loss=loss.cpu()
                BCEindi=BCEindi.cpu()
                KLDindi=KLDindi.cpu()                
            muall[istart:istart+nbatch,:] = np.squeeze(mu.data.numpy())
            logvarall[istart:istart+nbatch,:] = np.squeeze(logvar.data.numpy())
            lossall[istart:istart+nbatch] =  np.squeeze(loss.data.numpy())
            BCEall[istart:istart+nbatch] =  np.squeeze(BCEindi.data.numpy())
            KLDall[istart:istart+nbatch] = np.squeeze(KLDindi.data.numpy())
            istart = istart+nbatch        
        if isTrain:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss += loss.data[0]
        train_BCE += BCE.data[0]
        train_KLD += KLD.data[0]
    if issave:
        lossall = {"mu":muall,"logvar":logvarall,"loss":lossall,"BCE":BCEall,"KLD":KLDall}    
    else:
        lossall = {}
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    ntrain = len(train_loader.dataset)
    return train_loss/ntrain,train_BCE/ntrain,train_KLD/ntrain,lossall

ismode = "Train"
if args.continueTrain:
    ismode = "continue"

loaddata = dataloader(loaddata="Tonnage",format="torch",demean=True,scale=False)
trainData, validataData = dataloader.splitTensorDataset(loaddata.trainData,args.ratio)

train_loader = torch.utils.data.DataLoader(trainData,shuffle=True,batch_size=args.batch_size)
validate_loader = torch.utils.data.DataLoader(validataData,shuffle=True,batch_size=args.batch_size)
test_loader = torch.utils.data.DataLoader(loaddata.testData,shuffle=True,batch_size=args.batch_size)
print("# of samples are {}".format(len(train_loader.dataset)))

alltrain_loss = []
alltrain_BCE = []
alltrain_KLD = []

allvalidation_loss = []
allvalidation_BCE = []
allvalidation_KLD = []

alltest_loss = []
alltest_BCE = []
alltest_KLD = []
if isplot:
    lossfig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False)
    trainf, trainaxes = plt.subplots(nplot,nplot, sharex=True, sharey=False)
    testf, testaxes = plt.subplots(nplot,nplot, sharex=True, sharey=False)
    validatef, validateaxes = plt.subplots(nplot,nplot, sharex=True, sharey=False)
else:
    lossfig = ax1 = ax2 = trainf = trainaxes = testf = testaxes = validatef = validateaxes = []
    
logepoch = []

    

if ismode == "Train" or ismode == "continue":
    start_epoch = 1
    if ismode == "continue":
        thisrunfilename = args.file_root
        isplot = not args.noplot
        with open(thisrunfilename+".pickle",'rb') as f:
            model,sigma,isBCE,train_indi, validation_indi, test_indi,alltrain_loss,allvalidation_BCE,alltest_BCE,epoch,nchannel,args,nvalidate,ntest,T = pickle.load(f)        
        start_epoch = epoch
        print("continue Training with")
        print(args)
    npara = 0
    for i in model.parameters():
        npara = npara + i.numel()
        
    if islog:
        logfilename = os.path.join(args.file_root,args.logfile)
        flog = open(logfilename, 'w') 
        flog.write("# of parameters is {}\n".format(npara))
        flog.write(str(args))
        flog.write("\n")
    for epoch in range(start_epoch, start_epoch+args.epochs + 1):
        issave = 0 
        isplot = False

        if epoch % args.log_interval == 1:
            issave = 1
            isplot = not args.noplot
        train_loss,train_BCE,train_KLD,train_indi = train(epoch,train_loader,isTrain=True,issave=issave,iscuda=False,isplot=isplot,pltaxes=(trainf,trainaxes),pltfile = thisrunfilename+"train.eps") 
        alltrain_loss.append(train_loss)
        alltrain_BCE.append(train_BCE)
        alltrain_KLD.append(train_KLD)
        
        if epoch % args.log_interval == 1:
            logepoch.append(epoch)            
            validation_loss,validation_BCE,validation_KLD,validation_indi = train(epoch,validate_loader,isTrain=False,issave=issave,iscuda=False,isplot=isplot,pltaxes=(validatef,validateaxes),pltfile = thisrunfilename+"validation.eps") 
            allvalidation_loss.append(validation_loss)
            allvalidation_BCE.append(validation_BCE)
            allvalidation_KLD.append(validation_KLD)    
            
            test_loss,test_BCE,test_KLD,test_indi = train(epoch,test_loader,isTrain=False,iscuda=False,issave=issave,isplot=isplot,pltaxes=(testf,testaxes),pltfile = thisrunfilename+"test.eps")
            alltest_loss.append(test_loss)
            alltest_BCE.append(test_BCE)
            alltest_KLD.append(test_KLD)    

            thresh = np.percentile(train_indi["BCE"],99.5)
            nvalidate = np.sum(validation_indi["BCE"]>thresh)
            ntest = np.sum(test_indi["BCE"]>thresh)
            fpr, tpr, thresholds = metrics.roc_curve(np.hstack((validation_indi["BCE"]*0,test_indi["BCE"]*0+1)), np.hstack((validation_indi["BCE"],test_indi["BCE"])))
            auc = metrics.auc(fpr, tpr)
            
            T = (np.mean(test_indi["BCE"]) - np.mean(validation_indi["BCE"]))/(np.sqrt(np.std(test_indi["BCE"])**2+np.std(validation_indi["BCE"])**2))
            print("Model parameters {}, Incontrol false detection {}, OC Detection {} with AUC {} and T {}".format(npara,nvalidate,ntest,auc,T))

            with open(thisrunfilename+".pickle",'wb') as f:
                pickle.dump([model,sigma,isBCE,train_indi, validation_indi, test_indi,alltrain_loss,allvalidation_BCE,alltest_BCE,epoch,nchannel,args,nvalidate,ntest,auc], f)
                
            if isplot:    
                ax1.plot(range(epoch),alltrain_BCE,'k',logepoch,allvalidation_BCE,'b',logepoch,alltest_BCE,'r')
                ax1.legend(['Train', 'Validation','Test'])
                ax1.set_ylabel('BCE')
            
                ax2.plot(range(epoch),alltrain_KLD,'k',logepoch,allvalidation_KLD,'b',logepoch,alltest_KLD,'r')
                ax2.set_ylabel('KLD')
                
                ax3.plot(range(epoch),alltrain_loss,'k',logepoch,allvalidation_loss,'b',logepoch,alltest_loss,'r')
                ax3.set_ylabel('Loss')    
                lossfig.savefig(thisrunfilename+'loss.eps', format='eps')    
                ax1.cla()
                ax2.cla()
                ax3.cla()

            if islog:
                tosave = [epoch,auc, train_loss,train_BCE,train_KLD, validation_loss,validation_BCE,validation_KLD, test_loss,test_BCE,test_KLD]
                flog.write(','.join(map(str, tosave)) +"\n")
                
        torch.save(model.state_dict(), thisrunfilename+'model_epoch_DCVAE_1d.pth')
    flog.close()
    

elif ismode == "Test": 
    thisrunfilename = "runDVAEnew2/big"
    with open(thisrunfilename+".pickle",'rb') as f:
        #model,sigma,isBCE,train_indi, validation_indi, test_indi,train_loss,validation_loss,test_loss=pickle.load(f)
        model,sigma,isBCE,train_indi, validation_indi, test_indi,alltrain_loss,allvalidation_BCE,alltest_BCE,epoch,nchannel,args,nvalidate,ntest,T = pickle.load(f)
    nchannel = [30,20,5]
    issave = 1
    loaddata = dataloader(loaddata="Tonnage",format="torch",demean=True,scale=False)
    trainData,validataData = dataloader.splitTensorDataset(loaddata.trainData,ratioTrain=0.8)
    
    train_loader = torch.utils.data.DataLoader(trainData,shuffle=True,batch_size=128)
    validate_loader = torch.utils.data.DataLoader(validataData,shuffle=True,batch_size=128)
    test_loader = torch.utils.data.DataLoader(loaddata.testData,shuffle=True,batch_size=128)
    epoch = 1
    
    train_loss,train_BCE,train_KLD,train_indi = train(epoch,train_loader,isTrain=False,issave=issave,iscuda=False,isplot=False) 
    validation_loss,validation_BCE,validation_KLD,validation_indi = train(epoch,validate_loader,isTrain=False,issave=issave,iscuda=False,isplot=False) 
    test_loss,test_BCE,test_KLD,test_indi = train(epoch,test_loader,isTrain=False,iscuda=False,issave=issave,isplot=False)

        
    thresh = np.percentile(train_indi["BCE"],99.9999)
    nvalidate = np.sum(validation_indi["BCE"]>thresh)
    ntest = np.sum(test_indi["BCE"]>thresh)
        
    npara = 0
    for i in model.parameters():
        npara = npara + i.numel()
    fpr, tpr, thresholds = metrics.roc_curve(np.hstack((validate_score*0,test_score*0+1)), np.hstack((validation_indi["BCE"],test_indi["BCE"])))
    auc = metrics.auc(fpr, tpr)

#    T = (np.mean(test_indi["BCE"]) - np.mean(validation_indi["BCE"]))/(np.sqrt(np.std(test_indi["BCE"])**2+np.std(validation_indi["BCE"])**2))
    print("Model parameters {}, Incontrol false detection {}, OC Detection {} with AUC {}".format(npara,nvalidate,ntest,auc))

    ngrid = 8
    grid_x = norm.ppf(np.linspace(0.05, 0.95, ngrid))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, ngrid))
    
    fig = plt.figure(figsize=(8,8))
    outer_grid = gridspec.GridSpec(5, 5, wspace=0.0, hspace=0.0)
    for iplot in range(25):
        inner_grid = gridspec.GridSpecFromSubplotSpec(2, 2,
            subplot_spec=outer_grid[iplot], wspace=0.0, hspace=0.0)
        i, j = int(iplot/5),iplot%5
        xi = grid_x[i]
        yi = grid_y[j]
        z_sample = Variable(torch.from_numpy(np.array([[xi, yi]])),volatile=True)
        x_decoded = model.decode(z_sample.float().view(1,2,1,1))
        for j in range(4):
            ax = plt.Subplot(fig, inner_grid[j])
            ax.plot(x_decoded[0][0][j].data.numpy(),'k')
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)
        
    
    all_axes = fig.get_axes()

    #show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)
    
    plt.show()


    manifoldf,manifoldaxes = plt.subplots(ngrid,ngrid)
    plt.xticks([], [])
    ichannel = 0
    for i, xi in enumerate(grid_x):
        for j, yi in enumerate(grid_y):            
            z_sample = Variable(torch.from_numpy(np.array([[xi, yi]])),volatile=True)
            x_decoded = model.decode(z_sample.float().view(1,2,1,1))
            manifoldaxes[i][j].plot(x_decoded[0][0][ichannel].data.numpy(),'k')
            manifoldaxes[i][j].axis('off')


    plt.plot(train_indi["mu"][:,0],train_indi["mu"][:,1],'.')

#    digit_size = 28
#    figure = np.zeros((digit_size * n, digit_size * n))
#    
#    for i, yi in enumerate(grid_x):
#        for j, xi in enumerate(grid_y):            
#            z_sample = Variable(torch.from_numpy(np.array([[xi, yi]])),volatile=True)
#            x_decoded = model.decode(z_sample.float().view(1,2))
#            digit = x_decoded[0].view(digit_size, digit_size).data.cpu().numpy()
#            figure[i * digit_size: (i + 1) * digit_size,
#                   j * digit_size: (j + 1) * digit_size] = digit
#    plt.subplot(111)
#    plt.imshow(figure)
#    plt.savefig('model_train.eps', format='eps', dpi=300, bbox_inches='tight')

