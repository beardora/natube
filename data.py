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


class dataloader():
    def __init__(self,loaddata = "MNIST", format="torch", demean = False, scale = True,\
    rootlocation = os.path.join(os.path.expanduser('~'),'dropbox','Pytorch','data'),\
    alltransform = [],pickclass = range(9)):
        self.loaddata = loaddata
        self.rootlocation = rootlocation
        self.format = format
        
        self.demean = demean
        self.mean = []
        self.scale = scale
        self.alltransform = alltransform
        self.trainData = []
        self.validationData = []
        self.testData = []
        self.pickclass = pickclass
        if self.loaddata == "MNIST":
            if len(alltransform)==0:
                self.alltransform = transforms.Compose([
                    transforms.Scale(32),
                    transforms.ToTensor(), 
                    ])        
            self.trainData,self.validationData,self.testData = self.loadmnist()
            
        elif self.loaddata == "Tonnage":
            self.trainData, self.testData = self.loadTonnage()
        elif self.loaddata == "OG":
            if len(alltransform)==0:
                self.alltransform = transforms.Compose([
                    transforms.RandomSizedCrop(50),
                    transforms.Scale(32),
                    transforms.RandomHorizontalFlip(),                                                     
                    transforms.ToTensor(), 
                    transforms.Lambda(lambda x: x.mean(dim=0)),
                    transforms.Lambda(lambda x: (x - x.min())/(x.max()-x.min())),
                    ])        
        elif self.loaddata == "Solar":
            trainDataMat = sio.loadmat(os.path.join(self.rootlocation,"Solar","data.mat"))
            self.trainData = trainDataMat['data']
        elif self.loaddata == "Gaolei":
            if len(alltransform)==0:
                self.alltransform = transforms.Compose([
                    transforms.Scale((390,64)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ])        

        elif self.loaddata == "dose":
            self.trainData, self.testData = self.loadgendose(1000,100)    

    def gendose(self,nrep):
        u = np.linspace(0,10,128)
        x = np.zeros((nrep,1,len(u)))
        for i in range(nrep):
            theta1 = 1+0*np.random.randn()
            theta2 = 0+0*np.random.randn()
            theta3 = 5+0.3*np.random.randn()
            theta4 = 5+0.3*np.random.randn()
            x[i,0,:] = theta1 + (theta2 - theta1 )/(1 + (u/theta3)**theta4) + np.random.randn(1,128)*0.01
        return x*100

    def loadgendose(self,ntrain,ntest):
        trainTarget = torch.zeros(ntrain)
        trainData = self.gendose(ntrain)
        trainDataWrapper = torch.utils.data.TensorDataset(torch.from_numpy(trainData).float(),trainTarget)
        testTarget = torch.zeros(ntest)
        testData = self.gendose(ntest)
        testDataWrapper = torch.utils.data.TensorDataset(torch.from_numpy(testData).float(),testTarget)
        return trainDataWrapper, testDataWrapper

    def loadnano(self):
        preroot = os.path.join(self.rootlocation,'Nanotube')
        traindir = os.path.join(preroot,'Normal')
        testdir = os.path.join(preroot,'Anomaly')
        train_loader = datasets.ImageFolder(traindir,transform=self.alltransform)
        train_loader = self.cleanClass(train_loader)
        test_loader = datasets.ImageFolder(testdir,transform=self.alltransform)
        test_loader = self.cleanClass(test_loader)
        return train_loader,test_loader

    def loadmnist(self):
        preroot = os.path.join(self.rootlocation,'MNIST')
        traindir = os.path.join(preroot,'Train')
        testdir = os.path.join(preroot,'Test')
        train_loader = datasets.ImageFolder(traindir,transform=self.alltransform)
        validation_loader = datasets.ImageFolder(testdir,transform=self.alltransform)
        train_loader = self.cleanClass(train_loader)
        validation_loader = self.cleanClass(validation_loader)
        train_loader,test_loader = dataloader.splitClassImageFolder(train_loader,self.pickclass)
        validation_loader,_ = dataloader.splitClassImageFolder(validation_loader,self.pickclass)
        _,test_loader = dataloader.splitClassImageFolder(test_loader,self.pickclass)
        return train_loader,validation_loader,test_loader
        
    def loadTonnage(self):
        trainDataMat = sio.loadmat(os.path.join(self.rootlocation,"Tonnage","aligned_noisy_normal.mat"))
        testDataMat = sio.loadmat(os.path.join(self.rootlocation,"Tonnage","Fault_station4.mat"))
        trainData = np.zeros((306,1,4,1200))
        trainData[:,0,0,:] = trainDataMat['AX1'][:,:-1]
        trainData[:,0,1,:] = trainDataMat['AX2'][:,:-1]
        trainData[:,0,2,:] = trainDataMat['AX3'][:,:-1]
        trainData[:,0,3,:] = trainDataMat['AX4'][:,:-1]

        testData = np.zeros((69,1,4,1200))
        testData[:,0,0,:] = testDataMat['AX1'][:,:-1]
        testData[:,0,1,:] = testDataMat['AX2'][:,:-1]
        testData[:,0,2,:] = testDataMat['AX3'][:,:-1]
        testData[:,0,3,:] = testDataMat['AX4'][:,:-1]
        
        '''
        if self.demean:
            self.mean = trainData.mean(axis=0)
            trainData -=  self.mean
            testData -= self.mean
        '''
        
        if self.scale:
            minvalue = min(trainData.min(),testData.min())
            maxvalue = max(trainData.max(),testData.max())
            trainData = (trainData-minvalue)/(maxvalue-minvalue)
            testData = (testData-minvalue)/(maxvalue-minvalue)
        if self.format == "numpy":
            return trainData, testData
        elif self.format == "torch": 
            if self.pickclass == "all":
                trainTarget = torch.zeros(306)
                testTarget = torch.ones(69)
                labels = torch.cat((trainTarget,testTarget),0)
                data = np.concatenate((trainData,testData),0)
                allDataWrapper = torch.utils.data.TensorDataset(torch.from_numpy(data).float(),labels)
                return allDataWrapper,0
            else:
                trainTarget = torch.zeros(306)
                trainDataWrapper = torch.utils.data.TensorDataset(torch.from_numpy(trainData).float(),trainTarget)
                testTarget = torch.zeros(69)
                testDataWrapper = torch.utils.data.TensorDataset(torch.from_numpy(testData).float(),testTarget)
                return trainDataWrapper, testDataWrapper
            

            
    def loadOGdata(self):
        filename = os.path.join(self.rootlocation,'Rolling')
        alldata = datasets.ImageFolder(root=filename,transform = self.alltransform)
        alldata = self.cleanClass(alldata)
        return alldata
                    
    def loadGaoleidata(self):
        filename = os.path.join(self.rootlocation,'Gaolei')
        alldata = datasets.ImageFolder(root=filename,transform = self.alltransform)
        alldata = self.cleanClass(alldata)
        return alldata

    @staticmethod
    def cleanClass(alldata,removestr = '.DS_Store'):
        removestr = '.DS_Store'
        if removestr in alldata.classes: alldata.classes.remove(removestr)
        iremove = 0
        for key in alldata.class_to_idx:        
            if key == removestr:
                iremove = 1
        for key in alldata.class_to_idx:        
            alldata.class_to_idx[key] = alldata.class_to_idx[key] - iremove
        alldata.class_to_idx.pop(removestr, None)    
        allimgs = []
        for idxdata,(img,target) in enumerate(alldata.imgs):            
            allimgs.append((img,target-iremove))    
        alldata.imgs = allimgs
        return alldata
        
    @staticmethod
    def splitClassImageFolder(alldata,pickclass):
        alltarget = np.array([target for idata,target in alldata.imgs])
        iremove = np.min(alltarget)
        uniquetarget, counts = np.unique(alltarget, return_counts=True)
        nowcounts = counts*0
        trainimgs = []
        testimgs = []
        for idxdata,(img,target) in enumerate(alldata.imgs):  
            if target in pickclass:
                trainimgs.append((img,target))
            else:
                testimgs.append((img,target))   
        trainloader = copy.deepcopy(alldata)
        trainloader.imgs = trainimgs
        testloader = (alldata)
        testloader.imgs = testimgs
        return trainloader,testloader
        
    @staticmethod
    def splitImageFolderDataset(alldata,ratioTrain=0.8):
        alltarget = np.array([target for idata,target in alldata.imgs])
#        iremove = np.min(alltarget)
        uniquetarget, counts = np.unique(alltarget, return_counts=True)
        nowcounts = counts*0
        trainimgs = []
        testimgs = []
        for idxdata,(img,target) in enumerate(alldata.imgs):            
            idxtarget = np.where(uniquetarget==target)[0][0]
            nowcounts[idxtarget] = nowcounts[idxtarget] + 1
            if ratioTrain < 1:
                if nowcounts[idxtarget] < counts[idxtarget]*ratioTrain:
                    trainimgs.append((img,target))            
                else:
                    testimgs.append((img,target))    
            else:
                if nowcounts[idxtarget] < ratioTrain:
                    trainimgs.append((img,target))            
                else:
                    testimgs.append((img,target))    
        trainloader = copy.deepcopy(alldata)
        trainloader.imgs = trainimgs
        testloader = (alldata)
        testloader.imgs = testimgs
        return trainloader,testloader        

    @staticmethod
    def splitTensorDataset(train_loader,ratioTrain=0.8,randomize = False):
        ndata = train_loader.target_tensor.size()[0]
        try: 
            alltarget = np.array([int(target[0]) for idata,target in train_loader])
        except:
            alltarget = np.array([int(target) for idata,target in train_loader])
            pass
        
        uniquetarget, counts = np.unique(alltarget, return_counts=True)
        nowcounts = counts*0
        if not randomize:
            indices_train = torch.LongTensor(range(int(ndata*ratioTrain)))
            indices_test = torch.LongTensor(range(int(ndata*ratioTrain),ndata))
#        else:
            
        
        trainData_tensor = torch.index_select(train_loader.data_tensor, 0, indices_train)
        trainTarget_tensor = torch.index_select(train_loader.target_tensor, 0, indices_train)
        traindata = torch.utils.data.dataset.TensorDataset(data_tensor=trainData_tensor,target_tensor=trainTarget_tensor)            
        
        testData_tensor = torch.index_select(train_loader.data_tensor, 0, indices_test)
        testTarget_tensor = torch.index_select(train_loader.target_tensor, 0, indices_test)
        testdata = torch.utils.data.dataset.TensorDataset(data_tensor=testData_tensor,target_tensor=testTarget_tensor)            
        return traindata,testdata        





