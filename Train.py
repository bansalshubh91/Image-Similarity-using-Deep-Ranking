
# coding: utf-8

# In[ ]:





import numpy as np
import h5py
import time
import copy
from random import randint
import torch 
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from skimage import io, transform
import os
from torch.utils.data import Dataset
import os.path
from PIL import Image
import random
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from torchvision.models.resnet import model_urls






def find_classes(dir):
    classes = [d for d in os.listdir(dir)]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target, 'images')
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):

                path = os.path.join(root, fname)
                item = (path, target)
                images.append(item)

    return images

def make_test_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    d = os.path.join(dir, 'images')
    r = np.array([x.split('\t') for x in open(os.path.join(dir, 'val_annotations.txt')).readlines()])

    for i in range(len(r)):

        path = os.path.join(d, r[i][0])
        item = (path, r[i][1])
        images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class DatasetFolder(Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, loader = pil_loader, transform=None, train=True, test=None, sample=None):
        
        
        
        if test is not None:
            samples = make_test_dataset(root)
        else:
            classes, class_to_idx = find_classes(root)
            samples = make_dataset(root, class_to_idx)
            self.classes = classes
            self.class_to_idx = class_to_idx
        
        self.root = root
        self.loader = loader

        
        self.samples = samples

        self.transform = transform
        self.train = train
        self.sample = sample

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        query_image = self.loader(path)
        
        if self.transform is not None:
            query_image = self.transform(query_image)
            
        if self.train is not True and self.sample is None:
            return query_image,target,path          
        

        if self.train is not True and self.sample is not None:
            
            negative1 = random.randint(0,9999)          
            path_neg1, target_neg1 = self.samples[negative1]
            while target_neg1 == target:
                negative1 = random.randint(0,9999)
                path_neg1, target_neg1 = self.samples[negative1]

            negative2 = random.randint(0,9999)          
            path_neg2, target_neg2 = self.samples[negative2]
            while target_neg2 in [target, target_neg1]:
                negative2 = random.randint(0,9999)
                path_neg2, target_neg2 = self.samples[negative2]     

            negative3 = random.randint(0,9999)          
            path_neg3, target_neg3 = self.samples[negative3]
            while target_neg3 in [target, target_neg1, target_neg2]:
                negative3 = random.randint(0,9999)
                path_neg3, target_neg3 = self.samples[negative3]

            negative4 = random.randint(0,9999)          
            path_neg4, target_neg4 = self.samples[negative4]
            while target_neg4 in [target, target_neg1, target_neg2, target_neg3]:
                negative4 = random.randint(0,9999)
                path_neg4, target_neg4 = self.samples[negative4]

            negative_image1 = self.loader(path_neg1) 
            negative_image2 = self.loader(path_neg2) 
            negative_image3 = self.loader(path_neg3) 
            negative_image4 = self.loader(path_neg4) 

            if self.transform is not None:
                negative_image1 = self.transform(negative_image1)
                negative_image2 = self.transform(negative_image2)
                negative_image3 = self.transform(negative_image3)
                negative_image4 = self.transform(negative_image4) 

            return query_image,target,path,negative_image1,target_neg1,path_neg1,negative_image2,target_neg2,path_neg2,negative_image3,target_neg3,path_neg3,negative_image4,target_neg4,path_neg4
                    
                    

        
        positive = random.randint(0,99999)
        path_pos, target_pos = self.samples[positive]
        while target_pos != target or positive == index:
            positive = random.randint(0,99999)
            path_pos, target_pos = self.samples[positive]
            

        negative = random.randint(0,99999)          
        path_neg, target_neg = self.samples[negative]
        while target_neg == target:
            negative = random.randint(0,99999)
            path_neg, target_neg = self.samples[negative]          
        
        
        positive_image = self.loader(path_pos)
        negative_image = self.loader(path_neg)          
                  
        if self.transform is not None:
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)      
        
        
        
        return query_image,positive_image,negative_image


    def __len__(self):
        return len(self.samples)





# Download and construct CIFAR-10 dataset.
train_dataset = DatasetFolder(root='./tiny-imagenet-200/train',
                                transform=transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.RandomHorizontalFlip(),
                                                              transforms.ToTensor()
                                                              ]),
                              train=True)
                                            
batch_size = 16
                  
# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True, num_workers=32) 






# # #Model architecture

model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4096)
model.load_state_dict(torch.load('params.ckpt'))



model.cuda()


# # #Stochastic gradient descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 15


model.train()

train_loss = []

triplet_loss = nn.TripletMarginLoss(margin=1.0)



#Train Model
for epoch in range(num_epochs):
    
    for query_images,positive_images,negative_images in train_loader:
        Q, P, N = Variable(query_images).cuda(), Variable(positive_images).cuda(), Variable(negative_images).cuda()


        
        #PyTorch "accumulates gradients", so we need to set the stored gradients to zero when there’s a new batch of data.
        optimizer.zero_grad()
        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        Q_output = model(Q)
        P_output = model(P)
        N_output = model(N)
        
       
        
        #The objective function is the negative log-likelihood function.
        loss = triplet_loss(Q_output, P_output, N_output)
        #This calculates the gradients (via backpropagation)
        loss.backward()
        train_loss.append(loss.data[0])

        optimizer.step()
    loss_epoch = np.mean(train_loss)
    torch.save(model.state_dict(), 'params.ckpt')
    torch.save(model, 'model.ckpt')
    print(epoch, loss_epoch)

    
     



