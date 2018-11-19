
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










# # #Model architecture

model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4096)
model.load_state_dict(torch.load('params.ckpt'))



model.cuda()    
     
    

model.eval()

# Download and construct CIFAR-10 dataset.
train_dataset = DatasetFolder(root='./tiny-imagenet-200/train',
                                transform=transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()
                                                              ]),
                             train=False)
                                            
batch_size = 350
                  
# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, num_workers=32) 



train_embeds = []
train_labels = []
train_paths = []

with torch.no_grad():


    for query_images,labels,paths in train_loader:
        Q = Variable(query_images).cuda()

        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        Q_output = model(Q)
        train_embeds.append(Q_output.data.cpu().numpy())
        train_labels.append(labels)
        train_paths.append(paths)



train_paths_ = np.array([item for sublist in train_paths for item in sublist]).reshape(-1,1)

train_labels_ = np.array([item for sublist in train_labels for item in sublist]).reshape(-1,1)
print('setting KNN')
neigh = NearestNeighbors(n_neighbors=30, metric = 'euclidean' ,n_jobs=32)
print('converting train embeds to array')
train_embeds_ = np.array([item for sublist in train_embeds for item in sublist])
print('fitting KNN')
neigh.fit(train_embeds_)

n_neighbors = 30

train_accu =[]

for i in range(len(train_embeds_)):
    print('returning neighbor indices')
    nns =  neigh.kneighbors(X=train_embeds_[i].reshape(1,-1),return_distance=False)
    print('getting labels based on indices')
    nns_labels = np.array(train_labels_[nns].reshape(1,n_neighbors))
    print('calc accu')
    print(nns_labels)
    print(train_labels_[i])
    train_accu.append(np.count_nonzero(nns_labels == train_labels_[i])/30)
    print (np.mean(train_accu))



model.cuda()    
     

model.eval()

######## test dataset accuracy

# Download and construct CIFAR-10 dataset.
test_dataset = DatasetFolder(root='./tiny-imagenet-200/val',
                                transform=transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()
                                                              ]),
                             train=False, test=True)
                                            
batch_size = 350
                  
# Data loader (this provides queues and threads in a very simple way).
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, num_workers=32) 



test_embeds = []
test_labels = []

with torch.no_grad():


    for query_images,labels,paths in test_loader:
        Q = Variable(query_images).cuda()

        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        Q_output = model(Q)
        test_embeds.append(Q_output.data.cpu().numpy())
        test_labels.append(labels)


test_labels_ = np.array([item for sublist in test_labels for item in sublist]).reshape(-1,1)


print('converting test embeds to array')
test_embeds_ = np.array([item for sublist in test_embeds for item in sublist])

test_accu =[]

for i in range(len(test_embeds_)):
    print('returning neighbor indices')
    nns =  neigh.kneighbors(X=test_embeds_[i].reshape(1,-1),return_distance=False)
    print('getting labels based on indices')
    nns_labels = np.array(train_labels_[nns].reshape(1,n_neighbors))
    print('calc accu')
    print(nns_labels)
    print(test_labels_[i])
    test_accu.append(np.count_nonzero(nns_labels == test_labels_[i])/30)
    print (np.mean(test_accu))



################# 5 sample validation example

model.cuda()    
     

model.eval()


# Download and construct CIFAR-10 dataset.
test_dataset = DatasetFolder(root='./tiny-imagenet-200/val',
                                transform=transforms.Compose([transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()
                                                              ]),
                             train=False, test=True, sample=True)
                                            
batch_size = 1
                  
# Data loader (this provides queues and threads in a very simple way).
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=False, num_workers=2) 



test_embeds = []
test_labels = []
test_paths = []

with torch.no_grad():


    for query_image,label,path,negative_image1,target_neg1,path_neg1,negative_image2,target_neg2,path_neg2,negative_image3,target_neg3,path_neg3,negative_image4,target_neg4,path_neg4 in test_loader:
                                 
        Q = Variable(query_image).cuda()
        N1 = Variable(negative_image1).cuda()
        N2 = Variable(negative_image2).cuda()
        N3 = Variable(negative_image3).cuda()
        N4 = Variable(negative_image4).cuda()

        val = 1

        #Forward propagation of the model, i.e. calculate the hidden units and the output.
        Q_output = model(Q)
        test_embeds.append(Q_output.data.cpu().numpy())
        test_labels.append(label)
        test_paths.append(path)
        
        N1_output = model(N1)
        test_embeds.append(N1_output.data.cpu().numpy())
        test_labels.append(target_neg1)
        test_paths.append(path_neg1)
        
        N2_output = model(N2)
        test_embeds.append(N2_output.data.cpu().numpy())
        test_labels.append(target_neg2)
        test_paths.append(path_neg2)
        
        N3_output = model(N3)
        test_embeds.append(N3_output.data.cpu().numpy())
        test_labels.append(target_neg3)
        test_paths.append(path_neg3)
        
        N4_output = model(N4)
        test_embeds.append(N4_output.data.cpu().numpy())
        test_labels.append(target_neg4)
        test_paths.append(path_neg4)

        if val == 1 :
            break    


test_labels_ = np.array([item for sublist in test_labels for item in sublist]).reshape(-1,1)

test_paths_ = np.array([item for sublist in test_paths for item in sublist]).reshape(-1,1)

print('converting test embeds to array')
test_embeds_ = np.array([item for sublist in test_embeds for item in sublist])

nns_paths = []
nns_dist = []
for i in range(5):
    print('returning neighbor indices')
    nns =  neigh.kneighbors(X=test_embeds_[i].reshape(1,-1),n_neighbors=100000,return_distance=True)
    print('getting labels based on indices')
    nns_paths.append( np.array(train_paths_[nns[1][0][0:10]].reshape(1,10)) )
    nns_paths.append( np.array(train_paths_[nns[1][0][-10:]].reshape(1,10)) )
    nns_dist.append( nns[0][0][0:10].reshape(1,10) )
    nns_dist.append( nns[0][0][-10:].reshape(1,10) )


    
np.save('samplepaths1.npy', test_paths_ )    
np.save('paths1.npy', np.array(nns_paths) )
np.save('distances1.npy', np.array(nns_dist) )










