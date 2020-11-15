from google.colab import drive
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#drive.mount("/content/drive")
%cd /content/drive/My Drive/DATASET_1
%ls
%cd 10-gt
%ls
%pwd

filename = "/content/drive/My Drive/DATASET_1/10-gt/gtdump.xmp"
f = open(filename, "r")
list_time =[]
list_pulse =[]
list_signal = []
for lines in f.readlines():
  return_info = lines.split(",")
  list_time.append(return_info[0])
  list_pulse.append(return_info[1])
  list_signal.append(int(return_info[3]))
print(list_signal)
print(len(list_signal))
  #csv_file (string): Path to the csv file with annotations.
  #root_dir (string): Directory with all the images.
  #csv_file = return_info

import cv2 
import os 
import matplotlib.pyplot as plt

cam = cv2.VideoCapture("/content/drive/My Drive/DATASET_1/10-gt/vid.avi")  
list_images =[]
      
    # creating a folder named data 
    #if not os.path.exists('data'): 
%mkdir '/content/drive/My Drive/DATA'
  

%cd
# frame 
currentframe = 0
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
    i = 0
    #plt.imshow(frame)
    #plt.show()
        
    if ret: 
        if(currentframe % 4 == 0):
        # if video is still left continue creating images 
          name = '~/content/drive/My Drive/DATA/' + str(int(currentframe/4 )) + '.jpg'
          list_images.append(frame)
          #name = './content/drive/My Drive/DATA/' + str(int(currentframe/4)) + '.jpg'
          print ('Creating...' + name) 
  
        # writing the extracted images 
          cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 

list_images = list_images[0:296]
print(len(list_images))
print(list_images[1])

useful_signals =[]
for i in range(len(list_signal)):
    if(i % 72 == 0):
        useful_signals.append(list_signal[i])

print(len(useful_signals))


sample = {'image': list_images, 'signals': useful_signals}
print(sample)

 from torch.utils.data import Dataset, DataLoader
 class FaceLandmarksDataset(Dataset):
    def __init__(self,list_images,useful_signals):
  
      self.list_images = list_images
      self.useful_signals = useful_signals

    def __len__(self):
      return len(self.useful_signals)

    def __getitem__(self, idx):
      images = self.list_images[8 * idx: 8 * idx + 8]
      images = np.asarray(images)
      images = np.moveaxis(images,0,3)
      images = np.moveaxis(images,2,0)
      images = np.moveaxis(images,3,1)
      images = images.astype(np.float32)
      signals = self.useful_signals[idx: idx + 1]
      item = [images,signals]
      return item

face_dataset = FaceLandmarksDataset(list_images=list_images,
                                    useful_signals=useful_signals)

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]


    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')

    if i == 3:
        plt.show()
        break


from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
dataloader = DataLoader(face_dataset, batch_size=1,
                        shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):

  print(i_batch, len(sample_batched[0]),
          len(sample_batched[0]))

    # observe 4th batch and stop.
  if i_batch == 8:
    image=sample_batched[0][0]
    print(len(sample_batched[0][0]))
    #imgplot = plt.imshow(image)
    break


import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch
import pdb



class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):  
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4,1,1], stride=[2,1,1], padding=[1,0,0]),   #[1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        
        
        #self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space 
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))

        
    def forward(self, x):	    	# x [3, T, 128,128]
        x_visual = x
        [batch,channel,length,width,height] = x.shape
          
        x = self.ConvBlock1(x)		     # x [3, T, 128,128]
        x = self.MaxpoolSpa(x)       # x [16, T, 64,64]
        
        x = self.ConvBlock2(x)		    # x [32, T, 64,64]
        x_visual6464 = self.ConvBlock3(x)	    	# x [32, T, 64,64]
        x = self.MaxpoolSpaTem(x_visual6464)      # x [32, T/2, 32,32]    Temporal halve
        
        x = self.ConvBlock4(x)		    # x [64, T/2, 32,32]
        x_visual3232 = self.ConvBlock5(x)	    	# x [64, T/2, 32,32]
        x = self.MaxpoolSpaTem(x_visual3232)      # x [64, T/4, 16,16]
        

        x = self.ConvBlock6(x)		    # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)	    	# x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)      # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)		    # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)		    # x [64, T/4, 8, 8]
        x = self.upsample(x)		    # x [64, T/2, 8, 8]
        x = self.upsample2(x)		    # x [64, T, 8, 8]
        
        
        x = self.poolspa(x)     # x [64, T, 1,1]    -->  groundtruth left and right - 7 
        x = self.ConvBlock10(x)    # x [1, T, 1,1]
        
        rPPG = x.view(-1,length)            
        

        return rPPG, x_visual, x_visual3232, x_visual1616

from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse,os
import pandas as pd
import numpy as np
import random
import math
from torchvision import transforms

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()
        return
    def forward(self, preds, labels):       # tensor [Batch, Temporal]
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))

            #if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
            #    loss += 1 - pearson
            #else:
            #    loss += 1 - torch.abs(pearson)
            
            loss += 1 - pearson
            
            
        loss = loss/preds.shape[0]
        return loss

#1. Inference the model
model = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
opt = torch.optim.Adam(model.parameters(),lr=1e-4,betas =(0.9,0.99),eps=1e-8)
print(len(sample_batched))
print(sample_batched[0].shape)
#print(sample_batched[1].shape)
print(sample_batched[0].type())
rPPG, x_visual, x_visual3232, x_visual1616 = model(sample_batched[0])
#2. Normalized the Predicted rPPG signal and GroundTruth BVP signal
rPPG = (rPPG-torch.mean(rPPG)) /torch.std(rPPG)	 	# normalize

list_signal = np.asarray(list_signal, dtype = np.float32)
#print(list_signal)
list_signal = torch.as_tensor(list_signal)
list_signal = (list_signal-torch.mean(list_signal)) /torch.std(list_signal)	 	# normalize
'''
#3. Calculate the loss
evaluator = Neg_Pearson()
loss = evaluator(rPPG, list_signal)
#opt = torch.optim.Adam(model.parameters(),lr=1e-4,betas =(0.9,0.99),eps=1e-8)
opt.zero_grad()
loss.backward()
opt.step()
'''

    loss = 0
    
    #for i in range(rPPG.shape[0]):
    sum_x = torch.sum(rPPG[0])                # x
    sum_y = torch.sum(list_signal[i])               # y
    sum_xy = torch.sum(rPPG[0]*list_signal[i])        # xy
    sum_x2 = torch.sum(torch.pow(rPPG[0],2))  # x^2
    sum_y2 = torch.sum(torch.pow(list_signal[i],2)) # y^2
    N = rPPG.shape[0]
    pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
    if (pearson>=0).data.cpu().numpy():    # torch.cuda.ByteTensor -->  numpy
        loss += 1 - pearson
    else:
        loss += 1 - torch.abs(pearson)
            
    #loss += 1 - pearson       
    print(rPPG.shape[0])      
    loss = abs(loss/rPPG.shape[0])
    print("loss: " + str(float(loss)))