from google.colab import drive
import os

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
  list_signal.append(return_info[3])

print(list_signal)
print(len(list_signal))

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

%cd /content/drive/My\ Drive
%cd DATA/

print(len(list_images))
print(list_images[1])

useful_signals =[]
for i in range(len(list_signal)):
    if(i % 17 == 0):
        useful_signals.append(list_signal[i])

print(len(useful_signals))

sample = {'image': list_images, 'signals': useful_signals}
print(sample)

from torch.utils.data import Dataset, DataLoader
Data_loader = DataLoader(sample, batch_size=4,
                        shuffle=True)
for i_batch, sample_batched in enumerate(Data_loader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['signals'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break


class PhysNetED(nn.Module):
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
        self.poolspa = nn.AdaptiveAvgPool3d((frames,1,1))git