import glob
import numpy as np
import cv2
import torchvision.transforms as transforms
import kornia.color as colconv
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from utilities.customUtils import *
from dataTools.dataNormalization import *
from dataTools.customTransform import *
import os
from skimage.transform import  resize
from skimage import color  # require skimage


def noisy(image, noise_typ="gauss", povMin = 1, povMax = 0.2):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.1
        sigma = random.uniform(var**povMin, var**povMax) #var**0.3 

        #print("Gaussian sigma",sigma)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy


# img = cv2.imread("000000004134.jpg")
# print(img.shape)
# noisy_image = noisy(img/255.)
# cv2.imwrite("noisyImgCV.png", noisy_image*255)
# print(noisy_image.shape)
# img  = noisy_image#cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)/255.0
# img_input = img.astype(np.float32)#cv2.resize(img, (640, 640)).astype(np.float32)
# img = transformCV(img_input.copy()).cuda()#torch.from_numpy(img).unsqueeze(0)
# rgb = colconv.bgr_to_rgb(img)
# lab = colconv.rgb_to_lab(rgb)


class customDatasetReader(Dataset):
    def __init__(self, image_list, imagePathGT, height, width, cropFactor =128, resizeFactor=128, transformation=True):
        self.image_list = image_list
        self.imagePathGT = imagePathGT
        self.transformLR = transforms
        self.cropFactor = cropFactor
        self.resizeFactor= resizeFactor
        self.imageH = height
        self.imageW = width
        self.normalize = transforms.Normalize(normMean, normStd)
        self.var = 0.1
        self.mean = 0.0
        self.pov = 0.3

    def rgbTolab(self, sample, resizeImg=True):
        
        im = np.array(sample)
        # if resizeImg:
        #     print("***********"*30)
        #     im = resize(sample, (256,256), anti_aliasing=True)
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        L = lab_t[[0], ...] / 50.0 - 1.0
        ab = lab_t[[1, 2], ...] / 110.0

        return torch.cat([L, ab], dim=0)

    def __len__(self):
        return (len(self.image_list))
    
    def __getitem__(self, i):

        # Read Images
        #print ("print i",i, i+1)
        # try:    
        #     self.sampledImage = cv2.imread(self.image_list[i])#Image.open(self.image_list[i])
        # except:
        #     self.sampledImage = cv2.imread(self.image_list[i + 1])#Image.open(self.image_list[i + 1])
        #     os.remove(i)
        #     print ("File deleted:", i)
        #     i += 1


        self.transformHRGT = transforms.Compose([transforms.Resize((self.resizeFactor,self.resizeFactor), interpolation=Image.BICUBIC),
                                                transforms.ToTensor(),
                                                self.normalize,
                                                ])
        
        # #self.gtImageFileName = self.imagePathGT + extractFileName(self.image_list[i])
        # #self.gtImage = Image.open(self.gtImageFileName)

        # # Transforms Images for training 

            
        sample = self.image_list[i].replace("/gtPatch", "/inputPatch")
        #print(sample,self.image_list[i] )
        self.sampledImage =  Image.open(sample).convert("RGB")#(cv2.cvtColor(cv2.imread(sample), cv2.COLOR_BGR2RGB)/255.0).astype(np.float32)#Image.open(self.image_list[i])
        #print(self.image_list[i], self.image_list[i].replace("gtPatch", "inputPatch"))
        #self.gtImageFileName = self.imagePathGT + extractFileName(self.image_list[i])
        self.gtImage = Image.open(self.image_list[i]).convert("RGB")#(cv2.cvtColor(cv2.imread(self.image_list[i]), cv2.COLOR_BGR2RGB)/255.0).astype(np.float32)#Image.open(self.gtImageFileName)
        
        
        
        #print(self.sampledImage.shape[0]//2)
        width, height = self.gtImage.size

        # # Define the size of the random crop
        crop_size = 128

        # Calculate random crop coordinates
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size


        

        self.sampledImage  = self.sampledImage.crop((left, top, right, bottom))
        self.gtImage  = self.gtImage.crop((left, top, right, bottom))
        self.gtImage = self.rgbTolab(self.gtImage)
        self.sampledImage = self.rgbTolab(self.sampledImage)
        #print(self.gtImage.shape)

        # #self.sampledImage  = self.sampledImage.crop((left, top, right, bottom))#[ranHeight : ranHeight + self.cropFactor, ranWidth : ranWidth + self.cropFactor, ]
        # #self.sampledImage = cv2.resize(self.sampledImage, (self.resizeFactor, self.resizeFactor))
        
        # #self.gtImage  = self.gtImage.crop((left, top, right, bottom))#[ranHeight : ranHeight + self.cropFactor, ranWidth : ranWidth + self.cropFactor, ]
        # #self.gtImage = cv2.resize(self.gtImage, (self.resizeFactor, self.resizeFactor))
        # self.gtTensor = self.transformHRGT(self.gtImage).cuda()
        # #cv2.imwrite("noisyImgCV.png", noisy_image*255)
        # #print(noisy_image.shape)
        # #img  = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)#/255.0
        # #img_input = noisy_image.astype(np.float32)#cv2.resize(img, (640, 640)).astype(np.float32)
        # img = self.transformHRGT(self.sampledImage).cuda()#torch.from_numpy(img).unsqueeze(0)
        # #print(img.shape)
        # #rgb = colconv.bgr_to_rgb(img)
        # #lab = colconv.rgb_to_lab(img)
        # #print(lab)
        
        # #t[:, permute]
        # self.noisy_lab_tensor = img#colconv.rgb_to_lab(rgb)
        
        # # sigma = random.uniform(0, self.var ** self.pov)
        # # noiseModel = torch.clamp(torch.randn(self.gtImageHR.size()).uniform_(0, 1.) * sigma  + 0., 0., 1.)
        # # #noiseModel = NoiseModeling(self.gtImageHR)
    
        # '''self.transformRI = transforms.Compose([transforms.ToTensor(),
        #                                         self.normalize,
        #                                         AddGaussianNoise(noiseModel)
        #                                     ])'''
        # #print(self.gtImageHR.shape, noiseModel.shape)
        # #self.inputImage = self.gtImageHR + noiseModel#self.transformRI(self.sampledImage)
        # #print (self.gtImageHR.max(), self.gtImageHR.min(), self.inputImage.max(), self.inputImage.min())
        

        return self.sampledImage, self.gtImage
