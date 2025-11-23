import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os 
import glob
from shutil import copyfile
import matplotlib.pyplot as plt
from utilities.customUtils import *
#from dataTools.sampler import *
import numpy as np
import cv2
from PIL import Image
from dataTools.dataNormalization import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def ensemble_predict(models, x):
    # Aggregate predictions from each model using simple averaging
    predictions = []
    with torch.no_grad():
        for model in models:
            _, output = model(x)
            predictions.append(output)
    return sum(predictions) / len(predictions)
class AddGaussianNoise(object):
    def __init__(self, noiseLevel):
        self.var = 0.1
        self.mean = 0.0
        self.noiseLevel = noiseLevel
        
    def __call__(self, tensor):
        sigma = self.noiseLevel/100.
        noisyTensor = tensor + torch.randn(tensor.size()).uniform_(0, 1.) * sigma  + self.mean
        return noisyTensor 
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.var)
from skimage import color 
def rgbTolab( sample, resizeImg=True):
    
    im = np.array(sample)
    # if resizeImg:
    #     print("***********"*30)
    #     im = resize(sample, (256,256), anti_aliasing=True)
    lab = color.rgb2lab(im).astype(np.float32)
    lab_t = transforms.ToTensor()(lab)
    L = lab_t[[0], ...] / 50.0 - 1.0
    ab = lab_t[[1, 2], ...] / 110.0

    return torch.cat([L, ab], dim=0)

def lab2rgb(L, AB):
    """Convert an Lab tensor image to a RGB numpy output
    Parameters:
        L  (1-channel tensor array): L channel images (range: [-1, 1], torch tensor array)
        AB (2-channel tensor array):  ab channel images (range: [-1, 1], torch tensor array)

    Returns:
        rgb (RGB numpy image): rgb output images  (range: [0, 255], numpy array)
    """
    AB2 = AB * 110.0
    L2 = (L + 1.0) * 50.0
    #print("shape", AB2.shape, L2.shape)

    Lab = torch.cat([L2, AB2], dim=0)
    #print("concat LAB shape", Lab.shape)
    Lab = Lab.data.cpu().float().numpy()
    Lab = np.transpose(Lab.astype(np.float64), (1, 2, 0))
    rgb = color.lab2rgb(Lab) * 255
    return rgb

class inference():
    def __init__(self, gridSize, inputRootDir, outputRootDir, modelName, resize = None, validation = None ):
        self.inputRootDir = inputRootDir
        self.outputRootDir = outputRootDir
        self.gridSize = gridSize
        self.modelName = modelName
        self.resize = resize
        self.validation = validation
        self.unNormalize = UnNormalize()
    


    def inputForInference(self, imagePath):
        img = Image.open(imagePath).convert("RGB")
        img = img.resize((1024, 1024))

        # crop_size = 512

        # # Calculate random crop coordinates
        # left = 256
        # top = 256
        # right = left + crop_size
        # bottom = top + crop_size


        

        # img  = img.crop((left, top, right, bottom))

        #testImg = rgbTolab(img).unsqueeze(0)

        # if self.resize:
        #     #resize(256,256)
        #     transform = transforms.Compose([ transforms.Resize(self.resize, interpolation=Image.BICUBIC) ])
        #     img = transform(img)

        transform = transforms.Compose([ transforms.Resize((1024, 1024), interpolation=Image.BICUBIC), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(normMean, normStd),
                                        #AddGaussianNoise(noiseLevel=noiseLevel)
                                        ])
        

        testImg = transform(img).unsqueeze(0)

        return testImg 


    def saveModelOutput(self, modelOutput, inputImagePath, noiseLevel=None, step = None, ext = ".png"):
        datasetName = inputImagePath.split("/")[-2]
        if step:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True)  + \
                            + "_" + self.modelName + "_" + str(step) + ext
        else:
            imageSavingPath = self.outputRootDir + self.modelName  + "/"  + datasetName + "/" + extractFileName(inputImagePath, True) + "_" +self.modelName + ext#\
                            #"_sigma_" + str(noiseLevel) + "_FT_" + self.modelName + ext
        #print(imageSavingPath)
        #print("outshape", modelOutput[0].shape)
        #rgbImg = lab2rgb(modelOutput[0][:1,:,:], modelOutput[0][1:,:,:])
        #cv2.imwrite(imageSavingPath, rgbImg[...,::-1]) 
        save_image(self.unNormalize(modelOutput[0]), imageSavingPath)

    

    def testingSetProcessor(self):

        testSets = glob.glob(self.inputRootDir+"*/")
        #print (testSets)
        if self.validation:
            #print(self.validation)
            testSets = testSets[:1]
        #print (testSets)
        testImageList = []
        for t in testSets:
            #print (t.split("/")[-2])
            testSetName = t.split("/")[-2]
            createDir(self.outputRootDir + self.modelName  + "/" + testSetName)
            imgInTargetDir = imageList(t, False)
            testImageList += imgInTargetDir

        return testImageList


