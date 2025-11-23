import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import sys
import glob
import time
import colorama
from colorama import Fore, Style
from etaprogress.progress import ProgressBar
from torchsummary import summary
from ptflops import get_model_complexity_info
from utilities.torchUtils import *
from dataTools.customDataloader import *
from utilities.inferenceUtils import *
from utilities.aestheticUtils import *
from loss.pytorch_msssim import *
from loss.LCLoss import *
from loss.percetualLoss import *
from modelDefinitions.attentionDis import *
from modelDefinitions.NoiseDPN_GEN import *
from torchvision.utils import save_image
from modelDefinitions.forkformer6 import *



def modelLoad(model, checkpointPath, modelName, barLen=64):

        customPrint(Fore.RED + "Loading pretrained weight", textWidth=barLen)

        previousWeight = loadCheckpoints(checkpointPath, modelName)
        model.load_state_dict(previousWeight['stateDictEG'])
        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=barLen)
        return model


model = TwoTineFormerN()#Former(stage=1,n_feat=16,num_blocks=[1,2,2])

modelName = "TwoTineFormer"
checkpointRoot = "/media/sharif/XtrasHD2/modelLog/LowLightENhancement/checkpoint_TwoTineFormer/"
testImagesPath = "/media/sharif/XtrasHD2/LSD/TestingLSD/"
resultDir =  "/media/sharif/XtrasHD2/LLE/modelLog/logDir_TwoTineFormer/ModelOutput/EST/"

pretrainedModel = modelLoad(model, checkpointRoot, modelName)
pretrainedModel = pretrainedModel.cuda()

targetStep = 100000
#checking available path
cpList = glob.glob(checkpointRoot + "*/")
pretrainedModels = [] 
print(len(cpList))
for cp in cpList:
        try:
            if int(cp.split("/")[-2]) > targetStep:
                print(cp.split("/")[-2])
                pretrainedModels.append(modelLoad(model, cp, modelName))
        except:
            print(cp)

print(len(pretrainedModels))
def ensemble_predict(models, x):
    # Aggregate predictions from each model using simple averaging
    predictions = []
    with torch.no_grad():
        for model in models:
            _, output = model(x)
            predictions.append(output)
    return sum(predictions) / len(predictions)


modelInference = inference(gridSize=1, inputRootDir=testImagesPath, outputRootDir=resultDir, modelName=modelName, validation=False)
testImageList = modelInference.testingSetProcessor()

for count, imgPath in enumerate(testImageList):
        #print(imgPath)
        img = modelInference.inputForInference(imgPath)
        img = img.to(device)
        #print("Input shape", img.shape)
        ensemble_prediction = ensemble_predict(pretrainedModels, img)
        #print("prediction shape", ensemble_prediction.shape)
        #modelInference.saveModelOutput(inter, imgPath.replace(".", "_inter."), steps)
        # if img_dim:
                
        #         #print("need to resize", img_dim)
        #         w, h = img_dim
        #         transformResize = transforms.Compose([transforms.Resize((h, w))])
        #         ensemble_prediction = transformResize(ensemble_prediction)
        #print("output shape", ensemble_prediction.shape)
        modelInference.saveModelOutput(ensemble_prediction, imgPath)
        print("Image Inferenced [{}/{}]".format(str(count), len(testImageList)))
        #for model in models:

