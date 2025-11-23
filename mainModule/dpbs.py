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
from torch.cuda.amp import autocast, GradScaler

from torchvision.utils import save_image
from modelDefinitions.tfformer import *


class DPBS:
    def __init__(self, config):
        
        # Model Configration 
        self.gtPath = config['gtPath']
        self.targetPath = config['targetPath']
        self.checkpointPath = config['checkpointPath']
        self.logPath = config['logPath']
        self.testImagesPath = config['testImagePath']
        self.resultDir = config['resultDir']
        self.modelName = config['modelName']
        self.dataSamples = config['dataSamples']
        self.batchSize = int(config['batchSize'])
        self.imageH = 128#int(config['imageH'])
        self.imageW = 128#int(config['imageW'])
        self.inputC = int(config['inputC'])
        self.outputC = int(config['outputC'])
        self.scalingFactor = int(config['scalingFactor'])
        self.binnigFactor = int(config['binnigFactor'])
        self.totalEpoch = int(config['epoch'])
        self.interval = int(config['interval'])
        self.evalInterval = int(config['evalInterval'])
        self.learningRate = float(config['learningRate'])
        self.adamBeta1 = float(config['adamBeta1'])
        self.adamBeta2 = float(config['adamBeta2'])
        self.barLen = int(config['barLen'])
        
        # Initiating Training Parameters(for step)
        self.currentEpoch = 0
        self.startSteps = 0
        self.totalSteps = 0
        self.adversarialMean = 0
        self.PR = 0.0

        # Normalization
        self.unNorm = UnNormalize()

        # Noise Level for inferencing
        self.noiseSet = [0]
        
        
        # Preapring model(s) for GPU acceleration
        self.device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.attentionNet = TwoTineFormerN().to(self.device)
        #self.discriminator = attentiomDiscriminator().to(self.device)

        # Optimizers
        self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        #self.optimizerED = torch.optim.Adam(self.discriminator.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        self.scaler = GradScaler()
        # Scheduler for Super Convergance
        self.scheduleLR = None
        
    def customTrainLoader(self, overFitTest = False):
        
        targetImageList = imageList(self.gtPath)
        print ("Trining Samples (Input):", self.targetPath, len(targetImageList))
        #targetImageList = [k for k in targetImageList if '_nll' in k]
        #print ("Trining Samples (Filter):", self.targetPath, len(targetImageList))

        if overFitTest == True:
            targetImageList = targetImageList[-1:]
        if self.dataSamples:
            targetImageList = targetImageList[:self.dataSamples]

        datasetReadder = customDatasetReader(   
                                                image_list=targetImageList, 
                                                imagePathGT=self.gtPath,
                                                height = self.imageH,
                                                width = self.imageW,
                                            )

        self.trainLoader = torch.utils.data.DataLoader( dataset=datasetReadder,
                                                        batch_size=self.batchSize, 
                                                        shuffle=True,
                                                        num_workers=4 
                                                        )
        
        return self.trainLoader

    def modelTraining(self, resumeTraning=False, overFitTest=False, dataSamples = None):
        
        if dataSamples:
            self.dataSamples = dataSamples 

        # Losses
        # featureLoss = regularizedFeatureLoss().to(self.device)
        reconstructionLoss = torch.nn.L1Loss().to(self.device)
        ssimLoss = MSSSIM().to(self.device)
        lcLoss = LuminanceChrominanceLoss().to(self.device)
        adversarialLoss = nn.BCELoss().to(self.device)
 
        # Overfitting Testing
        if overFitTest == True:
            customPrint(Fore.RED + "Over Fitting Testing with an arbitary image!", self.barLen)
            trainingImageLoader = self.customTrainLoader(overFitTest=True)
            self.interval = 1
            self.totalEpoch = 100000
        else:  
            trainingImageLoader = self.customTrainLoader()


        # Resuming Training
        if resumeTraning == True:
            self.modelLoad()
            try:
                pass#self.modelLoad()

            except:
                #print()
                customPrint(Fore.RED + "Would you like to start training from sketch (default: Y): ", textWidth=self.barLen)
                userInput = input() or "Y"
                if not (userInput == "Y" or userInput == "y"):
                    exit()
        

        # Starting Training
        customPrint('Training is about to begin using:' + Fore.YELLOW + '[{}]'.format(self.device).upper(), textWidth=self.barLen)
        
        # Initiating steps
        self.totalSteps =  int(len(trainingImageLoader)*self.totalEpoch)
        startTime = time.time()
        mseLoss = nn.MSELoss()
        startTime = time.time()
        log10 = np.log(10)
        MAX_DIFF = 2 
        adjustStep = 1000
        
        # Instantiating Super Convergance 
        self.scheduleLR =  optim.lr_scheduler.ReduceLROnPlateau(self.optimizerEG, 'min')#open#optim.lr_scheduler.StepLR(self.optimizerEG, step_size=10000, gamma=0.1)  # Adjust parameters as needed
#optim.lr_scheduler.OneCycleLR(optimizer=self.optimizerEG, max_lr=self.learningRate, total_steps=self.totalSteps)
        # Initiating progress bar 
        bar = ProgressBar(self.totalSteps, max_width=int(self.barLen/2))
        currentStep = self.startSteps
        cumalativeLoss = [] 
        self.optimizerEG = torch.optim.Adam(self.attentionNet.parameters(), lr=self.learningRate, betas=(self.adamBeta1, self.adamBeta2))
        while currentStep < self.totalSteps:


            # Time tracker
            iterTime = time.time()
            #
            for LRImages, HRGTImages in trainingImageLoader:
    
                ##############################
                #### Initiating Variables ####
                ##############################
                if currentStep > self.totalSteps:
                    self.savingWeights(currentStep)
                    customPrint(Fore.YELLOW + "Training Completed Successfully!", textWidth=self.barLen)
                    exit()
                currentStep += 1

                # Move images to GPU
                rawInput = LRImages.to(self.device)
                highResReal = HRGTImages.to(self.device)

                ##############################
                ####### Training Phase #######
                ##############################
    
                # with autocast():
                    # Image Generation
                inter, highResFake = self.attentionNet(rawInput)


                
                # Optimization of generator 
                self.optimizerEG.zero_grad()
                
                lc_loss = lcLoss(highResFake, highResReal) + lcLoss(inter, highResReal)
                #f_loss = featureLoss(highResFake, highResReal) + featureLoss(inter, highResReal)
                r_loss = reconstructionLoss(highResFake , highResReal) + reconstructionLoss(inter , highResReal) 
                reg = 0.2
                generatorContentLoss = reg * (r_loss + lc_loss)#+ f_loss #

                lossEG = generatorContentLoss #+ 1e-3 * generatorAdversarialLoss
                if torch.isnan(lossEG):
                    print(f"❌ NaN detected in loss at step {currentStep}")
                    print("lc_loss:", lc_loss.item(), "r_loss:", r_loss.item())
                    print("highResFake stats:", highResFake.min().item(), highResFake.max().item())
                    print("highResReal stats:", highResReal.min().item(), highResReal.max().item())
                lossEG.backward()
                self.optimizerEG.step()
                cumalativeLoss.append(lossEG.item())

                cumalativeLoss.append(lossEG.item())
                if currentStep < 100000:

                    if currentStep % int(adjustStep *2.5)==0:
                        print("\n Adjusting learning rate. Previous Lr:{} | Cumalative Loss: {}".format(self.optimizerEG.param_groups[0]['lr'], round(sum(cumalativeLoss) / len(cumalativeLoss ), 3)))
                        lossCur = round(sum(cumalativeLoss) / len(cumalativeLoss), 3)
                        self.scheduleLR.step(lossCur)
                        cumalativeLoss = []
                else:
                    if currentStep % int(adjustStep )==0:
                            print("\n Adjusting learning rate. Previous Lr:{} | Cumalative Loss: {}".format(self.optimizerEG.param_groups[0]['lr'], round(sum(cumalativeLoss) / len(cumalativeLoss ), 3)))
                            lossCur = round(sum(cumalativeLoss) / len(cumalativeLoss), 3)
                            self.scheduleLR.step(lossCur)
                            cumalativeLoss = []

                    #print_lr(is_verbose, group, lr, epoch=None)

                # Steps for Super Convergance            
                #self.scheduleLR.step()
                psnr = 10*torch.log( MAX_DIFF**2 / mseLoss(highResFake, (highResReal)) ) / log10
                psnrInter = 10*torch.log( MAX_DIFF**2 / mseLoss(inter, (highResReal)) ) / log10


                ##########################
                ###### Model Logger ######
                ##########################   

                # Progress Bar
                if (currentStep  + 1) % 10== 0:
                    bar.numerator = currentStep + 1
                    print(Fore.YELLOW + "Steps |",bar,Fore.YELLOW + "| LossEG: {:.4f} | PSNR-I: {:.2f} PSNR: {:.2f}".format(lossEG, psnrInter,  psnr),end='\r')
                    
                
                # Updating training log
                if (currentStep + 1) % self.interval/10 == 0:
                   
                    # Updating Tensorboard
                    summaryInfo = { 
                                    'Input Images' : self.unNorm(rawInput),
                                    'AttentionNetGen Images' : self.unNorm(highResFake),
                                    'Interm Images' : self.unNorm(inter),
                                    'GT Images' : self.unNorm(highResReal),
                                    'Step' : currentStep + 1,
                                    'Epoch' : self.currentEpoch,
                                    'LossEG' : lossEG.item(),
                                    #'LossLC' : lc_loss.item(),
                                    'LossPSNR' : psnr,
                                    #'LossPSNR-I' : psnrInter,
                                    'Path' : self.logPath,
                                    'Atttention Net' : self.attentionNet,
                                  }
                    tbLogWritter(summaryInfo)
                    save_image(self.unNorm(highResFake[0]), 'modelOutput.png')

                    # Saving Weights and state of the model for resume training 
                    self.savingWeights(currentStep)
                
                if (currentStep + 1) % self.evalInterval == 0 : 
                    print("\n")
                    self.savingWeights(currentStep + 1, True)
                    self.modelInference(validation=True, steps = currentStep + 1)
                    eHours, eMinutes, eSeconds = timer(iterTime, time.time())
                    #print (Fore.CYAN +'Steps [{}/{}] | Time elapsed [{:0>2}:{:0>2}:{:0>2}] | LossC: {:.2f}, LossP : {:.2f}, LossEG: {:.2f}, LossED: {:.2f}' 
                            # .format(currentStep + 1, self.totalSteps, eHours, eMinutes, eSeconds, colorLoss(highResFake, highResReal), featureLoss(highResFake, highResReal),lossEG, lossED))
                    
   
    def modelInference(self, testImagesPath = None, outputDir = None, resize = None, validation = None, noiseSet = None, steps = None):
    
        if not validation:
            self.modelLoad()
            print("\nInferencing on pretrained weights.")
        else:
            print("Validation about to begin.")
        if not noiseSet:
            noiseSet = self.noiseSet
        if testImagesPath:
            self.testImagesPath = testImagesPath
        if outputDir:
            self.resultDir = outputDir
        

        modelInference = inference(gridSize=self.binnigFactor, inputRootDir=self.testImagesPath, outputRootDir=self.resultDir, modelName=self.modelName, validation=validation)

        testImageList = modelInference.testingSetProcessor()
        barVal = ProgressBar(len(testImageList) * len(noiseSet), max_width=int(50))
        imageCounter = 0
        from datetime import datetime
        with torch.no_grad():
            
                #print(noise)
            #try:
            for imgPath in testImageList:

                img = modelInference.inputForInference(imgPath).to(self.device)
                
                inter , output = self.attentionNet(img)
                torch.cuda.empty_cache()
                features = {
                "Illumination Feature": inter,
                "Illumination Map": output,
                }
                x_cpu = inter.detach().cpu().numpy()

                modelInference.saveModelOutput(inter, imgPath.replace(".", "_inter."), steps)
                modelInference.saveModelOutput(output, imgPath, steps)
                imageCounter += 1
                if imageCounter % 2 == 0:
                    barVal.numerator = imageCounter
                    print(Fore.CYAN + "Image Processd |", barVal,Fore.CYAN, end='\r')
        print("\n")

    def modelSummary(self,input_size = None):
        if not input_size:
            input_size = (3, self.imageH//self.scalingFactor, self.imageW//self.scalingFactor)

     
        customPrint(Fore.YELLOW + "AttentionNet (Generator)", textWidth=self.barLen)
        summary(self.attentionNet, input_size =input_size)
        print ("*" * self.barLen)
        print()

        customPrint(Fore.YELLOW + "AttentionNet (Discriminator)", textWidth=self.barLen)
        summary(self.discriminator, input_size =input_size)
        print ("*" * self.barLen)
        print()

        '''flops, params = get_model_complexity_info(self.attentionNet, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Gen):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Gen):{}'.format(params), self.barLen, '-')

        flops, params = get_model_complexity_info(self.discriminator, input_size, as_strings=True, print_per_layer_stat=False)
        customPrint('Computational complexity (Enhace-Dis):{}'.format(flops), self.barLen, '-')
        customPrint('Number of parameters (Enhace-Dis):{}'.format(params), self.barLen, '-')
        print()'''

        configShower()
        print ("*" * self.barLen)
    
    def savingWeights(self, currentStep, duplicate=None):
        # Saving weights 
        checkpoint = { 
                        'step' : currentStep + 1,
                        'stateDictEG': self.attentionNet.state_dict(),
                        #'stateDictED': self.discriminator.state_dict(),
                        'optimizerEG': self.optimizerEG.state_dict(),
                        #'optimizerED': self.optimizerED.state_dict(),
                        'schedulerLR': self.scheduleLR
                        }
        saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath, modelName = self.modelName)
        if duplicate:
            saveCheckpoint(modelStates = checkpoint, path = self.checkpointPath + str(currentStep) + "/", modelName = self.modelName, backup=None)



    def modelLoad(self):

        customPrint(Fore.RED + "Loading pretrained weight", textWidth=self.barLen)

        previousWeight = loadCheckpoints(self.checkpointPath, self.modelName)

        self.attentionNet.load_state_dict(previousWeight['stateDictEG'])
        #self.discriminator.load_state_dict(previousWeight['stateDictED'])
        self.optimizerEG.load_state_dict(previousWeight['optimizerEG']) 
        #self.optimizerED.load_state_dict(previousWeight['optimizerED']) 
        self.scheduleLR = previousWeight['schedulerLR']
        self.startSteps = int(previousWeight['step'])
        
        customPrint(Fore.YELLOW + "Weight loaded successfully", textWidth=self.barLen)


        
