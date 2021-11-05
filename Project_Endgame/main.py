# Importing the libraries
from pygame.math import Vector2
from matplotlib import pyplot as plt
from scipy.ndimage import rotate 
from models import ReplayBuffer,TD3 
from PIL import Image 
import numpy as np 
import pygame 
import os 
import time 
import torch 
import math 
import numpy as np 
import argparse
import cv2
from skimage.transform import resize

def init():
    '''
    Initializing variables and mask
    @return: None
    '''
    global nonRoadCount
    global destX
    global destY
    global firstUpdate
    global mapForCropping
    width = 1429
    height = 660
    mapForCropping = np.zeros((width,height))
    image = Image.open("mask.png").convert('L')   
    mapForCropping = np.zeros((width,height))
    mapForCropping = np.asarray(image)/255 
    destX , destY = coordinates[np.random.randint(0,len(coordinates))]
    firstUpdate = False 

class Car:
    '''
    @init : Initalising position,velocity,angle and image patch
    @func cropBy : Crop a image patch
    @func move : Calculate the position of car and rotation
    '''
    def __init__(self,angle = 0):
        '''
        28x28 cropping is done
        @param self: Initializing variables and mask
        @param angle: rotation angle of car
        @return: None
        '''
        global nonRoadCount
        self.position = Vector2(originX,originY)
        self.velocity = [0,0]
        self.angle = angle         
        self.cropping = np.zeros([1,int(28),int(28)])   

    def cropBy(self,prevX,prevY):
        '''
        7x7 patch used to reporesent car , shrinked to 28x28
        @param self:
        @param prevX: position of car in x axis
        @param prevY: position of car in y axis
        @return: CroppedImage
        '''
        global mapForCropping
        croppedImage = np.copy(mapForCropping)
        croppedImage = np.pad(croppedImage,28,constant_values=1.0)
        croppedImage = croppedImage[prevX:prevX+56,prevY:prevY+56]        
        croppedImage = rotate(croppedImage, angle=-self.angle,reshape=False,order=1,mode='constant',cval=1.0)
        croppedImage[21:28,25:32] = 1 
        croppedImage[28:35,25:32] = 0        
        croppedImage = resize(croppedImage, (28, 28))
        croppedImage = np.expand_dims(croppedImage,0)        
        return croppedImage

    def move(self,rotation):
        '''
        calculting the position with velocuty - cropping patch
        @param rotation: Angle of rotation
        @return: rotated angle 
        '''
        global episode
        prevX,prevY = self.position.x , self.position.y         
        self.position = Vector2(*self.velocity) + self.position
        self.rotateBy = rotation
        self.angle = self.angle + self.rotateBy         
        self.cropping = self.cropBy(int(prevX),int(prevY))
        tCropImage = self.cropping        
        return self.angle 

    
class Environment:
    '''
    @init initalize neccesary parameters for environment
    @func serve_car: gives center position of car
    @func runCarUi: provides UI for the car
    @func distanceCalc: Gives distance calcultion between two points
    @func reset: Resetting car and other parameters
    @func step: Calcualting each step with the action recived from TD3
    @func evaluate_policy: Evaluating the brain
    @func update: Updating 
    '''
    global nonRoadCount
    def __init__(self):
        '''
        calculting the position with velocuty - cropping patch
        @param self: importing images of mask,car and setting environment       
        @return: None
        '''        
        pygame.init()
        pygame.display.set_caption("Car tutorial")
        width = 1429 
        height = 660 
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False
        self.center = [234,245]
        self.car = Car()
        self.bg_img = pygame.image.load('MASK1.png')
        self.car_img = pygame.image.load('car.png')
        self.city_img = pygame.image.load('citymap.png')
        self.car_img = pygame.transform.scale(self.car_img,(15,10))
        global no_of_timesteps

    def runCarUi(self,carposx,carposy,desposx,desposy,tAngle):
        '''
        Displaying cars position and destination
        @param carposx: position of car - x axis
        @param carposy: position of car - y axis
        @param desposx: position of Destination - x axis
        @param desposx: position of Destination - y axis
        @param tAngle: Angle of rotation
        @return: None
        '''
        display_cam = np.copy(self.car.cropping.squeeze())
        display_cam2 = np.zeros([int(28),int(28),3])
        display_cam2[:,:,0] = display_cam*255
        display_cam2[:,:,1] = display_cam*255
        display_cam2[:,:,2] = display_cam*255
        display_cam3 = pygame.surfarray.make_surface(display_cam2)        
        rot_img = pygame.transform.rotate(self.car_img,tAngle)        
        self.screen.blit(self.city_img, self.city_img.get_rect())
        self.screen.blit(rot_img,[self.car.position.x,650-self.car.position.y])
        font = pygame.font.Font('freesansbold.ttf',20)
        green = (0,255,0)
        blue = (255,255,255)
        self.screen.blit(display_cam3,(1158,608)) 
        text = str(desposx) + "," + str(660-desposy)
        self.screen.blit(font.render("GOAL - " ,True,(0,0,255),blue),(1079,552))
        self.screen.blit(font.render(text ,True,green,blue),(1158,552))
        pygame.draw.rect(self.screen, (0, 255, 0), (desposx,660-desposy, 25,25))        
        pygame.display.flip()

        
    def distanceCalc(self,carx,cary,destX,destY):
        '''
        calculting distance from destination
        @param carx: position of car - x axis
        @param cary: position of car - y axis
        @param destX: position of Destination - x axis
        @param destY: position of Destination - y axis        
        @return: None 
        '''
        return np.sqrt((carx-destX)**2 + (cary-destY)**2)

    def reset(self):
        '''
        Reseting the environment and agent and initalising the state
        @param self: imagepatch, orientation, change in distance
        @return: state
        '''
        global prevDistance 
        global originX 
        global originY 
        self.car.position.x =  originX 
        self.car.position.y = originY 
        xx = destX - self.car.position.x 
        yy = destY - self.car.position.y         
        tempAngle = -(180 / math.pi) * math.atan2(
            self.car.velocity[0] * yy- self.car.velocity[1] * xx,
            self.car.velocity[0]* xx + self.car.velocity[1] * yy)
        orientation = tempAngle/180
        self.currDistance = self.distanceCalc(self.car.position.x,self.car.position.y,destX,destY)
        delta = prevDistance-self.currDistance       
        state = [self.car.cropping , orientation ,-orientation,delta ]
        return state 

    def step(self,action):
        '''
        calculting the state and giving rewards for the step taken
        @param action: Angle of rotation
        @return: state,previousReward,done  
        '''
        global destX 
        global destY 
        global originX
        global originY 
        global done 
        global prevDistance 
        global nonRoadCount 
        
        # Action got from TD3
        rotation = action.item()
        turnAngle = self.car.move(rotation) 
        self.runCarUi(self.car.position.x,self.car.position.y,destX,destY,turnAngle)
        xx = destX - self.car.position.x 
        yy = destY - self.car.position.y 
        tempAngle = -(180 / math.pi) * math.atan2(
            self.car.velocity[0] * yy- self.car.velocity[1] * xx,
            self.car.velocity[0]* xx + self.car.velocity[1] * yy)
        orientation = tempAngle/180

        self.currDistance = self.distanceCalc(self.car.position.x,self.car.position.y,destX,destY)
        
        delta = prevDistance-self.currDistance
        state = [self.car.cropping , orientation ,-orientation, delta ]

        # Reward alloting 
        if mapForCropping[int(self.car.position.x),int(self.car.position.y)] > 0:
            self.car.velocity = Vector2(0.5,0).rotate(self.car.angle)
            previousReward = -5
            
        else:
            self.car.velocity = Vector2(1.8,0).rotate(self.car.angle)
            previousReward = -2 
            if self.currDistance < prevDistance:
                previousReward = 2

        #boundary conditions
        if self.car.position.x < 5:
            self.car.position.x = 5 
            previousReward = -15
            nonRoadCount += 1
        if self.car.position.x > self.width - 5 :
            self.car.position.x = self.width - 5
            previousReward = -15 
            nonRoadCount += 1 
        if self.car.position.y < 5:
            self.car.position.y = 5 
            previousReward = -15
            nonRoadCount+=1 
        if self.car.position.y > self.height - 5:
            self.car.position.y = self.height-5 
            previousReward =-15
            nonRoadCount += 1

        # Destination check
        if self.currDistance < 40 :
            originX = destX
            originY = destY
            destX,destY = coordinates[np.random.randint(0,len(coordinates))]
            previousReward = 100 
            done = True 
        prevDistance = self.currDistance 
        return state,previousReward,done 

    
    def evaluate_policy(self, brain, eval_episodes=10):
        '''
        calculting average reward for few episodes
        @param brain: TD3 action selection
        @param eval_episodes: 
        @return: avg_reward 
        '''
        avg_reward = 0.
        for _ in range(eval_episodes):
            observation = self.reset() # ToDo reset env
            done = False
            while not done:
                action = brain.select_action(np.array(observation))
                observation,reward,done = self.step(action)
                avg_reward += reward
        avg_reward /= eval_episodes
        print("---------------------------------------")
        print("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print("---------------------------------------")
        return avg_reward

    def update(self):
        '''
        Episode construction and training is made        
        @return: None
        '''
        global firstUpdate
        global destX
        global destY
        global episodeNo
        global totalTimeSteps
        global newobservation
        global evaluations
        global timeStepsFromEval  
        global last_reward
        global reward
        global brain
        global maximumTimeSteps
        global maxEpisodeSteps
        global episodeTimeSteps
        global done
        global episodeReward
        global replayBuffer
        global observation
        self.width = 1429
        self.height = 660 

        if firstUpdate:
            init()                
            evaluations = [self.evaluate_policy(brain)]
            distance_travelled=0
            done = True
            observation = self.reset()
        
        if episodeReward<-2500:
            done=True        
        
        if totalTimeSteps < maximumTimeSteps:
            if done:
                print("Done - ","Episodeno",episodeNo,"Rewards",episodeReward)                
                episodeList.append(episodeNo)
                rewardsList.append(episodeReward)
                if totalTimeSteps!= 0:                    
                    brain.train(replayBuffer, episodeTimeSteps, batchSize, gamma, 
                    updateRate, policyNoise, noiseClip,
                                 policyFreq)                
                if timeStepsFromEval >= evalTime:
                    print("Saving Weights")
                    timeStepsFromEval %= evalTime
                    evaluations.append(self.evaluate_policy(brain))
                    brain.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)                
                observation = self.reset()
                done = False               
                episodeReward = 0
                episodeTimeSteps = 0
                episodeNo += 1
            if totalTimeSteps < startTimeSteps:
                action = np.random.uniform(low=-5, high=5, size=(1,))
            else:  
                action = brain.select_action(np.array(observation))
                if exploration != 0:
                    action = (action + np.random.normal(0, exploration, size=1)).clip(
                        -5, 5)
                      
            newobservation,reward, done = self.step(action)           
            doneCheck = 0 if episodeTimeSteps + 1 == maxEpisodeSteps else float(
                done)           
            episodeReward += reward                        
            replayBuffer.add((observation, newobservation, action, reward, doneCheck))
                        
            observation = newobservation
            episodeTimeSteps += 1
            totalTimeSteps += 1
            timeStepsFromEval += 1
            
            # Saving weights for every 50k time steps 
            if totalTimeSteps%50000==5:        
                plt.plot(episodeList,rewardsList)
                plt.xlabel("Episodes")
                plt.ylabel("Rewards")
                plt.savefig('graph.png')
                print("Graph Plotted","Total Time Steps",totalTimeSteps)        
                brain.save("%s" % (file_name), directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)
                print("Saved Model %s" % (file_name))
        else:
            action = brain.select_action(np.array(observation))            
            newobservation,reward, done = self.step(action)
            observation = newobservation
            totalTimeSteps += 1
            if totalTimeSteps%1000==1:
                print(totalTimeSteps)


if __name__ == '__main__':
    instruction = """ Kindly refer the readme for instructions to run main.py file \n
               For inferencing/testing car - please run below command \n    
                   "python main.py --operation test_car" \n
               For training car - please run below command \n
                   "python main.py --operation train_car"  \n
            """
    
    parser = argparse.ArgumentParser(description=instruction)
    parser.add_argument('--operation')
    parser.add_argument("-v", "--version", help="show program version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("This is car version 1.0")
    if args.operation == "train_car":
        testCar = False
    elif args.operation == "test_car":
        testCar = True 
    else :
        print(instruction)   
    
    file_name = 'TD3_CarApp_0'

    # origin points
    originX = 725 
    originY = 274

    global cropImage 
    global tCropImage    
    coordinates = [[581,348],[987,385],[985,592],[157,448],[351,397],[633,534]]
    episodeList = []
    rewardsList = [] 
    previousReward = 0
    prevDistance= 0
    nonRoadCount = 0
    firstUpdate = True 
    seed = 0
    np.random.seed(seed)    
    torch.manual_seed(seed)
    startTimeSteps = 6e3
    evalTime = 1e3 
    maximumTimeSteps = 2e6
    exploration = 0.1 
    batchSize = 100 
    gamma = 0.99 
    updateRate = 0.005 
    policyNoise = 0.2 
    noiseClip = 0.5 
    policyFreq = 2 
    totalTimeSteps = 0 
    timeStepsFromEval = 0 
    episodeNo = 0
    episodeReward = 0 
    maxEpisodeSteps = 1000 
    done = True 
    

    stateDim = 4 
    actionDim = 1 
    maxAction = 5 
    replayBuffer = ReplayBuffer() 
    brain = TD3(stateDim,actionDim,maxAction)

    observation = np.array([])
    newobservation = np.array([])
    evaluations = []

    # loading model
    if testCar == True :
        print("### Model loaded ###")
        totalTimeSteps = maximumTimeSteps 
        brain.load("%s" % (file_name), directory="./pytorch_models")

    parent = Environment()    
    startTicks = pygame.time.get_ticks()

    while True :
        parent.update()
        time.sleep(1/60)
    

    



