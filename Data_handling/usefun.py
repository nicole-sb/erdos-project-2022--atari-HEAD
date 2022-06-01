## For data handling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from seaborn import set_style
from csv import reader
import logging as logger
import os
import shutil
import tarfile
from csv import reader
import cv2
import seaborn as sns
set_style("whitegrid")

def findangamp(x,y):#calculate the angle and amplitude of a vector
    amp=np.sqrt(x**2+y**2)#this is amplitude of a vector
    if x<0 and y>0:
        theta=np.arctan(y/x)*180/np.pi+180
        #theta is the angle of a vector
    elif x<0 and y<0:
        theta=np.arctan(y/x)*180/np.pi-180
    else:
        theta=np.arctan(y/x)*180/np.pi
    return [theta,amp]

def findang(x,y):#calculate the angle of a unit vector
    if x<0 and y>0:
        theta=np.arctan(y/x)*180/np.pi+180
        #theta is the angle of a vector
    elif x<0 and y<0:
        theta=np.arctan(y/x)*180/np.pi-180
    else:
        theta=np.arctan(y/x)*180/np.pi
    return theta



def mostvalue(arr):#this function is used to find the most frequent number in 2D array matrix
    arr1=arr.flatten()
    if np.array_equal(arr1, np.array([])):
        return np.nan
    else:
        return np.bincount(arr1).argmax()


def possibleaction(trial,minrow,maxrow,mincol,maxcol): #extract the possible action by checking four blocks around pac_man in each frame
    action=[]
    right=trial[minrow:maxrow,maxcol+1:maxcol+9]
    left=trial[minrow:maxrow,mincol-8:mincol]
    up=trial[minrow-8:minrow,mincol:maxcol]
    down=trial[maxrow+1:maxrow+9,mincol:maxcol]
    if mostvalue(up)<60:
        action.append(0)         # 0 means up action
    if mostvalue(right)<60:
        action.append(1)        # 1 means right action
    if mostvalue(left)<60:
        action.append(2)       # 2 means left action
    if mostvalue(down)<60:
        action.append(3)        # 3 means down action
    return action

def findpacman(image,colornum): # use image array and color space of pac_man to find the position of pac_man and next possible action
    result = np.where(image == colornum) #find the pac_man, 167 is the color space of pac_man
    #print(result)
    pac=np.nan
    poss=np.nan
    if np.size(result[0],axis=None)>=5:#make sure the number of pac_man points is large enough to satisfy the pca requirement
        lenresult=len(result[0])
        X=np.ones((lenresult,2)) # X is np array to store the points of pac_man
        X[:,0]=result[1]
        X[:,1]=result[0]
        #print(X)
        from sklearn.decomposition import PCA
        pca = PCA(2)

        ## Fit the data
        pca.fit(X)
        pac=pca.mean_# using PCA, take mean point as the position of pac_man
        minrow=result[0].min()
        maxrow=result[0].max()
        mincol=result[1].min()
        maxcol=result[1].max()  # the above four parameters represent the range of pac_man
        poss=possibleaction(image,minrow,maxrow,mincol,maxcol)
    else:
        pac=np.nan
        poss=np.nan
    return [pac,poss]

def pickaction(lis,num):
    #this is used to see if num is in the list,
    #if it is, return 1, otherwise return 0
    if num in lis:
        return 1
    else:
        return 0


def accuracy(true, predicted):
    #this is used to calculate the accuracy between
    #predicted column and true column
    return np.sum(true==predicted)/len(predicted)
