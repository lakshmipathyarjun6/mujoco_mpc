import argparse
import glob
import json
import numpy as np
import os

from matplotlib import pyplot as plt

from utils import assignColorsToDataset, loadRunDataFromFile

def plotTimeToFailureData(groupedData, contactStartTime, contactEndTime):
    datasetColors = assignColorsToDataset(groupedData)

    fig, ax = plt.subplots(figsize=(12,4))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_alpha(0.2)

    ax.set_xlabel('Contact Trajactory Completed Before Failure (%)')
    
    contactTimeDiff = contactEndTime - contactStartTime
    
    # Convert to percentage for plotting
    
    allDataToPlot = []
    plotColors = []
    names = []
    
    for datasetName in groupedData.keys():
        data = groupedData[datasetName]
        
        dataToPlot = (data - contactStartTime) / contactTimeDiff
        dataToPlot = np.clip(dataToPlot, 0.0, 1.0)
        dataToPlot *= 100
        
        color = datasetColors[datasetName]
        
        ax.scatter(dataToPlot, np.zeros(len(dataToPlot)), s=30, marker='D', c=color, label=datasetName)
        
        allDataToPlot.append(dataToPlot)
        plotColors.append(color)
        names.append(datasetName)
        
    bp = ax.boxplot(allDataToPlot, patch_artist=True, notch=True, vert = False)
        
    colorsDoubled = []
    for color in plotColors:
        colorsDoubled.append(color)
        colorsDoubled.append(color)
        
    for patch, color in zip(bp['boxes'], plotColors):
        patch.set_facecolor(color)
        
    for whisker, color in zip(bp['whiskers'], colorsDoubled):
        whisker.set(color=color, linewidth=3)
    
    for cap, color in zip(bp['caps'], colorsDoubled):
        cap.set(color=color, linewidth = 2)
    
    for median in bp['medians']:
        median.set(color='orange', linewidth = 4)
    
    # changing style of fliers
    for flier, color in zip(bp['fliers'], plotColors):
        flier.set(marker='o',markeredgecolor=color,markerfacecolor=color)

    # ax.set_xlim([0.0, 102.0])
    # ax.axvline(0.0, color='k', lw=2, linestyle='solid')
    # ax.axvline(100.0, color='k', lw=2, linestyle='solid')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    parser.add_argument("-cs", "--contactstart", required=True, type=int)
    parser.add_argument("-ce", "--contactend", required=True, type=int)
    parser.add_argument("-f", "--framerate", required=True, type=int)
    
    args = parser.parse_args()

    path = args.path
    slowdown = args.slowdown
    contactStartFrame = args.contactstart
    contactEndFrame = args.contactend
    originalFramerate = args.framerate
    
    originalContactStartTime = contactStartFrame / originalFramerate
    originalContactEndTime = contactEndFrame / originalFramerate
    
    contactStartTime = originalContactStartTime * slowdown
    contactEndTime = originalContactEndTime * slowdown
    
    dataDirs = glob.glob(os.path.join(path, '*/'))
    
    groupedData = {}
    
    contactTimeDiff = contactEndTime - contactStartTime
    
    for dataDir in dataDirs:
        runFiles = glob.glob(os.path.join(dataDir, '*.json'))
        
        dataset = dataDir.split("/")[-2]
        timeToFailures = []
        
        print(dataset)
        print()
        
        for runFile in runFiles:
            runDataArr = loadRunDataFromFile(runFile)
            timestamps = runDataArr[0,:]
            timeToFailure = timestamps[-1]
            
            ratio = (timeToFailure - contactStartTime) / contactTimeDiff
            ratio = min(ratio, 1.0)
            print(runFile, ratio)
            
            timeToFailures.append(timeToFailure)
            
        groupedData[dataset] = np.array(timeToFailures)
        
        print()
        print()
        print()
        
    plotTimeToFailureData(groupedData, contactStartTime, contactEndTime)
