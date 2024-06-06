import json
import numpy as np

def assignColorsToDataset(groupedData):
    numKeys = len(groupedData.keys())
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    
    datasetColors = {}
    dictKeysList = list(groupedData.keys())
    
    for i in range(numKeys):
        datasetName = dictKeysList[i]
        datasetColors[datasetName] = colors[i]
    
    return datasetColors

def extractRelevantContactBounds(timestamps, contactStartTime, contactEndTime):
    startIndex = 0
    endIndex = len(timestamps) - 1
    
    while timestamps[startIndex] < contactStartTime:
        startIndex += 1
    
    while timestamps[endIndex] > contactEndTime:
        endIndex -= 1
    
    return startIndex, endIndex

def loadRunDataFromFile(dataFilepath, entrySize):
    f = open(dataFilepath)
    jsf = json.load(f)

    numDataEntries = int(jsf['numDataEntries'])
    data = np.array(jsf['data'])
    
    fullDataArr = np.reshape(data, (numDataEntries, entrySize))
    fullDataArr = fullDataArr.T
    
    return fullDataArr