import json
import numpy as np

ENTRY_SIZE = 8

def assignColorsToDataset(groupedData):
    numKeys = len(groupedData.keys())
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    
    datasetColors = {}
    dictKeysList = list(groupedData.keys())
    
    for i in range(numKeys):
        datasetName = dictKeysList[i]
        datasetColors[datasetName] = colors[i]
    
    return datasetColors

def loadRunDataFromFile(dataFilepath):
    f = open(dataFilepath)
    jsf = json.load(f)

    numDataEntries = int(jsf['numDataEntries'])
    data = np.array(jsf['data'])
    
    fullDataArr = np.reshape(data, (numDataEntries, ENTRY_SIZE))
    fullDataArr = fullDataArr.T
    
    return fullDataArr