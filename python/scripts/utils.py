import json
import numpy as np

from scipy.interpolate import BSpline

def assignColorsToDataset(groupedData):
    numKeys = len(groupedData.keys())
    
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    
    datasetColors = {}
    dictKeysList = list(groupedData.keys())
    
    for i in range(numKeys):
        datasetName = dictKeysList[i]
        datasetColors[datasetName] = colors[i]
    
    return datasetColors

def constructBSplines(controlPointData, splineDegree):
    numSplines, numControlPoints, _ = controlPointData.shape
    
    uniformKnots = np.linspace(0, 1, numControlPoints - splineDegree + 1)
    uniformKnots = np.concatenate((np.zeros(splineDegree), uniformKnots, np.ones(splineDegree)))
    
    bsplines = []
    
    for splineDataIndex in range(numSplines):
        bspline = BSpline(uniformKnots, controlPointData[splineDataIndex], splineDegree)
        bsplines.append(bspline)
    
    return bsplines

def extractRelevantContactBounds(timestamps, contactStartTime, contactEndTime):
    startIndex = 0
    endIndex = len(timestamps) - 1
    
    while timestamps[startIndex] < contactStartTime:
        startIndex += 1
    
    while timestamps[endIndex] > contactEndTime:
        endIndex -= 1
    
    return startIndex, endIndex

def loadBSplineControlPoints(splineData, splineDimension):
    type = splineData["type"]
    numControlPoints = int(splineData["numControlPoints"])
    controlPointData = np.array(splineData["controlPointData"])
    
    splineArr = np.reshape(controlPointData, (numControlPoints, splineDimension))
    
    return splineArr

def loadBSplinesFromFile(dataFilepath, slowdown):
    f = open(dataFilepath)
    js = json.load(f)
    
    splineDegree = int(js["degree"])
    splineDimension = int(js["dimension"])
    splineRuntime = float(js["time"]) * slowdown
    splineControlData = np.array([loadBSplineControlPoints(entry, splineDimension) for entry in js["data"]])
    
    bsplines = constructBSplines(splineControlData, splineDegree)
    
    return bsplines, splineRuntime
    
def loadRunDataFromFile(dataFilepath, entrySize):
    f = open(dataFilepath)
    js = json.load(f)

    numDataEntries = int(js['numDataEntries'])
    data = np.array(js['data'])
    
    fullDataArr = np.reshape(data, (numDataEntries, entrySize))
    fullDataArr = fullDataArr.T
    
    return fullDataArr