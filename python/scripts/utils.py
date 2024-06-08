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

def loadBSplineControlPoints(splineData, splineDimension):
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

def loadPCMatrixData(dataFilepath):
    f = open(dataFilepath)
    js = json.load(f)
    
    numComponents = int(js['numComponents'])
    data = js['data']
    
    center = np.array(data['center'])
    componentData = data['components']
    
    componentMatrix = np.zeros((numComponents, numComponents))
    
    for componentIndex in range(len(componentData)):
        componentIndexData = componentData[componentIndex]
        
        componentIndexVec = np.array(componentIndexData['componentData'])
        
        componentMatrix[componentIndex,:] = componentIndexVec
    
    return componentMatrix, center

def loadRunDataFromFile(dataFilepath, entrySize):
    f = open(dataFilepath)
    js = json.load(f)

    numDataEntries = int(js['numDataEntries'])
    data = np.array(js['data'])
    
    fullDataArr = np.reshape(data, (numDataEntries, entrySize))
    fullDataArr = fullDataArr.T
    
    return fullDataArr

def standardizeEulerAngles(anglesArr):
    anglesArr *= np.pi / 180.0
    return anglesArr
