import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import BSpline

ENTRY_SIZE = 8
SCALE_FACTOR = 1.0 / 100.0

def constructBSplines(controlPointData, splineDegree):
    numSplines, numControlPoints, _ = controlPointData.shape
    
    uniformKnots = np.linspace(0, 1, numControlPoints - splineDegree + 1)
    uniformKnots = np.concatenate((np.zeros(splineDegree), uniformKnots, np.ones(splineDegree)))
    
    bsplines = []
    
    for splineDataIndex in range(numSplines):
        bspline = BSpline(uniformKnots, controlPointData[splineDataIndex], splineDegree)
        bsplines.append(bspline)
    
    return bsplines
    
def loadSplineControlPoints(splineData, splineDimension):
    numControlPoints = int(splineData["numControlPoints"])
    controlPointData = np.array(splineData["controlPointData"])
    
    splineArr = np.reshape(controlPointData, (numControlPoints, splineDimension))
    
    return splineArr

def plotResults(bsplines, fullDataArr, totalRuntime):
    timestamps = fullDataArr[0,:]
    
    positionBSplines = bsplines[:3]
    positions = fullDataArr[1:4,:]
    
    # orientationBSplines = bsplines[3:]
    # orientations = fullDataArr[4:,:]
    
    xposReference = np.array([positionBSplines[0](ts / totalRuntime) for ts in timestamps]) * SCALE_FACTOR
    yposReference = np.array([positionBSplines[1](ts / totalRuntime) for ts in timestamps]) * SCALE_FACTOR
    zposReference = np.array([positionBSplines[2](ts / totalRuntime) for ts in timestamps]) * SCALE_FACTOR
    
    referencePositions = np.vstack((xposReference[:,1], yposReference[:,1], zposReference[:,1]))
    
    # referencePositionMags = np.linalg.norm(referencePositions, axis=0)
    # positionMags = np.linalg.norm(positions, axis=0)
    
    # diffPosMags = np.abs(referencePositionMags - positionMags)
    
    plt.plot(timestamps, referencePositions[0], label = "positionXReference")
    plt.plot(timestamps, positions[0], label = "positionXData")
    
    plt.plot(timestamps, referencePositions[1], label = "positionYReference")
    plt.plot(timestamps, positions[1], label = "positionYData")
    
    plt.plot(timestamps, referencePositions[2], label = "positionZReference")
    plt.plot(timestamps, positions[2], label = "positionZData")
    
    # plt.plot(timestamps, diffPosMags, label = "posMagDiffs")
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Postion (m)') 

    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--reference", required=True, type=str)
    parser.add_argument("-p", "--filepath", required=True, type=str)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    
    args = parser.parse_args()

    reference = args.reference
    filepath = args.filepath
    slowdown = args.slowdown
    
    r = open(reference)
    jsr = json.load(r)
    
    numSplines = int(jsr["numDofs"])
    splineDegree = int(jsr["degree"])
    splineDimension = int(jsr["dimension"])
    splineRuntime = float(jsr["time"]) * slowdown
    splineControlData = np.array([loadSplineControlPoints(entry, splineDimension) for entry in jsr["data"]])
    
    bsplines = constructBSplines(splineControlData, splineDegree)
    
    f = open(filepath)
    jsf = json.load(f)

    numDataEntries = int(jsf['numDataEntries'])
    data = np.array(jsf['data'])
    
    fullDataArr = np.reshape(data, (numDataEntries, ENTRY_SIZE))
    fullDataArr = fullDataArr.T
    
    plotResults(bsplines, fullDataArr, splineRuntime)
    