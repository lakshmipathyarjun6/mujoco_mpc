import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import quaternion

from scipy.interpolate import BSpline

from utils import loadRunDataFromFile

ENTRY_SIZE = 8
SCALE_FACTOR = 1.0 / 100.0

def convertEulerAnglesToQuat(eulerAngles):
    roll = eulerAngles[0]
    pitch = eulerAngles[1]
    yaw = eulerAngles[2]

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    
    quatVec = np.zeros(4)

    quatVec[0] = cr * cp * cy + sr * sp * sy
    quatVec[1] = sr * cp * cy - cr * sp * sy
    quatVec[2] = cr * sp * cy + sr * cp * sy
    quatVec[3] = cr * cp * sy - sr * sp * cy
    
    quat = quaternion.as_quat_array(quatVec)
    
    return quat

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
    
    xposReference = np.array([positionBSplines[0](ts / totalRuntime) for ts in timestamps]) * SCALE_FACTOR
    yposReference = np.array([positionBSplines[1](ts / totalRuntime) for ts in timestamps]) * SCALE_FACTOR
    zposReference = np.array([positionBSplines[2](ts / totalRuntime) for ts in timestamps]) * SCALE_FACTOR
    
    referencePositions = np.vstack((xposReference[:,1], yposReference[:,1], zposReference[:,1]))
    
    referencePositionMags = np.linalg.norm(referencePositions, axis=0)
    positionMags = np.linalg.norm(positions, axis=0)
    
    diffPosMags = np.abs(referencePositionMags - positionMags)
    
    plt.plot(timestamps, referencePositions[0], label = "positionXReference")
    plt.plot(timestamps, positions[0], label = "positionXData")
    
    plt.plot(timestamps, referencePositions[1], label = "positionYReference")
    plt.plot(timestamps, positions[1], label = "positionYData")
    
    plt.plot(timestamps, referencePositions[2], label = "positionZReference")
    plt.plot(timestamps, positions[2], label = "positionZData")
    
    plt.plot(timestamps, diffPosMags, label = "diff")
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Postion (m)') 

    plt.legend()
    plt.show()
    
    orientationBSplines = bsplines[3:]
    orientations = fullDataArr[4:,:]
    qsOrientations = quaternion.as_quat_array(orientations.T)
    
    xeulerReference = np.array([orientationBSplines[0](ts / totalRuntime) for ts in timestamps])
    yeulerReference = np.array([orientationBSplines[1](ts / totalRuntime) for ts in timestamps])
    zeulerReference = np.array([orientationBSplines[2](ts / totalRuntime) for ts in timestamps])
    
    referenceOrientations = standardizeEulerAngles(np.vstack((xeulerReference[:,1], yeulerReference[:,1], zeulerReference[:,1])))
    
    qsReferenceOrientations = []
    
    for colIndex in range(referenceOrientations.shape[1]):
        eulerAngleEntry = referenceOrientations[:,colIndex]
        quat = convertEulerAnglesToQuat(eulerAngleEntry)
        qsReferenceOrientations.append(quat)
    
    qsReferenceOrientations = np.array(qsReferenceOrientations)
    
    qsReferenceTimeSeries = quaternion.as_float_array(qsReferenceOrientations).T
    qsTimeSeries = quaternion.as_float_array(qsOrientations).T
    
    quatDiffs = []
    
    for qInd in range(len(qsReferenceOrientations)):
        qRef = qsReferenceOrientations[qInd]
        qDat = qsOrientations[qInd]
        
        diffVel = subtractQuaternionsAsVelocities(qRef, qDat)
        quatDiffs.append(diffVel)
        
    quatDiffs = np.array(quatDiffs).T
    
    quatDiffMags = np.linalg.norm(quatDiffs, axis=0)
    
    plt.plot(timestamps, qsReferenceTimeSeries[0], label = "orientationWReference")
    plt.plot(timestamps, qsTimeSeries[0], label = "orientationWData")
    
    plt.plot(timestamps, qsReferenceTimeSeries[1], label = "orientationXReference")
    plt.plot(timestamps, qsTimeSeries[1], label = "orientationXData")
    
    plt.plot(timestamps, qsReferenceTimeSeries[2], label = "orientationYReference")
    plt.plot(timestamps, qsTimeSeries[2], label = "orientationYData")
    
    plt.plot(timestamps, qsReferenceTimeSeries[3], label = "orientationZReference")
    plt.plot(timestamps, qsTimeSeries[3], label = "orientationZData")
    
    plt.plot(timestamps, quatDiffMags, label = "diff")
    
    plt.xlabel('Time (s)') 
    plt.ylabel('Orientation') 

    plt.legend()
    plt.show()

def standardizeEulerAngles(anglesArr):
    # All reference angles are in degrees and not necessarily clamped

    for dofIndex in range(len(anglesArr)):
        angleArr = anglesArr[dofIndex]
    
        for i in range(len(angleArr)):
            dofValue = angleArr[i]
            
            while (dofValue > 360.0):
                dofValue -= 360.0
            while (dofValue < -360.0):
                dofValue += 360.0
            
            dofValue *= np.pi / 180.0
            angleArr[i] = dofValue
            
        anglesArr[dofIndex] = angleArr
    
    return anglesArr

def subtractQuaternionsAsVelocities(q1, q2):
    q2Conj = q2.conjugate()
    qdif = q2Conj * q1
    
    qarr = quaternion.as_float_array(qdif)
    
    axis = qarr[1:]
    sin_a_2 = np.linalg.norm(axis)

    speed = 2 * np.arctan2(sin_a_2, qarr[0])

    if speed > np.pi:
        speed -= 2 * np.pi
        
    axis *= speed

    return axis

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
    
    fullDataArr = loadRunDataFromFile(filepath, ENTRY_SIZE)
    
    plotResults(bsplines, fullDataArr, splineRuntime)
    