import argparse
import glob
import math
import numpy as np
import os

from matplotlib import pyplot as plt

from utils import loadBSplinesFromFile, loadRunDataFromFile

ENTRY_SIZE = 17
ROOT_DOFS = 6

def plotOverlaidDofDistributions(originalTrajectories, runtime, dofData1, dofData2):
    fig = plt.figure(figsize=(20,4))
    fig.tight_layout()
    
    timestamps1 = dofData1[0,:]
    timestamps2 = dofData2[0,:]
    
    dofData1 = dofData1[1:,:]
    dofData2 = dofData2[1:,:]
    
    numDofs = dofData1.shape[0]
    
    numCols = math.ceil(numDofs / 2)
    numRows = math.floor(numDofs / numCols)
    
    for dofIndex in range(1):
        dofIndexData1 = dofData1[dofIndex]
        dofIndexData2 = dofData2[dofIndex]
        
        referenceData1 = np.array([originalTrajectories[dofIndex](ts / runtime) for ts in timestamps1])[:,1]
        referenceData2 = np.array([originalTrajectories[dofIndex](ts / runtime) for ts in timestamps2])[:,1]
        
        referenceData1 = standardizeEulerAngles(referenceData1)
        referenceData2 = standardizeEulerAngles(referenceData2)
        
        print(referenceData1.shape, referenceData2.shape)
        
        # Plot diffs from reference by default
        dataToPlot1 = dofIndexData1 - referenceData1
        dataToPlot2 = dofIndexData2 - referenceData2
        
        ax = fig.add_subplot(numRows, numCols, dofIndex + 1)
        
        ax.hist(dataToPlot1, alpha=0.7, color="tab:blue")
        ax.hist(dataToPlot2, alpha=0.7, color="tab:orange")
    
    plt.subplots_adjust(wspace=0.6)
    plt.show()

def standardizeEulerAngles(anglesArr):
    # All reference angles are in degrees and not necessarily clamped
    
    print(anglesArr)
    
    # for i in range(len(anglesArr)):
    #     dofValue = anglesArr[i]
        
    #     while (dofValue > 360.0):
    #         dofValue -= 360.0
    #     while (dofValue < -360.0):
    #         dofValue += 360.0
        
    #     dofValue *= np.pi / 180.0
    #     anglesArr[i] = dofValue
    
    return anglesArr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--reference", required=True, type=str)
    parser.add_argument("-p1", "--path1", required=True, type=str)
    parser.add_argument("-p2", "--path2", required=True, type=str)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    parser.add_argument("-cs", "--contactstart", required=True, type=int)
    parser.add_argument("-ce", "--contactend", required=True, type=int)
    parser.add_argument("-f", "--framerate", required=True, type=int)
    
    args = parser.parse_args()

    reference = args.reference
    path1 = args.path1
    path2 = args.path2
    slowdown = args.slowdown
    contactStartFrame = args.contactstart
    contactEndFrame = args.contactend
    originalFramerate = args.framerate
    
    originalContactStartTime = contactStartFrame / originalFramerate
    originalContactEndTime = contactEndFrame / originalFramerate
    
    contactStartTime = originalContactStartTime * slowdown
    contactEndTime = originalContactEndTime * slowdown
    
    # Use for folders
    path1RunFiles = glob.glob(os.path.join(path1, '*.json'))
    path2RunFiles = glob.glob(os.path.join(path2, '*.json'))
    
    dofData1 = np.zeros((ENTRY_SIZE,1))
    dofData2 = np.zeros((ENTRY_SIZE,1))
    
    for path1RunFile in path1RunFiles:
        runDataArr = loadRunDataFromFile(path1RunFile, ENTRY_SIZE)
        print(path1RunFile, runDataArr[0])
        dofData1 = np.hstack((dofData1, runDataArr))
        
    for path2RunFile in path2RunFiles:
        runDataArr = loadRunDataFromFile(path2RunFile, ENTRY_SIZE)
        dofData2 = np.hstack((dofData2, runDataArr))
    
    # dofData1 = dofData1[:,1:]
    # dofData2 = dofData2[:,1:]
    
    # originalTrajectories, runtime = loadBSplinesFromFile(reference, slowdown)
    
    # originalTrajectories = originalTrajectories[ROOT_DOFS:]
    
    # for splineDataIndex in range(len(originalTrajectories)):
    #     spline = originalTrajectories[splineDataIndex]
        
    # print(dofData1[0,:])
        
    # for dofIndex in range(1):
    
    # plotOverlaidDofDistributions(originalTrajectories, runtime, dofData1, dofData2)
    