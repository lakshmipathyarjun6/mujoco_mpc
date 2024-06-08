import argparse
import glob
import math
import numpy as np
import os

from matplotlib import pyplot as plt

from utils import loadBSplinesFromFile, loadRunDataFromFile, standardizeEulerAngles

ROOT_DOFS = 6

def plotOverlaidDofDistributions(originalTrajectories, runtime, dofData1, dofData2):
    fig = plt.figure(figsize=(20,4))
    fig.tight_layout()
    
    timestamps1 = dofData1[0,:]
    timestamps2 = dofData2[0,:]

    dofData1 = dofData1[1:,:]
    dofData2 = dofData2[1:,:]
    
    numDofs = dofData1.shape[0]

    numRows = math.ceil(np.sqrt(numDofs))
    numCols = math.floor(np.sqrt(numDofs))
    
    for dofIndex in range(numDofs):
        dofIndexData1 = dofData1[dofIndex]
        dofIndexData2 = dofData2[dofIndex]
        
        referenceData1 = np.array([originalTrajectories[dofIndex](ts / runtime) for ts in timestamps1])[:,1]
        referenceData2 = np.array([originalTrajectories[dofIndex](ts / runtime) for ts in timestamps2])[:,1]
        
        referenceData1 = standardizeEulerAngles(referenceData1)
        referenceData2 = standardizeEulerAngles(referenceData2)
        
        # Plot diffs
        dataToPlot1 = dofIndexData1 - referenceData1
        dataToPlot2 = dofIndexData2 - referenceData2
        
        ax = fig.add_subplot(numRows, numCols, dofIndex + 1)

        # Use PC data since it has fewer samples
        bins = np.histogram(dataToPlot2)[1]

        ax.hist(dataToPlot1, bins=bins, alpha=0.7, color="tab:blue")
        ax.hist(dataToPlot2, bins=bins, alpha=0.7, color="tab:orange")
        
    plt.subplots_adjust(wspace=0.6)
    plt.show()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--reference", required=True, type=str)
    parser.add_argument("-p1", "--path1", required=True, type=str)
    parser.add_argument("-p2", "--path2", required=True, type=str)
    parser.add_argument("-es", "--entrysize", required=True, type=int)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    
    args = parser.parse_args()
    
    reference = args.reference
    path1 = args.path1
    path2 = args.path2
    entrysize = args.entrysize
    slowdown = args.slowdown

    path1RunFiles = glob.glob(os.path.join(path1, '*.json'))
    path2RunFiles = glob.glob(os.path.join(path2, '*.json'))
    
    dofData1 = np.zeros((entrysize,1))
    dofData2 = np.zeros((entrysize,1))
    
    for path1RunFile in path1RunFiles:
        runDataArr = loadRunDataFromFile(path1RunFile, entrysize)
        dofData1 = np.hstack((dofData1, runDataArr))

    for path2RunFile in path2RunFiles:
        runDataArr = loadRunDataFromFile(path2RunFile, entrysize)
        dofData2 = np.hstack((dofData2, runDataArr))        
    
    originalTrajectories, runtime = loadBSplinesFromFile(reference, slowdown)
    originalTrajectories = originalTrajectories[ROOT_DOFS:]
        
    plotOverlaidDofDistributions(originalTrajectories, runtime, dofData1, dofData2)
