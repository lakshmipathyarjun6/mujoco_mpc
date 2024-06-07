import argparse
import glob
import math
import numpy as np
import os

from matplotlib import pyplot as plt

from utils import loadBSplinesFromFile, loadPCMatrixData, loadBSplinesFromPCFile, loadRunDataFromFile

ROOT_DOFS = 6

def plotOverlaidPCDistributions(pcTrajectories, runtime, dofData1, dofData2, componentMatrix, center):
    fig = plt.figure(figsize=(20,4))
    fig.tight_layout()
    
    numpcs = len(pcTrajectories)
    
    timestamps1 = dofData1[0,:]
    timestamps2 = dofData2[0,:]

    dofData1 = dofData1[1:,:]
    dofData2 = dofData2[1:,:]
    
    numDofs, numEntries1 = dofData1.shape
    numEntries2 = dofData2.shape[1]

    numRows = math.ceil(np.sqrt(numpcs))
    numCols = math.floor(np.sqrt(numpcs))

    centerTiled1 = np.tile(center, (numEntries1, 1))
    centerTiled1 = centerTiled1.T
    
    centerTiled2 = np.tile(center, (numEntries2, 1))
    centerTiled2 = centerTiled2.T
    
    pcData1 = np.matmul(componentMatrix, dofData1) - centerTiled1
    pcData2 = np.matmul(componentMatrix, dofData2) - centerTiled2        
    
    for pcIndex in range(numpcs):
        pcIndexData1 = pcData1[pcIndex]
        pcIndexData2 = pcData2[pcIndex]
        
        referenceData1 = np.array([pcTrajectories[pcIndex](ts / runtime) for ts in timestamps1])[:,1]
        referenceData2 = np.array([pcTrajectories[pcIndex](ts / runtime) for ts in timestamps2])[:,1]

        # Plot diffs
        pcDataToPlot1 = pcIndexData1 - referenceData1
        pcDataToPlot2 = pcIndexData2 - referenceData2
        
        ax = fig.add_subplot(numRows, numCols, pcIndex + 1)

        ax.hist(pcDataToPlot1, alpha=0.7, color="tab:blue")
        ax.hist(pcDataToPlot2, alpha=0.7, color="tab:orange")
        
    plt.subplots_adjust(wspace=0.6)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument("-r", "--reference", required=True, type=str)
    parser.add_argument("-pc", "--pcpath", required=True, type=str)
    parser.add_argument("-p1", "--path1", required=True, type=str)
    parser.add_argument("-p2", "--path2", required=True, type=str)
    parser.add_argument("-es", "--entrysize", required=True, type=int)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    
    args = parser.parse_args()
    
    # reference = args.reference
    pcpath = args.pcpath
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
    
    # originalTrajectories, runtime = loadBSplinesFromFile(reference, slowdown)
    # originalTrajectories = originalTrajectories[ROOT_DOFS:]
    
    pcTrajectories, runtime = loadBSplinesFromPCFile(pcpath, slowdown)
    
    # Going to be convreting from original to PC space: do not transpose, subtract mean
    componentMatrix, center = loadPCMatrixData(pcpath)
    
    plotOverlaidPCDistributions(pcTrajectories, runtime, dofData1, dofData2, componentMatrix, center)
