import argparse
import glob
import math
import numpy as np
import os

from matplotlib import pyplot as plt

from utils import loadBSplinesFromFile, loadPCMatrixData, loadRunDataFromFile, standardizeEulerAngles

ROOT_DOFS = 6

def plotOverlaidPCDistributions(originalTrajectories, numpcs, runtime, dofData1, dofData2, componentMatrix, center):
    fig = plt.figure(figsize=(20,4))
    fig.tight_layout()
    
    numdofs = len(originalTrajectories)
    
    timestamps1 = dofData1[0,:]
    timestamps2 = dofData2[0,:]

    dofData1 = dofData1[1:,:]
    dofData2 = dofData2[1:,:]
    
    numEntries1 = dofData1.shape[1]
    numEntries2 = dofData2.shape[1]

    numRows = math.ceil(np.sqrt(numpcs))
    numCols = math.floor(np.sqrt(numpcs))

    centerTiled1 = np.tile(center, (numEntries1, 1))
    centerTiled1 = centerTiled1.T
    
    centerTiled2 = np.tile(center, (numEntries2, 1))
    centerTiled2 = centerTiled2.T
    
    pcData1 = np.matmul(componentMatrix, dofData1) - centerTiled1
    pcData2 = np.matmul(componentMatrix, dofData2) - centerTiled2
    
    refData1 = []
    refData2 = []
    
    for dofIndex in range(numdofs):
        referenceData1 = np.array([originalTrajectories[dofIndex](ts / runtime) for ts in timestamps1])[:,1]
        referenceData2 = np.array([originalTrajectories[dofIndex](ts / runtime) for ts in timestamps2])[:,1]
        
        referenceData1 = standardizeEulerAngles(referenceData1)
        referenceData2 = standardizeEulerAngles(referenceData2)
        
        refData1.append(referenceData1)
        refData2.append(referenceData2)
        
    refData1 = np.array(refData1)
    refData2 = np.array(refData2)
    
    pcRefData1 = np.matmul(componentMatrix, refData1) - centerTiled1
    pcRefData2 = np.matmul(componentMatrix, refData2) - centerTiled2
    
    for pcIndex in range(numpcs):
        pcIndexData1 = pcData1[pcIndex]
        pcIndexData2 = pcData2[pcIndex]

        # Plot diffs
        pcDataToPlot1 = pcIndexData1 - pcRefData1[pcIndex]
        pcDataToPlot2 = pcIndexData2 - pcRefData2[pcIndex]
        
        ax = fig.add_subplot(numRows, numCols, pcIndex + 1)
        
        # Use PC data since it has fewer samples
        bins = np.histogram(pcDataToPlot2)[1]

        ax.hist(pcDataToPlot1, bins=bins, alpha=0.7, color="tab:blue")
        ax.hist(pcDataToPlot2, bins=bins, alpha=0.7, color="tab:orange")
        
    plt.subplots_adjust(wspace=0.6)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--reference", required=True, type=str)
    parser.add_argument("-pc", "--pcpath", required=True, type=str)
    parser.add_argument("-p1", "--path1", required=True, type=str)
    parser.add_argument("-p2", "--path2", required=True, type=str)
    parser.add_argument("-es", "--entrysize", required=True, type=int)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    
    args = parser.parse_args()
    
    reference = args.reference
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
    
    originalTrajectories, runtime = loadBSplinesFromFile(reference, slowdown)
    originalTrajectories = originalTrajectories[ROOT_DOFS:]
    
    # Going to be convreting from original to PC space: do not transpose, subtract mean
    componentMatrix, center = loadPCMatrixData(pcpath)
    numpcs = len(center)
    
    plotOverlaidPCDistributions(originalTrajectories, numpcs, runtime, dofData1, dofData2, componentMatrix, center)
