import argparse
import math

from matplotlib import pyplot as plt

from utils import loadRunDataFromFile

ENTRY_SIZE = 17

def plotOverlaidDofDistributions(dofData1, dofData2):
    fig = plt.figure(figsize=(20,4))
    fig.tight_layout()
    
    numDofs = dofData1.shape[0]
    
    numCols = math.ceil(numDofs / 2)
    numRows = math.floor(numDofs / numCols)
    
    for dofIndex in range(numDofs):
        dofIndexData1 = dofData1[dofIndex]
        dofIndexData2 = dofData2[dofIndex]
        
        ax = fig.add_subplot(numRows, numCols, dofIndex + 1)
        
        ax.hist(dofIndexData1, alpha=0.7)
        ax.hist(dofIndexData2, alpha=0.7)
    
    plt.subplots_adjust(wspace=0.6)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p1", "--path1", required=True, type=str)
    parser.add_argument("-p2", "--path2", required=True, type=str)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    parser.add_argument("-cs", "--contactstart", required=True, type=int)
    parser.add_argument("-ce", "--contactend", required=True, type=int)
    parser.add_argument("-f", "--framerate", required=True, type=int)
    
    args = parser.parse_args()

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
    
    runDataArr1 = loadRunDataFromFile(path1, ENTRY_SIZE)
    runDataArr2 = loadRunDataFromFile(path2, ENTRY_SIZE)
    
    timestamps1 = runDataArr1[0,:]
    timestamps2 = runDataArr2[0,:]

    dofData1 = runDataArr1[1:,:]
    dofData2 = runDataArr2[1:,:]
    
    plotOverlaidDofDistributions(dofData1, dofData2)
    