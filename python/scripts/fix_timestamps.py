import argparse
import glob
import json
import numpy as np
import os

from utils import loadRunDataFromFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--path", required=True, type=str)
    parser.add_argument("-es", "--entrysize", required=True, type=int)
    parser.add_argument("-s", "--slowdown", required=True, type=float)
    parser.add_argument("-ot", "--originaltime", required=True, type=float)
    
    args = parser.parse_args()

    path = args.path
    entrysize = args.entrysize
    slowdown = args.slowdown
    originaltime = args.originaltime
    
    loopTime = originaltime * slowdown
    
    runFiles = glob.glob(os.path.join(path, '*.json'))
    
    for runFile in runFiles:
        runDataArr = loadRunDataFromFile(runFile, entrysize)
        timestamps = runDataArr[0,:]
        
        numDataEntries = len(timestamps)
        
        correctedTimestamps = timestamps % loopTime
        correctedData = np.vstack((correctedTimestamps, runDataArr[1:,:]))
        
        print(timestamps)
        
        dataToWrite = list(correctedData.T.flatten())
        
        writeDict = {
            "numDataEntries": numDataEntries,
            "data": dataToWrite
        }
        
        with open(runFile, 'w') as f:
            json.dump(writeDict, f, indent=4)
            
        print("Fixed: ", runFile)
