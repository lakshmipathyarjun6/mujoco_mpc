import argparse
import glob
import os
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-p1", "--path1", required=True, type=str)
    parser.add_argument("-p2", "--path2", required=True, type=str)
    parser.add_argument("-o", "--outputpath", required=True, type=str)
    
    args = parser.parse_args()

    path1 = args.path1
    path2 = args.path2
    outputpath = args.outputpath
    
    path1RunFiles = glob.glob(os.path.join(path1, '*.json'))
    path2RunFiles = glob.glob(os.path.join(path2, '*.json'))
    
    maxRunNumber = 0
    
    for path1RunFile in path1RunFiles:
        name = path1RunFile.split("/")[-1]
        
        destRunFile = os.path.join(outputpath, name)
        
        runNumber = int(name.split(".")[0].split("_")[-1])
        maxRunNumber = max(runNumber, maxRunNumber)
        
        shutil.copyfile(path1RunFile, destRunFile)

    for path2RunFile in path2RunFiles:
        name = path2RunFile.split("/")[-1]
        
        runNumber = int(name.split(".")[0].split("_")[-1])
        newRunNumber = runNumber + maxRunNumber + 1 # to prevent zero-indexing collision
        
        destRunFilename = "agent_run_" + str(newRunNumber) + ".json"
        destRunFile = os.path.join(outputpath, destRunFilename)

        shutil.copyfile(path2RunFile, destRunFile)