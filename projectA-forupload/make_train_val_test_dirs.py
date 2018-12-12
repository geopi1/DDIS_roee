from config import *
import numpy as np

base_dir = config.base_dir

# list all the dirs
dirs = os.listdir(base_dir)

# go over all the folders
randOrder = np.random.permutation(len(dirs))
randDirs = [dirs[i] for i in randOrder]

# split to random folders for train,validate,test by 70,20,10 ratio
ValFoldersNum = int(len(dirs)*0.2)
ValFolderNames = [randDirs[i] for i in range(ValFoldersNum)]

TestFoldersNum = int(len(dirs)*0.1)
TestFolderNames = [randDirs[i+ValFoldersNum] for i in range(TestFoldersNum)]

TrainFoldersNum = len(dirs)-ValFoldersNum-TestFoldersNum
TrainFolderNames = [randDirs[i+ValFoldersNum+TestFoldersNum] for i in range(TrainFoldersNum)]

# Move validation into their mother folder
for i in range(ValFoldersNum):
    curPath = os.path.join(base_dir, ValFolderNames[i])
    newPath = os.path.join(base_dir, "Validate", ValFolderNames[i])
    os.renames(curPath, newPath)

# Move test into their mother folder
for i in range(TestFoldersNum):
    curPath = os.path.join(base_dir, TestFolderNames[i])
    newPath = os.path.join(base_dir, "Test", TestFolderNames[i])
    os.renames(curPath, newPath)

# Move train into their mother folder
for i in range(TrainFoldersNum):
    curPath = os.path.join(base_dir, TrainFolderNames[i])
    newPath = os.path.join(base_dir, "Train", TrainFolderNames[i])
    os.renames(curPath, newPath)
