import os 
import random
import subprocess
import re
import random
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
defaultDist = [(5,(i*5,(i+1)*5)) for i in range(0,20)]
defaultDist = [(1,(2,3)),(1,(3,4))]

defaultDist = [(5,(0,20)), (5, (20,40))]
defaultDist = [(2,(30,40)), (2, (0,10))]

logsDF = pd.DataFrame({'VideoName': [], 'Iteration':[], 'Codec': [], 'CRF':[], 'LowerTarget':[], 'UpperTarget':[], 'VMAF':[], 'CriteriaExists': []})

class VideoDataset():
    def __init__(self, folder, extension="yuv", dataOutput = 'DataOutput/'):
        self.folder = folder
        self.extension = "."+extension
        self.videoFilePaths = []
        self.distribution = []
        self.dataset = {}
        self.videoIds = {}
        self.dataOutput = dataOutput
    
    def genVideoList(self):
        for (dirPath, dirNames, fileNames) in os.walk(self.folder):
            for currentFile in fileNames:
                if os.path.splitext(currentFile)[1] == self.extension:
                    self.videoFilePaths.append(os.path.join(dirPath, currentFile))
            for idx in range(len(self.videoFilePaths)):
                _, tail = os.path.split(self.videoFilePaths[idx])
                filename, _ =  os.path.splitext(tail)
                self.videoIds[idx] = {
                    'raw': self.videoFilePaths[idx],
                    'degraded': f"{self.dataOutput}{filename}_comp{self.extension}"
                }
    
    def setDistribution(self, userDist = defaultDist, shuffle=True):
        total = sum(i for i,j in userDist)
        if total != len(self.videoFilePaths):
            raise Exception(f"Sum of videos in distribution must be equal to total number of videos, {total}!={len(self.videoFilePaths)}")
        else: 
            self.distribution = userDist
            videoIdsKeys = list(self.videoIds.keys())
            if shuffle == True:
                random.shuffle(videoIdsKeys)
            for i in range(len(self.distribution)):
                if i == 0:
                    listLowerSlice = 0
                slicedVideos = videoIdsKeys[listLowerSlice:listLowerSlice+self.distribution[i][0]]
                slicedVideos = {x:{'CRF':0, 'VMAF':0} for x in slicedVideos}
                listLowerSlice = listLowerSlice+self.distribution[i][0]
                self.dataset[i] = {
                    'Capactity' : self.distribution[i][0],
                    'Lower' : self.distribution[i][1][0],
                    'Upper' : self.distribution[i][1][1],
                    'Videos' : slicedVideos
                }
    
    def optimizeDataset(self, vmafpath='/home/ramsookd/vmaf/model/vmaf_v0.6.1.json', searchAgents=5, logs=False):
        if logs:
            global logsDF
        for key in self.dataset:
            lowerBound = self.dataset[key]['Lower']
            upperBound = self.dataset[key]['Upper']
            for video in self.dataset[key]['Videos']:
                videoName = self.videoIds[video]['raw']
                videoOutput = self.videoIds[video]['degraded']
                self.searchAgentCollection = SearchAgentCollection(searchAgents, lowerBound, upperBound, vmafpath, metricRange=(0,100))
                self.dataset[key]['Videos'][video]['VMAF'], self.dataset[key]['Videos'][video]['CRF'] = self.searchAgentCollection.updateCollectionSearch(videoName, videoOutput, logs=logs)
        
        if logs:
            logsDF.to_csv("DataOutput/suggestedEncodes.csv", index=False)



class SearchAgent():
    def __init__(self, lowerTarget, upperTarget, initialSearchParams, vmafPath, metricRange = (0,100)):
        self.targetLower = lowerTarget
        self.targetUpper = upperTarget
        self.range = metricRange
        self.currentParams = initialSearchParams
        self.currentValue = None
        self.vmafPath = vmafPath
    
    def updateSearch(self, inputVideo, outputVideo, crf, codec="H264"):
        outputVideoParallel = outputVideo[:len(outputVideo)-4]+str(os.getpid())+".mkv"
        if codec == "H264":
            command = f"ffmpeg -y -video_size 1280x720 -i {inputVideo} -c:v libx264 -preset medium -crf {crf} -c:a copy {outputVideoParallel}"
        if codec == "H265":
            command = f"ffmpeg -y -video_size 1280x720 -i {inputVideo} -c:v libx265 -preset medium -crf {crf} -c:a copy {outputVideoParallel}"
        codecRun = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, cwd='/')
        self.processOut, self.processErr = codecRun.communicate()
        codecRun.wait()

        outputVideoParallelYUV = outputVideoParallel[:len(outputVideoParallel)-4] + '.yuv'
        command = f"ffmpeg -y -i {outputVideoParallel} -c:v rawvideo -pix_fmt yuv420p {outputVideoParallelYUV}"
        mp4ToYUV = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, cwd='/')
        self.processOut, self.processErr = mp4ToYUV.communicate()
        mp4ToYUV.wait()

        qualityMeasure = f'ffmpeg -video_size 1280x720 -pix_fmt yuv420p -i {inputVideo} -video_size 1280x720 -pix_fmt yuv420p -i {outputVideoParallelYUV} -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; [1:v]setpts=PTS-STARTPTS[distorted]; [distorted][reference]libvmaf=model_path={self.vmafPath}"     -f null -'
        qualityRun = subprocess.Popen(qualityMeasure, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, cwd='/')
        self.processOut, self.processErr = qualityRun.communicate()
        qualityRun.wait()
        vmaf = self.processErr.splitlines()[-1].decode('utf-8')
        vmaf = float(re.search('VMAF score: (.*?)$', vmaf).group(1))
        
        self.currentValue = vmaf
        print(f"Currently searching : {inputVideo} \nCurrent CRF : {crf} \nCurrent VMAF : {self.currentValue}")

        deleteCommand = f"rm {outputVideoParallel}"
        deleteRun = subprocess.Popen(deleteCommand, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, cwd='/')
        self.processOut, self.processErr = deleteRun.communicate()

        deleteCommand = f"rm {outputVideoParallelYUV}"
        deleteRun = subprocess.Popen(deleteCommand, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, cwd='/')
        self.processOut, self.processErr = deleteRun.communicate()

        return self.currentValue

    def updateParams(self, lowerTarget, upperTarget, searchParams, currentValue):
        self.targetLower = lowerTarget
        self.targetUpper = upperTarget
        self.currentParams = searchParams
        self.currentValue = currentValue


class SearchAgentCollection():
    def __init__(self, numAgents, lowerTarget, upperTarget, vmafPath, metricRange = (0,100)):
        self.numAgents = numAgents
        self.agents = []
        self.crfValues = np.rint(np.linspace(0,51,num=self.numAgents)).astype(int).tolist()
        self.lowerTarget = lowerTarget
        self.upperTarget = upperTarget
        self.criteriaFound = False
        for i in range(self.numAgents):
            self.agents.append(SearchAgent(lowerTarget, upperTarget, self.crfValues[i], vmafPath, metricRange = (0,100)))
    
    def updateCollectionSearch(self, inputVideo, outputVideo, codec="H264", topMatchNum = 3, maxDepth = 5, logs=False):
        depth = 0
        global logsDF
        while self.criteriaFound != True:
            vmafScores = []
            processPool = []
            with ProcessPoolExecutor() as executor:
                for agent in self.agents:
                    processPool.append(executor.submit(agent.updateSearch,inputVideo, outputVideo, agent.currentParams, codec))
                for process in processPool:
                    vmafScores.append(process.result())
            for i in range(len(vmafScores)):
                self.agents[i].currentValue = vmafScores[i]
            if logs:
                for agent in self.agents:
                    if self.lowerTarget <= agent.currentValue <= self.upperTarget:
                        criteriaExists = 1
                    else:
                        criteriaExists = 0
                    new_row = {'VideoName': inputVideo, 'Iteration': depth, 'Codec': codec, 'CRF': agent.currentParams, 'LowerTarget':self.lowerTarget, 'UpperTarget':self.upperTarget, 'VMAF':agent.currentValue, 'CriteriaExists': criteriaExists}
                    logsDF = logsDF.append(new_row, ignore_index=True)

            for agent in self.agents:
                if self.lowerTarget <= agent.currentValue <= self.upperTarget:
                    self.criteriaFound = True
                    return agent.currentValue, agent.currentParams
            vmafMidpoint = (self.upperTarget + self.lowerTarget)/2
            vmafScores = [absDist(i,vmafMidpoint) for i in vmafScores]
            sortedScoresIdx = np.argsort(vmafScores)[:topMatchNum]
            topAgents = [self.agents[idx].currentParams for idx in sortedScoresIdx]
            newLowerCrf = min(topAgents)
            newUpperCrf = max(topAgents)
            self.crfValues = np.rint(np.linspace(newLowerCrf,newUpperCrf,num=self.numAgents)).astype(int).tolist()
            for i in range(self.numAgents):
                self.agents[i].updateParams(self.lowerTarget, self.upperTarget, self.crfValues[i], self.agents[i].currentValue)
            if depth == maxDepth:
                return self.agents[sortedScoresIdx[0]].currentValue, self.agents[sortedScoresIdx[0]].currentParams
            depth += 1

    # def updateCollectionSearch(self, inputVideo, outputVideo, codec="H264", topMatchNum = 3):
    #     selectedValue = None
    #     selectedParams = None
    #     while self.criteriaFound != True:
    #         vmafScores = []
    #         for agent in self.agents:
    #             agent.updateSearch(inputVideo, outputVideo, agent.currentParams, codec)
    #             vmafScores.append(agent.currentValue)
    #             if self.lowerTarget <= agent.currentValue <= self.upperTarget:
    #                 self.criteriaFound = True
    #                 return agent.currentValue, agent.currentParams
    #         vmafMidpoint = (self.upperTarget + self.lowerTarget)/2
    #         vmafScores = [absDist(i,vmafMidpoint) for i in vmafScores]
    #         sortedScoresIdx = np.argsort(vmafScores)[:topMatchNum]
    #         topAgents = [self.agents[idx].currentParams for idx in sortedScoresIdx]
    #         newLowerCrf = min(topAgents)
    #         newUpperCrf = max(topAgents)
    #         self.crfValues = np.rint(np.linspace(newLowerCrf,newUpperCrf,num=self.numAgents)).astype(int).tolist()
    #         for i in range(self.numAgents):
    #             self.agents[i].updateParams(self.lowerTarget, self.upperTarget, self.crfValues[i], self.agents[i].currentValue)
    #         print(f"Current Video: {inputVideo} \nTarget Score: {vmafMidpoint} \nNew CRF range: {newLowerCrf},{newUpperCrf}")  

def absDist(a, b):
    return(abs(a-b))



        
        
