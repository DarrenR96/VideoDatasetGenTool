import subprocess

class SearchAgent():
    def __init__(self, lowerTarget, upperTarget, initialSearchParams, metricRange = (0,100)):
        self.targetLower = lowerTarget
        self.targetUpper = upperTarget
        self.range = metricRange
        self.currentParams = initialSearchParams
        self.currentValue = None
    
    def updateSearch(self, inputVideo, crf, codec="H264"):
        if codec == "H264":
            command = f"ffmpeg -i {inputVideo} -c:v libx264 -preset medium -crf {crf} -c:a copy {outputVideo}"
        codecRun = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
        self.processOut, self.processErr = codecRun.communicate()
        


    def updateParams(self, lowerTarget, upperTarget, searchParams, currentValue):
        self.targetLower = lowerTarget
        self.targetUpper = upperTarget
        self.range = metricRange
        self.currentValue = currentValue


