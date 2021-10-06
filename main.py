from resources import VideoDataset
import json


a = VideoDataset("/home/ramsookd/DatasetGeneration/testDataYUV/", "yuv", "/home/ramsookd/DatasetGeneration/DataOutput/")
a.genVideoList()
a.setDistribution()
a.optimizeDataset(logs=True)




with open("DataOutput/recommendedEncodes.json", 'w') as fp:
    json.dump(a.dataset, fp)

with open("DataOutput/VideoIdx.json", 'w') as fp:
    json.dump(a.videoIds, fp)
