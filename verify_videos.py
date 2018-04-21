from dataset import dataset
from models.c3d_model import C3DModel
import numpy as np
import math
if __name__ == "__main__":
    ds = dataset()
    model = C3DModel()
    for video in ds.training:
        video.resize(112, 112)
    vids = [vid.getSegments() for vid in ds.training]
    for vid in vids:
        print(type(vid))
        print("Predicting Video with shape: " + str(vid.shape))
        if vid.shape[0] % 32 != 0:
            vid = vid[:-(vid.shape[0] % 32)]
        print("Predicting Video with NEW shape: " + str(vid.shape))
        segs = np.split(vid, math.ceil(len(vid)/32))
        count = 1
        for seg in segs:
            print("Predicting for seg #%d..." % (count))
            _, _ = model.predict(seg)
            count += 1
    print("FINISHED!!!")
    