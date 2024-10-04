#measures
import numpy as np
import sys

def MeasureFactory(type:str):
    #we'll treat all as a distance measurement, so the similarities we'll treated as negative
    if type == "HellingerKernel":
        return HellingerKernelSimilarity()
    if type == "Intersection":
        return HistogramIntersectionSimilarity()
    else:
        sys.exit("ERROR: Unknow Measure type: " + type)

class Measurement(object):
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        sys.exit("ERROR: The __call__ method should be implemented by a subclass")
        # # Query and database histogram must be in the same histogram mode
        # if self.metric == 'HellingerKernel':
        #     dist = []
        #     for c in range(len(im1_hist)):
        #         dist.append(np.sum(np.sqrt(im1_hist[c] * im2_hist[c])))
        #     dist = np.array(dist).mean()
        #     return dist
        # else:
        #     #Placeholder
        #     return None

class HellingerKernelSimilarity(Measurement):
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        # Query and database histogram must be in the same histogram mode
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sum(np.sqrt(im1_hist[c] * im2_hist[c])))
        dist = np.array(dist).mean()
        return 1-dist #its a similiratiy (the greater the better) so if we wish to treat it as a distance we should multiply by -1

class HistogramIntersectionSimilarity(Measurement):
    #TODO TEST to ensure correct working
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        # Query and database histogram must be in the same histogram mode
        # sum min(xi,yi) for every i
        return 1-sum ([ min(x,y) for x,y in zip(im1_hist, im2_hist) ])#its a similiratiy (the greater the better) so if we wish to treat it as a distance we should multiply by -1