#measures
import numpy as np
import sys

def MeasureFactory(type:str):
    #we'll treat all as a distance measurement, so the similarities we'll treated as negative
    if type == "HellingerKernel":
        return HellingerKernelSimilarity()
    if type == "HellingerKernel-Median":
        return HellingerKernelMedianSimilarity()
    if type == "Intersection":
        return HistogramIntersectionSimilarity()
    if type == "L1":
        return L1Distance()
    if type == "L1-Median":
        return L1MedianDistance()
    if type == "L2":
        return L2Distance()
    if type == "X2":
        return Chi2Distance()
    if type == "X2-Median":
        return Chi2MedianDistance()
    if type == "LInfinity":
        return LInfinityDistance()
    else:
        sys.exit("ERROR: Unknow Measure type: " + type)

class Measurement(object):
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        sys.exit("ERROR: The __call__ method should be implemented by a subclass")


class HellingerKernelSimilarity(Measurement):
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        # Query and database histogram must be in the same histogram mode
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sum(np.sqrt(im1_hist[c] * im2_hist[c])))
        final_dist = np.array(dist).mean()
        return 1-final_dist, [1-d for d in dist]  #its a similiratiy (the greater the better) so if we wish to treat it as a distance we should multiply by -1

class HellingerKernelMedianSimilarity(Measurement):
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        # Query and database histogram must be in the same histogram mode
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sum(np.sqrt(im1_hist[c] * im2_hist[c])))
        final_dist = np.median(np.array(dist))
        return 1-final_dist, [1-d for d in dist]  #its a similiratiy (the greater the better) so if we wish to treat it as a distance we should multiply by -1


class HistogramIntersectionSimilarity(Measurement):
    #TODO TEST to ensure correct working
    def __init__(self):
        pass

    def __call__(self, im1_hist : list, im2_hist : list):
        # Query and database histogram must be in the same histogram mode
        # sum min(xi,yi) for every i
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sum(np.minimum(im1_hist[c], im2_hist[c])))
        final_dist = np.array(dist).mean()
        return 1-final_dist, [1-d for d in dist]
    
class L1Distance(Measurement):
    def __call__(self, im1_hist:list, im2_hist:list):
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sum(np.abs(im1_hist[c]-im2_hist[c])))
        final_dist = np.array(dist).mean()
        return final_dist, dist
    
class L1MedianDistance(Measurement):
    def __call__(self, im1_hist:list, im2_hist:list):
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sum(np.abs(im1_hist[c]-im2_hist[c])))
        final_dist = np.median(np.array(dist))
        return final_dist, dist
    
class L2Distance(Measurement):
    def __call__(self, im1_hist:list, im2_hist:list):
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.sqrt(np.sum((im1_hist[c]-im2_hist[c])**2)))
        final_dist = np.array(dist).mean()
        return final_dist, dist
    
class Chi2Distance(Measurement):
    def __call__(self, im1_hist:list, im2_hist:list):
        dist = []
        chi2 = lambda a, b: np.sum(np.divide((a-b)**2, (a + b), out=np.zeros_like(a), where=(a + b)!=0))
        for c in range(len(im1_hist)):
            dist.append(chi2(im1_hist[c], im2_hist[c]))
        final_dist = np.array(dist).mean()
        return final_dist, dist
    
class Chi2MedianDistance(Measurement):
    def __call__(self, im1_hist:list, im2_hist:list):
        dist = []
        chi2 = lambda a, b: np.sum(np.divide((a-b)**2, (a + b), out=np.zeros_like(a), where=(a + b)!=0))
        for c in range(len(im1_hist)):
            dist.append(chi2(im1_hist[c], im2_hist[c]))
        final_dist = np.median(np.array(dist))
        return final_dist, dist

class LInfinityDistance(Measurement):
    def __call__(self, im1_hist: list, im2_hist: list):
        if len(im1_hist) != len(im2_hist):
            sys.exit("ERROR: Histograms must have the same length")
        dist = []
        for c in range(len(im1_hist)):
            dist.append(np.max(np.abs(np.array(im1_hist[c]) - np.array(im2_hist[c]))))
        final_dist = np.array(dist).mean()
        return final_dist, dist