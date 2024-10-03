#measures
import numpy as np


class Similarity(object):
    def __init__(self, metric : str = 'HellingerKernel'):
        self.metric = metric

    def __call__(self, im1_hist : list, im2_hist : list):
        # Query and database histogram must be in the same histogram mode
        if self.metric == 'HellingerKernel':
            dist = []
            for c in range(len(im1_hist)):
                dist.append(np.sum(np.sqrt(im1_hist[c] * im2_hist[c])))
            dist = np.array(dist).mean()
            return dist
        else:
            #Placeholder
            return None