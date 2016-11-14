import logging
import pandas as pd
import sys as sys
import numpy as np
import sklearn.metrics.pairwise as sk_pair
import scipy.spatial as spatial
import scipy.sparse as sps

import pdb

from datetime import datetime as dt

logger = logging.getLogger(__name__)

class TopPop(object):
    """Top Popular recommender"""
    def __init__(self):
        super(TopPop, self).__init__()

    def fit(self, train):
        if isinstance(train, sps.csr_matrix):
            # convert to csc matrix for faster column-wise sum
            train_csc = train.tocsc()
        else:
            train_csc = train
        item_pop = (train_csc > 0).sum(axis=0)	# this command returns a numpy.matrix of size (1, nitems)
        item_pop = np.asarray(item_pop).squeeze() # necessary to convert it into a numpy.array of size (nitems,)
        self.pop = np.argsort(item_pop)[::-1]

    def recommend(self, profile, k=None, exclude_seen=True):
        #unseen_mask = np.in1d(self.pop, profile, assume_unique=True, invert=True)
        #return self.pop[unseen_mask][:k]
        return self.pop[:k]
