import logging
import pandas as pd
import sys as sys
import numpy as np
import sklearn.metrics.pairwise as sk_pair
import scipy.spatial as spatial
import scipy.sparse as sps
import joblib

import multiprocessing as mp

import pdb

from pyspark import SparkContext, SparkConf

logger = logging.getLogger(__name__)

#conf = SparkConf().setAppName("RecSys").setMaster("spark://Fernandos-MacBook-Pro.local:7077")
conf = SparkConf().setAppName("RecSys")
sc = SparkContext(conf=conf)

class ContentBased(object):
    """ContentBased recommender"""
    def __init__(self,icm):
        super(ContentBased, self).__init__()
        # Building the ICM matrix.
        self.icm = icm
    
    def buildSimilaritiesMatrix(self): # the easiest one.
        self.icm =  np.dot(self.icm,self.icm.transpose())
    
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
        unseen_mask = np.in1d(self.pop, profile, assume_unique=True, invert=True)
        return self.pop[unseen_mask][:k]

def create_item_matrix(data,title_dict,tags_dict, no_items):
    matrix = []
    tf = 1/12
    i_row = [0,0,0] # ID,Title, Tags.
    i = 0;
    tags = -1
    titles = -1
    index_old = 0
    for index,row in data.iterrows():
        if (index != index_old):
            matrix.append(i_row)
            index_old = index
            i_row = [0,0,0]
            i += 1

        i_row[0] = row['id']
        
        for job in row['title'].split(","):
            if (job == "0"):
                i_row[1] += 0
            else:
                idf = np.log10(no_items/ title_dict[job])
                i_row[1] += 1*tf*idf
        
        for job in row['tags'].split(","):
            if (job == "0"):
                i_row[2] += 0
            else:
                idf = np.log10(no_items/ tags_dict[job])
                i_row[2] += 1*tf*idf
                    
    matrix = np.array(matrix)
    return matrix

def buildSimilaritiesMatrix(icm):
    # Creating the python dict for parallelization.
    data = list(
                zip(
                    icm[0:,0],
                    icm[:,1:]
                )
           )

    distData = sc.parallelize(data) # Parallelizing data.
##    Obtained from: http://apache-spark-user-list.1001560.n3.nabble.com/Computing-cosine-similiarity-using-pyspark-td6254.html
    #sim = distData.cartesian(distData).map(lambda ((k1,v1),(k2,v2)): ((k1,k2),1-spatial.distance.cosine(v1,v2))).collect()
    sim = distData.cartesian(distData).map(lambda kv: ((kv[0][0],kv[1][0]),1-spatial.distance.cosine(kv[0][1],kv[1][1]))).collect()
    return sim
    




#def buildSimilaritiesMatrix(icm):
#    # Just create the matrix as the formula given in class.
#    matrix = icm[:,1:]
#    sij = np.empty(shape = (len(icm),len(icm)))
#    
#    # Parallelization.
#    #pool = mp.Pool(mp.cpu_count())
#    num_cores = mp.cpu_count()
#    n_rows = 41989
#    #pdb.set_trace()
#    arguments = [(n_rows,matrix[n_rows*i:n_rows*(i+1),:]) for i in range(0,mp.cpu_count())] # Split the data (for my pc is 1,5)
#    results = joblib.Parallel(n_jobs =num_cores)(joblib.delayed(calculateCosine) (row,mat) for row,mat in arguments)
##    [(n_rows,matrix[n_rows*i:n_rows*(i+1),:]) for i in range(0,mp.cpu_count())] # Split the data (for my pc is 1,5)
##    results = [pool.apply_async(calculateCosine,arg) for arg in arguments ]
#
#    i = 0
#    result = tuple(results)
#    
#    sij = np.vstack(result)
#    
#    H = 3 # Shrink Term
#    return sij
##    for i in range(len(icm)):
##        v1 = matrix[i,:]
##        for j in range(i+1,len(icm)):
##            tic = dt.now()
##            v2 = matrix[j,:]
##            sij[i,j] = ( np.dot(v1,v2) ) / ( np.linalg.norm(v1,2) * np.linalg.norm(v2,2) + H)
##            logger.info('Training completed built in {}'.format(dt.now() - tic))
##            break
##        break
#
#    for i in range(len(icm)):
#        print(i)
#        v1 = matrix[i,:]
#        for j in range(i+1,len(icm)):
#            v2 = matrix[j,:]
#            sij[i,j] = 1 - spatial.distance.cosine(v1,v2)
#
#        if (i == 10000):
#            print("STOP")
#            break
#
#    return sij

#
#    def f(x):
#        return x*x
#
#    wv1 Pool(5) as p:
#        print(p.map(f, [1, 2, 3]))


#    def f(x):
#        return x*x
#
#    if __name__ == '__main__':
#        wv1 Pool(5) as p:
#            print(p.map(f, [1, 2, 3]))

#def calculateCosine(n_rows,matrix):
#    sij = np.empty(shape = (n_rows,len(matrix)))
#
#    for i in range(n_rows):
#        v1 = matrix[i,:]
#        for j in range(i+1,len(matrix)):
#            v2 = matrix[j,:]
#            sij[i,j] = 1 - spatial.distance.cosine(v1,v2)
#
#    return sij
