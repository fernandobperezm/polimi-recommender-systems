import logging
import pandas as pd
import sys as sys
import numpy as np
import sklearn.metrics.pairwise as sk_pair
import scipy.spatial as spatial
import scipy.sparse as sps
#import joblib

import multiprocessing as mp

import pdb

from datetime import datetime as dt

from pyspark import SparkContext, SparkConf

logger = logging.getLogger(__name__)

#conf = SparkConf().setAppName("RecSys").setMaster("spark://Fernandos-MacBook-Pro.local:7077")
#conf = SparkConf().setAppName("RecSys")
conf = SparkConf()
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

def create_item_matrix(data,no_items,attributes):
    matrix = []
    title_dict = attributes[0]
    tags_dict = attributes[9]
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
#
def buildSimilaritiesMatrix(icm):
    # Shrinking factor.
#    H = 3
    # Creating the python dict for parallelization.
    data = list(
                zip(
                    icm[0:,0],
                    icm[:,1:]
                )
           )

    distData = sc.parallelize(data) # Parallelizing data.
##    Obtained from: http://apache-spark-user-list.1001560.n3.nabble.com/Computing-cosine-similiarity-using-pyspark-td6254.html
    #sim = distData.cartesian(distData).map(lambda kv: ((kv[0][0],kv[1][0]),1-spatial.distance.cosine(kv[0][1],kv[1][1]))).collect()
#    sim = distData.cartesian(distData)\
#            .map(\
#                lambda kv:\
#                    ((kv[0][0],kv[1][0]),\
#                    np.dot(kv[0][1],kv[1][1]) / (np.linalg.norm(kv[0][1])*np.linalg.norm(kv[1][1]) + H)))\
#            .collect()

    sim = distData.cartesian(distData).filter(lambda kv: kv[0][0] < kv[1][0]).fold({},build).collect()


    return sim

def build(accum, kv):
    H = 3
    k = 20
    k1,v1,k2,v2 = kv[0][0],kv[0][1],kv[1][0],kv[1][1]
    sim = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + H)
    
    if (accum == None):
        accum = dict()
    
    if (k1 in accum):
        min_index = np.argmin(accum[k1])
        min_value = accum[k1][min_index]
    
        if (sim > min_value):
            accum[k1][min_index] = sim
    else:
        sim_ij = np.zeros(shape = (k))
        sim_ij[0] = sim
        accum[k1] = sim_ij
        #print("KEY1: " + str(k1) + " - KEY2: " + str(k2))
    #print("KEY1: " + str(k1) + " - KEY2: " + str(k2))

    return accum





###################################### OPTION 2, using joblib.
#def buildSimilaritiesMatrix(icm):
#    k = 50
#    # Parallelization.
#    num_cores = mp.cpu_count()
#    n_rows = 199
#    # Split of data to each process.
#    arguments = [(n_rows,icm[n_rows*i:,:],i,k) for i in range(0,844)]
#    results = joblib.Parallel(n_jobs = num_cores)(joblib.delayed(calculateCosine) (row,mat,process_number,k) for row,mat,process_number,k in arguments)
#
##    sim = tuple(results[i][0] for i in range(len(results)))
##    id = tuple([results[i][1] for i in range(len(results))])
##
##    sij = np.vstack(sim)
##    ids = np.vstack(id)
#
#    return 0
#
#
#def calculateCosine(n_rows,matrix,process_number,k):
#    sim_ij = np.zeros(shape = (n_rows,k+1))
#    obj_ij = np.zeros(shape = (n_rows,k+1))
#
#    for i in range(n_rows):
#        tic = dt.now()
#        v1 = matrix[i,1:] # Vector of attributes.
#        job_id = matrix[i,0] # Jobid.
#        sim_ij[i,0] = job_id # Setting jobid on similarities.
#        obj_ij[i,0] = job_id # Setting jobid on objets.
#        row_sim = sim_ij[i,1:] # Getting row of limilarities.
#        for j in range(i+1,len(matrix)):
#            min_index = np.argmin(row_sim)
#            min_value = np.nanmin(row_sim)
#            v2 = matrix[j,1:]
#            sim = 1 - spatial.distance.cosine(v1,v2)
#            if (sim > min_value):
#                #print(str(sim))
#                # Set the similarity on the position +1 on the similarities row. (0 is for id)
#                sim_ij[i,min_index+1] = sim
#                obj_ij[i,min_index+1] = j
#
#        logger.info('Training completed built in {}'.format(dt.now() - tic))
#
#    fname1 = "models/cbr_sim_" + str(process_number) + ".csv"
#    fname2 = "models/cbr_index_" + str(process_number) + ".csv"
#    np.savetxt(fname=fname1, X=sim_ij, delimiter='!', newline='\n', header='', footer='', comments='# ')
#    np.savetxt(fname=fname2, X=obj_ij, delimiter='!', newline='\n', header='', footer='', comments='# ')
#
#    return 0
