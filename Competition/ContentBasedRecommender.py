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

from datetime import datetime as dt

from pyspark import SparkContext, SparkConf

logger = logging.getLogger(__name__)

#conf = SparkConf().setAppName("RecSys").setMaster("spark://Fernandos-MacBook-Pro.local:7077")
#conf = SparkConf().setAppName("RecSys")
conf = SparkConf()
sc = SparkContext(conf=conf)

class ContentBased(object):
    """ContentBased recommender"""
    def __init__(self,similarities):
        super(ContentBased, self).__init__()
        # Building the ICM matrix.
        self.similarities = similarities
    
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



def get_most_popular_attributes(items_matrix, items_ids):
    items_matrix = items_matrix[items_matrix['id'].isin(items_ids)]

    # titles. // Based on StackOverflow.
    titles = items_matrix['title'].str.split(',').apply(pd.Series, 1).stack()
    titles.index = titles.index.droplevel(-1) # to line up with data's index
    titles.name = 'title' # needs a name to join

    # tags. // Based on StackOverflow.
    tags = items_matrix['tags'].str.split(',').apply(pd.Series, 1).stack()
    tags.index = tags.index.droplevel(-1) # to line up with data's index
    tags.name = 'tags' # needs a name to join
    
    items = items_matrix['id'].unique()
    no_items = len(items)

    titles = titles.value_counts()
    tags = tags.value_counts()
    career_level = items_matrix['career_level'].value_counts() # 242 NaN's, all zeros are NaN's
    discipline_id = items_matrix['discipline_id'].value_counts() # No NaN.
    industry_id = items_matrix['industry_id'].value_counts() #No NaN.
    country = items_matrix['country'].value_counts() # No NaN.
    region = items_matrix['region'].value_counts() #No NaN.
    latitude = items_matrix['latitude'].value_counts() # 12250 NaN's, all zeros are NaN's
    longitude = items_matrix['longitude'].value_counts() # 12250 NaN's, all zeros are NaN's
    employment = items_matrix['employment'].value_counts() # No NaN.
    created_at = items_matrix['created_at'].value_counts() # 44285 NaN's, all zeros are NaN's
    active_during_test = items_matrix['active_during_test'].value_counts() # No Nan.
    
    attr = [titles,career_level,discipline_id,industry_id,country,region,latitude,\
            longitude, employment,tags,created_at,active_during_test]

    return items_matrix, no_items, attr
    

def create_item_matrix(data,no_items,attributes):
#    pdb.set_trace()
    matrix = []
    title_dict = attributes[0]
    tags_dict = attributes[9]
    tf = 1/12
    i_row = np.ndarray(shape=(13)) # ID, Attrs.
    
    for index,row in data.iterrows():
        i_row[0] = row['id']
        
        # Iteration over columns.
        i = 1
        for actual_attr_dict in attributes:
            if ( (actual_attr_dict.name == "title") or (actual_attr_dict.name == "tags") ):
                for job in row[actual_attr_dict.name].split(","):
                    if (job == "0"):
                        i_row[i] += 0
                    else:
                        idf = np.log10(no_items/ actual_attr_dict[job])
                        i_row[i] += 1*tf*idf
            else:
                attr = row[actual_attr_dict.name]
                
                if (attr == 0):
                    i_row[i] += 0
                else:
                    idf = np.log10(no_items/ actual_attr_dict.ix[attr])
                    i_row[i] += 1*tf*idf
            i +=1

        matrix.append(i_row)
        i_row = np.ndarray(shape=(13))

    matrix = np.array(matrix)
    return matrix
#
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
    sim = distData\
            .cartesian(distData)\
            .filter(filtrar)\
            .map(calculateCosine)\
            .collect()
            
    # Convert the data into a matrix.
    # Extracted from: http://stackoverflow.com/questions/21446323/converting-a-dictionary-of-tuples-into-a-numpy-matrix

    keys = np.array( [key for dic in sim for key in dic.keys()] )
    vals = np.array( [value for dic in sim for value in dic.values()] )

    unq_keys, key_idx = np.unique(keys, return_inverse=True)
    key_idx = key_idx.reshape(-1, 2)

    n = len(unq_keys)
    sim = np.zeros((n, n) ,dtype=vals.dtype)
    sim[key_idx[:,0], key_idx[: ,1]] = vals

    return sim,unq_keys,key_idx

def calculateCosine(kv):
    H = 3
    k1, v1, k2, v2 = kv[0][0],kv[0][1],kv[1][0],kv[1][1]
    
    sim = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + H)
    
    return {(k1,k2):sim}

def filtrar(kv):
    k1, v1, k2, v2 = kv[0][0],kv[0][1],kv[1][0],kv[1][1]

    return True if (k1 < k2) else False


##################################### OPTION 2, using joblib.
#def buildSimilaritiesMatrix(icm):
#    k = 50
#    # Parallelization.
#    num_cores = mp.cpu_count()
#    n_rows = 100
#    # Split of data to each process.
#    arguments = [(n_rows,icm[n_rows*i:,:],i,k) for i in range(0,5)]
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
#        #logger.info('Training completed built in {}'.format(dt.now() - tic))
#
#    fname1 = "models/cbr_sim_" + str(process_number) + ".csv"
#    fname2 = "models/cbr_index_" + str(process_number) + ".csv"
#    np.savetxt(fname=fname1, X=sim_ij, delimiter='!', newline='\n', header='', footer='', comments='# ')
#    np.savetxt(fname=fname2, X=obj_ij, delimiter='!', newline='\n', header='', footer='', comments='# ')
#
#    return 0
