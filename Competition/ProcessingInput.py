import logging
import pandas as pd
import numpy as np
import pdb

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")

def read_dataset_up(path, header=None, columns=None, user_key='user_id', item_key='item_id', rating_key='rating', sep=','):
    logger.info('Reading {}'.format(path))
    data = pd.read_csv(path, header=header, names=columns, sep=sep)
    logger.info('Columns: {}'.format(data.columns.values))
    # build user and item maps (and reverse maps)
    # this is used to map ids to indexes starting from 0 to nitems (or nusers)
    
    # Separating indices.
    items_str = data[item_key]
    jobs = set()
    for l_jobs in items_str:
        for job in l_jobs.split(","):
            jobs.add(job)

    items = pd.Series(list(jobs))
    users = data[user_key].unique()
    item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
    user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
    idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
    idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)
    # map ids to indices
    data['item_idx'] = item_to_idx[data[item_key].values].values
    data['user_idx'] = user_to_idx[data[user_key].values].values
    
    return data,idx_to_user,idx_to_item

def read_dataset_ip(path, header=None, columns=None, user_key='user_id', item_key='item_id', rating_key='rating', sep=','):
    dtype = {'id':np.uint64,
        'title':str,
        'career_level':str,
        'discipline_id':str,
        'industry_id':str,
        'country':str,
        'region':str,
        'latitude':str,
        'longitude':str,
        'employment':str,
        'tags':str,
        'created_at':str,
        'active_during_test':str}
    
    data = pd.read_csv(path, header=header, names=columns, sep=sep, dtype=dtype)
    logger.info('Reading {}'.format(path))
    logger.info('Columns: {}'.format(data.columns.values))
    data = data.fillna("0")
    
    # titles. // Based on StackOverflow.
    titles = data['title'].str.split(',').apply(pd.Series, 1).stack()
    titles.index = titles.index.droplevel(-1) # to line up with data's index
    titles.name = 'title' # needs a name to join


    # tags. // Based on StackOverflow.
    tags = data['tags'].str.split(',').apply(pd.Series, 1).stack()
    tags.index = tags.index.droplevel(-1) # to line up with data's index
    tags.name = 'tags' # needs a name to join
    
    items = data[item_key].unique()
    no_items = len(items)

    titles = titles.value_counts() # Zeros are NaN's
    tags = tags.value_counts() # Zeros are NaN's 
    career_level = data['career_level'].value_counts() # 242 NaN's, all zeros are NaN's
    discipline_id = data['discipline_id'].value_counts() # No NaN.
    industry_id = data['industry_id'].value_counts() #No NaN.
    country = data['country'].value_counts() # No NaN.
    region = data['region'].value_counts() #No NaN.
    latitude = data['latitude'].value_counts() # 12250 NaN's, all zeros are NaN's
    longitude = data['longitude'].value_counts() # 12250 NaN's, all zeros are NaN's
    employment = data['employment'].value_counts() # No NaN.
    created_at = data['created_at'].value_counts() # 44285 NaN's, all zeros are NaN's
    active_during_test = data['active_during_test'].value_counts() # No Nan.

    attr = [titles,career_level,discipline_id,industry_id,country,region,latitude,\
            longitude, employment,tags,created_at,active_during_test]

    return data, no_items, attr,

def read_dataset_inter(path, header=None, columns=None, user_key='user_id', item_key='item_id', rating_key='rating', sep=','):
    logger.info('Reading {}'.format(path))
    data = pd.read_csv(path, header=header, names=columns, sep=sep)
    logger.info('Columns: {}'.format(data.columns.values))
    # build user and item maps (and reverse maps)
    # this is used to map ids to indexes starting from 0 to nitems (or nusers)
    items = data[item_key].unique()
    users = data[user_key].unique()
    item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
    user_to_idx = pd.Series(data=np.arange(len(users)), index=users)
    idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
    idx_to_user = pd.Series(index=user_to_idx.data, data=user_to_idx.index)
    # map ids to indices
    data['item_idx'] = item_to_idx[data[item_key].values].values
    data['user_idx'] = user_to_idx[data[user_key].values].values
    return data, idx_to_user, idx_to_item


def read_dataset_tu(path, header=None, columns=None, user_key='user_id', item_key='item_id', rating_key='rating', sep=','):
    logger.info('Reading {}'.format(path))
    users = pd.read_csv(path, header=header, names=columns, sep=sep)
    logger.info('Columns: {}'.format(users.columns.values))
    return users
