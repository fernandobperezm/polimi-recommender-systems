import logging
import pandas as pd
import numpy as np

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
        'career_level':np.float64,
        'discipline_id':np.float64,
        'industry_id':np.uint8,
        'country':str,
        'region':str,
        'latitude':np.float64,
        'longitude':np.float64,
        'employment':np.uint8,
        'tags':str,
        'created_at':np.float64,
        'active_during_test':np.uint8}
    
    data = pd.read_csv(path, header=header, names=columns, sep=sep, dtype=dtype)
    logger.info('Reading {}'.format(path))
    logger.info('Columns: {}'.format(data.columns.values))
    data = data.fillna("0")
    
    # titles. // Based on StackOverflow.
    titles = data['title'].str.split(',').apply(pd.Series, 1).stack()
    titles.index = titles.index.droplevel(-1) # to line up with data's index
    titles.name = 'title' # needs a name to join
    #    del(data["title"])
#    data = data.join(titles)

    # tags. // Based on StackOverflow.
    tags = data['tags'].str.split(',').apply(pd.Series, 1).stack()
    tags.index = tags.index.droplevel(-1) # to line up with data's index
    tags.name = 'tags' # needs a name to join
    #    del(data["tags"])
    #    data = data.join(tags)
    
    items = data[item_key].unique()
    no_items = len(items)
    
    # TF-IDF.
    #    data['title_freq'] = data.groupby('title')['title'].transform('count') # Gettings the frequencies.
    #    data['tags_freq'] = data.groupby('tags')['tags'].transform('count') # Gettings the frequencies.
    # Creating frequency dictionaries.
    titles = titles.value_counts()
    tags = tags.value_counts()
    
    #    item_to_idx = pd.Series(data=np.arange(len(items)), index=items)
    #    idx_to_item = pd.Series(index=item_to_idx.data, data=item_to_idx.index)
    #
    #    titles = data['title'].unique()
    #    print(titles)
    #    title_to_idx = pd.Series(data=np.arange(len(titles)), index=titles)
    #    idx_to_title = pd.Series(index=title_to_idx.data, data=title_to_idx.index)
    #
    #    tags = data['tags'].unique()
    #    tag_to_idx = pd.Series(data=np.arange(len(tags)), index=tags)
    #    idx_to_tag = pd.Series(index=tag_to_idx.data, data=tag_to_idx.index)
    
    
    # map ids to indices
    #    data['item_idx'] = item_to_idx[data[item_key].values].values
    #    data['title_idx'] = title_to_idx[data['title'].values].values
    #    data['tag_idx'] = tag_to_idx[data['tags'].values].values
    return data, titles, tags, no_items

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
