# Politecnico di Milano.
# basic_recommender.py
#
# Description: This file is a template for building RecSys. It was provided in classes.
#
# Author: Massimo.
# Modified by: Fernando Pérez on 29/10/16.


import pdb
import argparse
from datetime import datetime as dt

# Own Files.
import ProcessingInput as pi
import ContentBasedRecommender as cbr
from metrics import roc_auc, precision, recall, map, ndcg, rr

import logging
logger = logging.getLogger(__name__)

def holdout_split(data, perc=0.8, seed=1234):
	# set the random seed
	rng = np.random.RandomState(seed)
	# shuffle data
	nratings = data.shape[0]
	shuffle_idx = rng.permutation(nratings)
	train_size = int(nratings * perc)
	# split data according to the shuffled index and the holdout size
	train_split = data.ix[shuffle_idx[:train_size]]
	test_split = data.ix[shuffle_idx[train_size:]]
	return train_split, test_split

def df_to_csr(df, nrows, ncols, user_key='user_idx', item_key='item_idx', rating_key='rating'):
	rows = df[user_key].values
	columns = df[item_key].values
	ratings = df[rating_key].values
	shape = (nrows, ncols)
	# using the 4th constructor of csr_matrix 
	# reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
	return sps.csr_matrix((ratings, (rows, columns)), shape=shape)

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
		unseen_mask = np.in1d(self.pop, profile, assume_unique=True, invert=True)
		return self.pop[unseen_mask][:k]


# let's use an ArgumentParser to read input arguments
parser = argparse.ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('header', type=int, default=None)
parser.add_argument('sep', type=str, default=',')
parser.add_argument('item_key', type=str, default='item_id')
parser.add_argument('--holdout_perc', type=float, default=0.8)
#parser.add_argument('--head', type=int, default=None)
parser.add_argument('--columns', type=str, default=None)
#parser.add_argument('--sep', type=str, default=',')
parser.add_argument('--user_key', type=str, default='user_id')
#parser.add_argument('--item_key', type=str, default='item_id')
parser.add_argument('--rating_key', type=str, default='rating')
parser.add_argument('--rnd_seed', type=int, default=1234)
args = parser.parse_args()

# Getting all files to be read.
args.dataset = args.dataset.split(',')

# Getting all item keys to be read.
args.item_key = args.item_key.split(',')

# convert the column argument to list
if args.columns is not None:
	args.columns = args.columns.split(',')

## read the users profiles.
#data_up, users_up, items_up = read_dataset_up(
#                                              args.dataset[0],
#                                              header=args.header,
#                                              sep=args.sep,
#                                              columns=args.columns,
#                                              item_key=args.item_key[0],
#                                              user_key=args.user_key,
#                                              rating_key=args.rating_key)
#
#
#nusers_up, nitems_up = len(users_up), len(items_up)
#logger.info('The dataset user profile has {} users and {} items'.format(nusers_up, nitems_up))

# read the items profiles.
data_ip,no_items_ip, attr_ip = pi.read_dataset_ip(
                                  args.dataset[1],
                                  header=args.header,
                                  sep=args.sep,
                                  columns=args.columns,
                                  item_key=args.item_key[1],
                                  user_key=args.user_key,
                                  rating_key=args.rating_key)


ntitles_ip,ntags_ip = len(attr_ip[0]), len(attr_ip[9])
logger.info('The dataset items profile has {} items, {} titles and {} tags'.format(no_items_ip,ntitles_ip, ntags_ip))

# Read the training set.
#data_inter, users_inter,items_inter = pi.read_dataset_inter(
#                           args.dataset[2],
#                           header=args.header,
#                           sep=args.sep,
#                           columns=args.columns,
#                           item_key=args.item_key[2],
#                           user_key=args.user_key,
#                           rating_key=args.rating_key)
#
#
#nusers_inter,nitems_inter = len(users_inter),len(items_inter)
#logger.info('The dataset interactions has {} users and {} items'.format(nusers_inter,nitems_inter))

# Read the target users.
#users_tu = pi.read_dataset_tu(
#                          args.dataset[3],
#                          header=args.header,
#                          sep=args.sep,
#                          columns=args.columns,
#                          item_key=args.item_key,
#                          user_key=args.user_key,
#                          rating_key=args.rating_key)
#
#
#nusers_tu = len(users_tu)
#logger.info('The dataset target users has {} users'.format(nusers_tu))

data_ip = cbr.create_item_matrix(data_ip, no_items_ip,attr_ip)


## compute the holdout split
#logger.info('Computing the {:.0f}% holdout split'.format(args.holdout_perc*100))
#train_df, test_df = holdout_split(data_ip, perc=args.holdout_perc, seed=args.rnd_seed)
#train = df_to_csr(train_df, nrows=nusers_ip, ncols=nitems)
#test = df_to_csr(test_df, nrows=nusers, ncols=nitems)
#print(train)
#print(test)
#
# Content-Based recommender
logger.info('Building the ContentBased recommender')
#recommender = ContentBased(data_ip)
#data_ip = cbr.buildSimilaritiesMatrix(data_ip)
#print(data_ip)
#pdb.set_trace()
#tic = dt.now()
#logger.info('Training started')
#recommender.fit(train)
#logger.info('Training completed built in {}'.format(dt.now() - tic))
#
## ranking quality evaluation
#roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#at = 20
#neval = 0
#for test_user in range(nusers):
#	user_profile = train[test_user].indices #what is doing here?
#	relevant_items = test[test_user].indices
#	if len(relevant_items) > 0:
#		neval += 1
#		#
#		# TODO: Here you can write to file the recommendations for each user in the test split.
#		# WARNING: there is a catch with the item idx!
#		#
#		# this will rank *all* items
#		recommended_items = recommender.recommend(user_profile, exclude_seen=True)
#		# use this to have the *top-k* recommended items (warning: this can underestimate ROC-AUC for small k)
#		# recommended_items = recommender.recommend(user_profile, k=at, exclude_seen=True)
#		roc_auc_ += roc_auc(recommended_items, relevant_items)
#		precision_ += precision(recommended_items, relevant_items, at=at)
#		recall_ += recall(recommended_items, relevant_items, at=at)
#		map_ += map(recommended_items, relevant_items, at=at)
#		mrr_ += rr(recommended_items, relevant_items, at=at)
#		ndcg_ += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
#roc_auc_ /= neval
#precision_ /= neval
#recall_ /= neval
#map_ /= neval
#mrr_ /= neval
#ndcg_ /= neval
#
#logger.info('Ranking quality')
#logger.info('ROC-AUC: {:.4f}'.format(roc_auc_))
#logger.info('Precision@{}: {:.4f}'.format(at, precision_))
#logger.info('Recall@{}: {:.4f}'.format(at, recall_))
#logger.info('MAP@{}: {:.4f}'.format(at, map_))
#logger.info('MRR@{}: {:.4f}'.format(at, mrr_))
#logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_))
#










########################
########################
########################
########################
########################
########################
#nusers, nitems = len(idx_to_user), len(idx_to_item)
#logger.info('The dataset has {} users and {} items'.format(nusers, nitems))
#
## compute the holdout split
#logger.info('Computing the {:.0f}% holdout split'.format(args.holdout_perc*100))
#train_df, test_df = holdout_split(dataset, perc=args.holdout_perc, seed=args.rnd_seed)
#train = df_to_csr(train_df, nrows=nusers, ncols=nitems)
#test = df_to_csr(test_df, nrows=nusers, ncols=nitems)
#
## top-popular recommender
#logger.info('Building the top-popular recommender')
#recommender = TopPop()
#tic = dt.now()
#logger.info('Training started')
#recommender.fit(train)
#logger.info('Training completed built in {}'.format(dt.now() - tic))
#
## ranking quality evaluation
#roc_auc_, precision_, recall_, map_, mrr_, ndcg_ = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
#at = 20
#neval = 0
#for test_user in range(nusers):
#	user_profile = train[test_user].indices #what is doing here?
#	relevant_items = test[test_user].indices
#	if len(relevant_items) > 0:
#		neval += 1
#		#
#		# TODO: Here you can write to file the recommendations for each user in the test split. 
#		# WARNING: there is a catch with the item idx!
#		#
#		# this will rank *all* items
#		recommended_items = recommender.recommend(user_profile, exclude_seen=True)	
#		# use this to have the *top-k* recommended items (warning: this can underestimate ROC-AUC for small k)
#		# recommended_items = recommender.recommend(user_profile, k=at, exclude_seen=True)	
#		roc_auc_ += roc_auc(recommended_items, relevant_items)
#		precision_ += precision(recommended_items, relevant_items, at=at)
#		recall_ += recall(recommended_items, relevant_items, at=at)
#		map_ += map(recommended_items, relevant_items, at=at)
#		mrr_ += rr(recommended_items, relevant_items, at=at)
#		ndcg_ += ndcg(recommended_items, relevant_items, relevance=test[test_user].data, at=at)
#roc_auc_ /= neval
#precision_ /= neval
#recall_ /= neval
#map_ /= neval
#mrr_ /= neval
#ndcg_ /= neval
#
#logger.info('Ranking quality')
#logger.info('ROC-AUC: {:.4f}'.format(roc_auc_))
#logger.info('Precision@{}: {:.4f}'.format(at, precision_))
#logger.info('Recall@{}: {:.4f}'.format(at, recall_))
#logger.info('MAP@{}: {:.4f}'.format(at, map_))
#logger.info('MRR@{}: {:.4f}'.format(at, mrr_))
#logger.info('NDCG@{}: {:.4f}'.format(at, ndcg_))
#
#
#

