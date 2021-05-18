"""
  Train a LightGBM model on the entire descriptor data, using the same meta parameters as AFEC's
  crawler. 

  Dependencies (pip): numpy, sklearn, pandas, matplotlib, lightgbm 
"""

import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from utils import parse_args, load_dataset, split_augmented_samples

# -------------------------------------------------------------------------------------------------
# parse args

args = parse_args()

# -------------------------------------------------------------------------------------------------
# options

DUMP_FEATURE_IMPORTANCE = False
DUMP_FEATURE_IMPORTANCE_SORTED = False

# -------------------------------------------------------------------------------------------------
# load dataset

df_labels, df_ids, df_data, num_classes, feature_names = load_dataset(
    os.path.join(args['classifier_root_path'], 'afec-ll.csv'))

# randomize and split datasets
sss = None
if (args['number_of_runs'] == 1):
	sss = StratifiedShuffleSplit(n_splits=1, test_size=1/5)
else:
	sss = StratifiedKFold(n_splits=args['number_of_runs'], shuffle=True)

lgb_train_sets = []
lgb_test_sets = []
for train_index, test_index in sss.split(df_data, df_labels):
	# deal with augmented samples
	train_index, test_index = split_augmented_samples(train_index, test_index, df_ids)

	# fetch pandas rows for current split
	X_train, X_test = df_data.iloc[train_index], df_data.iloc[test_index]
	y_train, y_test = df_labels.iloc[train_index], df_labels.iloc[test_index]

	# convert to lightgbm datasets
	lgb_train = lgb.Dataset(X_train, y_train)
	lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
	lgb_train_sets.append(lgb_train)
	lgb_test_sets.append(lgb_test)

# -------------------------------------------------------------------------------------------------
# train

print('Start training...')

params = {}
if (args['classifier_name'] == "OneShot-Categories"):
	params = {
		'tree_learner': 'feature',
		'objective': 'multiclass_ova',
		'metric': 'multi_error',
		'boosting': 'gbdt',
		'num_class': num_classes,
		'max_bin': 32,
		'max_depth': 6,
		'num_leaves': 6,
		'min_data_in_bin': 12,
		'min_data_in_leaf': 32,
		'bagging_freq': 1,
		'bagging_fraction': 0.5,
		'feature_fraction': 0.25,
		'learning_rate': 0.06,
		'seed': args['random_seed'],
		'verbose': -1,
	}
elif (args['classifier_name'] == "OneShot-vs-Loops"):
	params = {
		'tree_learner': 'feature',
		'objective': 'multiclass',
		'metric': 'multi_error',
		'boosting': 'gbdt',
		'num_class': num_classes,
		'max_bin': 32,
		'max_depth': 8,
		'num_leaves': 3,
		'min_data_in_bin': 12,
		'min_data_in_leaf': 32,
		'bagging_freq': 0,
		'learning_rate': 0.06,
		'seed': args['random_seed'],
		'verbose': -1,
	}
elif (args['classifier_name'] == "ESC-50"):
	params = {
		'tree_learner': 'data',
		'objective': 'multiclass_ova',
		'metric': 'multi_error',
		'boosting': 'gbdt',
		'num_class': num_classes,
		'max_bin': 32,
		'max_depth': 6,
		'num_leaves': 8,
		'min_data_in_bin': 12,
		'min_data_in_leaf': 12,
		'min_split_gain': 0.75,
		'bagging_freq': 1,
		'bagging_fraction': 0.5,
		'feature_fraction': 0.5,
		'learning_rate': 0.06,
		'seed': args['random_seed'],
		'verbose': -1,
	}
else:
	raise Exception("Unknown classifier to train: '%s'" % args['classifier_name'])

validation_accuracies = []

for i in range(len(lgb_train_sets)):
	print("\n*** Training model %s of %s..." % (i + 1, len(lgb_train_sets)))
	print("Train on %s samples, validate on %s samples" %  
		(len(lgb_train_sets[i].data), len(lgb_test_sets[i].data)))

	# train
	results = {}
	gbm = lgb.train(
		params=params,
		train_set=lgb_train_sets[i],
		num_boost_round=1200,
		early_stopping_rounds=500,
		# NB: don't pass/dump lgb_train_sets[i] here force early stopping on 'valid'
		valid_sets=[lgb_test_sets[i]],
		valid_names=["valid"],
		feature_name=feature_names,
		evals_result=results,
		verbose_eval=10)
	
	# evaluate current results
	validation_errors = results['valid']['multi_error']
	best_iteration = gbm.best_iteration - 1 # starts from 1
	accuracy = (1 - validation_errors[best_iteration]) * 100
	validation_accuracies.append(accuracy)
	print("-> Accuracy: %.2f%%" % accuracy)

# -------------------------------------------------------------------------------------------------
# evaluate all results

print("\nResults: %s" % validation_accuracies)
print("-> Mean Accuracy: %.2f%% +- %.2f%%" % 
	(np.mean(validation_accuracies), np.std(validation_accuracies)))

if DUMP_FEATURE_IMPORTANCE:
	print('Feature importance:')
	
	feature_importance = gbm.feature_importance()
	feature_importance_map = {}
	for i in range(len(feature_importance)):
	  feature_importance_map[feature_names[i]] = feature_importance[i]
	feature_importance_results = feature_importance_map.items()
	if DUMP_FEATURE_IMPORTANCE_SORTED:
	  feature_importance_results = sorted(
		  feature_importance_map.items(), key=lambda kv: kv[1], reverse=True) 
	for key, value in feature_importance_results:
		print("%s: %s" % (key, value))
