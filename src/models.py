# src/models.py

import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger("PROTAC_Pipeline.trainer")

import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger("PROTAC_Pipeline.trainer")

def get_xgb_model(params=None):
    default_params = {'booster': 'gbtree', 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42}
    if params: default_params.update(params)
    return XGBClassifier(**default_params)

def get_rf_model(params=None):
    default_params = {'n_estimators': 500, 'max_depth': 15, 'class_weight': 'balanced_subsample', 'n_jobs': -1, 'random_state': 42}
    if params: default_params.update(params)
    return RandomForestClassifier(**default_params)

def get_svm_model(params=None):
    default_params = {'kernel': 'poly', 'probability': True, 'class_weight': 'balanced', 'cache_size': 4000, 'random_state': 42}
    if params: default_params.update(params)
    return SVC(**default_params)

def get_knn_model(params=None):
    default_params = {'n_neighbors': 15, 'weights': 'distance', 'metric': 'cosine', 'n_jobs': -1}
    if params: default_params.update(params)
    return KNeighborsClassifier(**default_params)