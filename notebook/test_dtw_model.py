from hacktops.model import TopFinder
from tslearn.metrics import dtw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hacktops.data import generate_top_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from hacktops.utils import instance_norm
import random
import time
from datetime import datetime

random.seed(0)

top_ = 'CONRAD'
df_logs_ = pd.read_parquet("../data/logs.parquet")
df_loc_ = pd.read_parquet("../data/loc.parquet")
df_tops_ = pd.read_parquet("../data/tops.parquet")

all_well_names = df_logs_['wellName'].unique()
print('all well num:', len(all_well_names))

test_well_names = random.sample(all_well_names.tolist(), int(len(all_well_names) * 0.1))
train_well_names = [wn for wn in all_well_names.tolist() if wn not in test_well_names]

print('test_well_names num:', len(test_well_names))
print('train_well_names num:', len(train_well_names))

train_dataset = generate_top_dataset(df_logs=df_logs_[df_logs_['wellName'].isin(train_well_names)], df_tops=df_tops_, top=top_)
print(len(train_dataset[0]))

test_dataset = generate_top_dataset(df_logs=df_logs_[df_logs_['wellName'].isin(test_well_names)], df_tops=df_tops_, top=top_)
print(len(test_dataset[0]))

X = np.array(train_dataset[0]).squeeze(axis=2)
y = np.array(train_dataset[1])

X = np.array([instance_norm(x) for x in X])
    
print('X:', X.shape)
print('y:', y.shape)

# The model needs only true windows
train_dataset_true_idx = [idx for idx in range(len(X)) if y[idx] == True]
print('train_dataset_true_idx num:', len(train_dataset_true_idx))

plt.plot(X[train_dataset_true_idx[1]])


class WindowEvaluator:
    def __init__(self, metric=dtw):
        self.metric = metric

    def fit(self, X):
        self._X = X

    def predict_proba(self, X):
        res = []
        for x in X:
            dists = np.array([self.metric(_x, x) for _x in self._X])
            weights = 1/(1+dists)
            res.append(weights.sum()/weights.shape[0])
        return res

neigh = WindowEvaluator(metric=dtw)
neigh.fit(X[train_dataset_true_idx])

top_finder = TopFinder(neigh, top_)
neigh.evaluate_windows = neigh.predict_proba
top_finder.examine_dataset(df_tops_)

count = 1
for test_well_name in test_well_names:
    print(f'{count}/{len(test_well_names)} : {test_well_name}')
    count += 1
    df_test_well = df_logs_[df_logs_['wellName'] == test_well_name]
    windows = top_finder.get_candidate_windows(df_test_well)
    windows = [(w[0], np.array(instance_norm(w[1]))) for w in windows]

    print('\twindows candidate size:', len(windows))
    windows_depth = np.array([w[0] for w in windows])
    windows_data = np.array([w[1] for w in windows])
    
    true_depth = df_tops_.loc[test_well_name, top_]
    print('\ttrue depth: ', true_depth)
    temp = [w for w in windows if w[0]==true_depth]
    if len(temp) == 0 or np.isnan(true_depth):
        print('\tdid not find matched log of true depth, continue')
        continue
    
    start_time = datetime.now() 
    proba = neigh.predict_proba(windows_data)
    time_elapsed = datetime.now() - start_time 
    print('\tTime elapsed: {}'.format(time_elapsed))

    np.savetxt(f"./test_output/{str(test_well_name)}.txt", np.array(proba),fmt='%f',delimiter=',')
    # np_proba = np.loadtxt(f"./test_output/{str(test_well_name)}.txt",delimiter=',')
    print('\tdone')

print('end')
