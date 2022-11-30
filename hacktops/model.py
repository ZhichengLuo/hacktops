import findspark
findspark.init('/opt/multi/spark-3.0.1-bin-hadoop2.7')

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw as tslearn_dtw
from hacktops.settings import WINDOW_LENGTH, DILATION_RATIO
from hacktops.utils import instance_norm
from dtaidistance.dtw import distance as dtai_dtw
from pyspark import SparkContext
from tqdm import tqdm

class TopFinder:
    """
    TopFinder: wrapper for window classifier
    
    Limitations:
    - Work on single one top and assume independence among tops
    - Find top by classifying windows extracted from well data and discard
      the correlation between windows
    - Does not utilize geographical info of wells

    Usage example:

        >>> model.fit(dataset)
        >>> model.evaluate_windows = a_func

        >>> top_finder = TopFinder(model, top_name)
        >>> top_finder.examine_dataset(df_tops)

        >>> predicted_depth = top_finder.find_top(df_well)

    """

    def __init__(self, fitted_window_classifier, top_name):
        if fitted_window_classifier.evaluate_windows is None:
            raise ValueError("fitted_window_classifier has to have function evaluate_windows")
        self.window_classifier = fitted_window_classifier
        self.work_on_top = top_name
        self.stats = {}

    def examine_dataset(self, df_tops:pd.DataFrame):
        self.stats['top_depth_max'] = df_tops[self.work_on_top].max()
        self.stats['top_depth_min'] = df_tops[self.work_on_top].min()

    def extract_window(self, df_well:pd.DataFrame, center_idx, window_length):
        left_limit = center_idx - window_length
        right_limit = center_idx + window_length
        window = df_well.loc[left_limit : right_limit, 'GR'].to_numpy()
        return window

    def get_candidate_windows(self, df_well:pd.DataFrame):
        '''
        extra prior knowledge may be used to narrow down the scope of candidates, 
        e.g. top distribution. 

        return list of windows. Each window includes the depth of its center & GR data.
        '''
        max_, min_ = self.stats['top_depth_max'], self.stats['top_depth_min']
        center_  = (max_ + min_) / 2
        depth_diff_ = max_ - min_
        dilated_max_ =  center_ + DILATION_RATIO * depth_diff_ / 2
        dilated_min_ =  center_ - DILATION_RATIO * depth_diff_ / 2

        windows = []
        for idx, row in df_well.iterrows():
            if row['DEPTH'] < dilated_max_ and row['DEPTH'] > dilated_min_:
                window_depth = row['DEPTH']
                window_data = self.extract_window(df_well, idx, WINDOW_LENGTH)
                if window_data.shape != (WINDOW_LENGTH * 2 + 1,):
                    # print(window_data.shape) 
                    # It happens when the window gets out of the scope of well depth
                    continue
                windows.append((window_depth, window_data))
        return windows

    def select_window(self, windows, scores: np.array):
        '''
        extra prior knowledge may be used here, e.g. top relationships
        '''
        index_max = np.argmax(scores, axis=0)
        return windows[index_max]

    def find_top(self, df_well):
        """
        Step:
            1. Extract all candidate windows from the well
            2. Evalute each candidate by window classifier
            3. Select the best candidate
            4. Return its associated depth
        """
        if self.window_classifier is None:
            raise Exception("window_classifier is not set")
        if df_well.shape[0] == 0:
            raise Exception("input well has no data")

        self.windows = self.get_candidate_windows(df_well)
        print(f'{len(self.windows)} candidate windows')
        windows_data = np.array([w[1] for w in self.windows])
        self.scores = self.window_classifier.evaluate_windows(windows_data)
        selected_window = self.select_window(self.windows, self.scores)
        self.top_depth = selected_window[0]

        return self.top_depth

class SimpleDTWWindowEvaluator:
    """
    Simple DTW Window Evaluator

    Fit: Caches a set of real top windows

    Evaluate: Scores = [AVG(1 / 1 + dtw(candidate, real)) for each candidate]
    """
    def __init__(self, metric=tslearn_dtw, norm=instance_norm):
        self.metric = metric
        self.norm = norm

    def fit(self, real_top_windows):
        real_top_windows = np.array([self.norm(w) for w in real_top_windows])
        self._real_top_windows = real_top_windows

    def evaluate_windows(self, candidate_windows) -> np.array:
        scores = []
        candidate_windows = [self.norm(w) for w in candidate_windows]
        for w in tqdm(candidate_windows):
            dists = np.array([self.metric(_w, w) for _w in self._real_top_windows])
            weights = 1 / (1 + dists)
            scores.append(weights.sum() / weights.shape[0])
        scores = np.array(scores)
        return scores

class SimpleDTWWindowEvaluator_Spark:
    """
    Spark Version of Simple DTW Window Evaluator
    """
    def __init__(self, sc:SparkContext, metric=tslearn_dtw, norm=instance_norm):
        self.metric = metric
        self.sc = sc
        self.norm = norm

    def fit(self, real_top_windows):
        real_top_windows = np.array([self.norm(w) for w in real_top_windows])
        self._real_top_windows_rdd = self.sc.parallelize(real_top_windows)

    def evaluate_windows(self, candidate_windows):
        candidate_windows = [self.norm(w) for w in candidate_windows]
        input_rdd = self.sc.parallelize([(i, candidate_windows[i]) for i in range(len(candidate_windows))])
        aTuple = (0,0)
        metric = self.metric
        scores = input_rdd.cartesian(self._real_top_windows_rdd)\
                .map(lambda t : (t[0][0], 1 / (1 + metric(t[0][1], t[1]))))\
                .aggregateByKey(aTuple, lambda a,b: (a[0] + b,    a[1] + 1),
                                        lambda a,b: (a[0] + b[0], a[1] + b[1]))\
                .mapValues(lambda v: v[0] / v[1])\
                .sortByKey().map(lambda t : t[1])\
                .collect()
        return scores


# class WindowClassifier_KNN:
#     def __init__(self, **kwargs):
#         self.model = KNeighborsClassifier(**kwargs)

#     def fit(self, **kwargs):
#         self.model.fit(**kwargs)

#     def evaluate_windows(self, windows_data):
#         scores = self.model.predict_proba(windows_data)
#         return scores


# if __name__ == '__main__':
#     baseline_model = WindowClassifier_KNN(n_neighbors=3, weights='distance', metric=dtw, n_jobs=4)
#     top_finder = TopFinder(baseline_model)
#     dataset = None
#     top_finder.fit(dataset)
#     df_well = None
#     top_name = 'MARCEL'
#     depth = top_finder.find_top(df_well, top_name)
#     print(f'Predicated depth of {top_name}: {depth}')

