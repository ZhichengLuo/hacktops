import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw
from hacktops.settings import WINDOW_LENGTH, DILATION_RATIO

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

    def __init__(self, window_classifier, top_name):
        self.window_classifier = window_classifier
        self.work_on_top = top_name
        self.stats = {}

    def examine_dataset(self, df_tops:pd.DataFrame):
        self.stats['top_depth_max'] = df_tops[self.work_on_top].max()
        self.stats['top_depth_min'] = df_tops[self.work_on_top].min()

    def fit(self, **kwargs):
        self.window_classifier.fit(**kwargs)

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
                    print(window_data.shape) # it should never happen
                    continue
                windows.append((window_depth, window_data))
        return windows

    def evaluate_windows(self, windows):
        scores = self.window_classifier.evaluate_windows(windows)
        return scores

    def select_window(self, windows, scores: np.array):
        '''
        extra prior knowledge may be used here, e.g. top relationships
        '''
        index_max = np.argmax(scores, axis=0)[1]
        return windows[index_max]

    def find_top(self, df_well):

        if self.window_classifier is None:
            raise Exception("window_classifier is not set")
        if df_well.shape[0] == 0:
            raise Exception("input well has no data")

        windows = self.get_candidate_windows(df_well)
        print(f'{len(windows)} candidate windows')
        windows_data = np.array([w[1] for w in windows])
        scores = self.window_classifier.evaluate_windows(windows_data)
        print(f'scores: {scores}')
        selected_window = self.select_window(windows, scores)
        top_depth = selected_window[0]

        return top_depth

class WindowClassifier_KNN:
    def __init__(self, **kwargs):
        self.model = KNeighborsClassifier(**kwargs)

    def fit(self, **kwargs):
        self.model.fit(**kwargs)

    def evaluate_windows(self, windows_data):
        scores = self.model.predict_proba(windows_data)
        return scores


if __name__ == '__main__':
    baseline_model = WindowClassifier_KNN(n_neighbors=3, weights='distance', metric=dtw, n_jobs=4)
    top_finder = TopFinder(baseline_model)
    dataset = None
    top_finder.fit(dataset)
    df_well = None
    top_name = 'MARCEL'
    depth = top_finder.find_top(df_well, top_name)
    print(f'Predicated depth of {top_name}: {depth}')

