import numpy as np
import pandas as pd

from typing import List
from typing import Union
from typing import Tuple
from hacktops.settings import NB_SAMPLES
from hacktops.settings import SHIFT_STEP
from hacktops.settings import WINDOW_LENGTH


def get_well_relevant_windows(top_index: int, df_well: pd.DataFrame, nb_samples: int=NB_SAMPLES,
                              shift: int=SHIFT_STEP, ratio: Union[None, float]=None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Given df_well : 'wellName', 'DEPTH', 'GR' and top_index the position of a top in df_well
    Returns a list of numerous windows around top_index, and their label
    Labels are either True or False
    for a given selected window it is labelled True if the distance between its center and
    the top position is less than 4

    :param top_index: int
    :param df_well: pd.DataFrame(columns=['wellName', 'DEPTH', 'GR'])
    :param nb_samples: int
    :param shift: int=SHIFT_STEP
    :param ratio: Union[None, float]=None
    :return: list
    """
    windows = []
    labels = []
    positives = 0
    negatives = 0
    for i in range(top_index - nb_samples, top_index + nb_samples, shift):
        left_limit = i - WINDOW_LENGTH
        right_limit = i + WINDOW_LENGTH
        window_data = list(map(lambda x: np.array([x]), list(df_well['GR'].values[left_limit:right_limit + 1])))
        if np.array(window_data).shape != (WINDOW_LENGTH * 2 + 1, 1):
            continue
        label = abs(df_well['DEPTH'].iloc[i] - df_well['DEPTH'].iloc[top_index]) < 4
        if ratio:
            if label:
                windows.append(np.array(window_data))
                labels.append(np.array(label))
            elif negatives / max(positives + negatives, 1) < ratio:
                pass
            else:
                windows.append(np.array(window_data))
                labels.append(np.array(label))
        else:
            windows.append(np.array(window_data))
            labels.append(np.array(label))

        if label:
            positives += 1
        else:
            negatives += 1

    return windows, labels


# TODO: check index and len depth

def generate_top_dataset(df_logs: pd.DataFrame, df_tops: pd.DataFrame,
                         top: str='CONRAD', ratio: Union[None, float]=None):
    """
    From df_logs and df_tops for each well
    return a list of relevant windows from the whole signal of the well and the labels of the windows
    a relevant window depend on the top
    for more explanation about window selection cf get_well_relevant_windows

    df_logs contains : 'wellName', 'DEPTH', 'GR'
    df_tops contains a column top

    :param df_logs: pd.DataFrame
    :param df_tops: pd.DataFrame
    :param top: str='CONRAD'
    :param ratio: Union[None, float]=None
    :return:
    """
    windows = []
    labels = []
    for well_name in df_logs['wellName'].drop_duplicates().tolist():
        df_well = df_logs[df_logs['wellName'] == well_name]
        top_position = df_tops.loc[well_name][top]
        if np.isnan(top_position):
            print("NAN FOUND")
            continue
        depth_list = list(df_well['DEPTH'].values)
        real_top_position = min(df_well['DEPTH'].values,
                                key=lambda x: abs(x - top_position))  # SOMETIMES top_position not in df_logs
        if abs(real_top_position - top_position) > 3:
            print("DATA BAD LABELLED")
            continue
        top_index = depth_list.index(real_top_position)
        windows_, labels_ = get_well_relevant_windows(top_index=top_index, df_well=df_well, shift=1, nb_samples=1 * 100,
                                                      ratio=ratio)
        windows += windows_
        labels += labels_
    return windows, labels


def get_location_dataset(df_loc: pd.DataFrame, df_tops: pd.DataFrame, top: str):
    """
    result :
     - index : wellName
     - columns : Latitude, Longitude, top
    top must be a column of df_tops

    :param df_loc: pd.DataFrame
    :param df_tops: pd.DataFrame
    :param top: str
    :return:
    """
    assert top in df_tops
    well_data = df_loc.merge(df_tops[[top]], how='inner', left_index=True, right_index=True)
    well_data = well_data[well_data[top].notnull()]

    return well_data.reset_index(drop=True)


if __name__ == '__main__':
    top_ = 'CONRAD'
    df_logs_ = pd.read_parquet("../data/logs.parquet")
    df_loc_ = pd.read_parquet("../data/loc.parquet")
    df_tops_ = pd.read_parquet("../data/tops.parquet")

    a = generate_top_dataset(df_logs=df_logs_, df_tops=df_tops_, top=top_)
    print(1)

