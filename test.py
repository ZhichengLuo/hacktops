import numpy as np
import pandas as pd

top = "CONRAD"
SHIFT_STEP = 3
NB_SAMPLES = SHIFT_STEP * 100
WINDOW_LENGTH = 30


def get_window(index_: int, window_len: int=30):
    window = [i for i in range(index_ - window_len, index_ + window_len + 1)]
    return window


def get_dataset_windows(top_index: int, df_well: pd.DataFrame):
    windows = []
    labels = []
    for i in range(top_index - NB_SAMPLES, top_index + NB_SAMPLES, SHIFT_STEP):
        left_limit = i - WINDOW_LENGTH
        right_limit = i + WINDOW_LENGTH
        window_data = df_well['GR'].values[left_limit:right_limit + 1]
        print(len(window_data))
        print(len(window_data))
        label = 1 if i == top_index else 0
        if len(window_data) != 61:
            pass
            # print(1)
        windows.append(window_data)
        labels.append(label)
    return windows, labels


if __name__ == '__main__':
    df_logs = pd.read_parquet("C:/Users/User/Documents/CentraleSupelec/3A/hacktops/data/logs.parquet")
    df_loc = pd.read_parquet("C:/Users/User/Documents/CentraleSupelec/3A/hacktops/data/loc.parquet")
    df_tops = pd.read_parquet("C:/Users/User/Documents/CentraleSupelec/3A/hacktops/data/tops.parquet")
    df_logs = df_logs[df_logs['GR'] > 0]
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
        windows_, labels_ = get_dataset_windows(top_index=top_index, df_well=df_well)
        # print(len(windows_))
        if len(windows_) < 61:
            break

    print(1)