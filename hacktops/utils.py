import numpy as np
from hacktops.data import generate_top_dataset
import plotly.graph_objs as go

##### 
# data normalization 
#####

def instance_center_norm(sample):
    center = int((len(sample)-1)/2)
    s = (sample - sample[center]) / (np.max(sample) - np.min(sample) + 1)
    return s

def instance_norm(sample: np.array):
    s = (sample - np.min(sample)) / (np.max(sample) - np.min(sample) + 1)
    return s


##### 
# dataset preparation 
#####

def get_true_windows(df_logs, df_tops, top_, keep_depth = False):
    dataset = generate_top_dataset(df_logs=df_logs, df_tops=df_tops, top=top_)
    all_well_names = df_logs['wellName'].unique()
    print(f'{len(dataset[0])} windows extracted from {len(all_well_names)} wells')

    X = np.array(dataset[0]).squeeze(axis=2)
    y = np.array(dataset[1])
    
    print('X:', X.shape)
    print('y:', y.shape)

    true_idx = [idx for idx in range(len(X)) if y[idx] == True]
    print(f'{len(true_idx)} true windows left')

    return X[true_idx]

def get_true_depth(wellname, top, df_tops):
    return df_tops.loc[wellname, top]


##### 
# visualization
#####

def visual_scores(depths, scores, max_score_depth=None, true_depth=None, well_name=None):
    data = []
    data.append(go.Scatter(x=depths,y=scores))
    title = "Evaluation Score w.r.t depth"
    if well_name:
        title += f' [well: {well_name}]'
    fig = go.Figure(data=data, layout={'title':title})
    if max_score_depth:
        fig.add_vline(x=max_score_depth, line_width=2, line_color="yellow", \
            annotation_text='Predicated', annotation_position='top left')
    if true_depth:
        fig.add_vline(x=true_depth, line_width=2, line_color="green", \
            annotation_text='True', annotation_position='top right')
    return fig