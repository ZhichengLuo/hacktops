import numpy as np

def instance_center_norm(sample):
    center = int((len(sample)-1)/2)
    s = (sample - sample[center]) / (np.max(sample) - np.min(sample) + 1)
    return s

def instance_norm(sample: np.array):
    s = (sample - np.min(sample)) / (np.max(sample) - np.min(sample)+1)
    return s

