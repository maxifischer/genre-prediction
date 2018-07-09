from helper_files import load_track
import sys
import numpy as np
import os
import pandas as pd
import pickle

def get_features(labels):
    # Directory where mp3 are stored.
    audio_dir = "fma_small"

    X = pd.DataFrame(dtype=np.float32)
    track_paths = {}
    count = 0
    for track_index in labels.index.tolist():
        tid_str = '{:06d}'.format(track_index)
        path = os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')
        X = X.append(pd.DataFrame(load_track(path)))
        track_paths[track_index] = os.path.abspath(path)
        count += 1
        if count%10==0:
            print(count)
    return (X, track_paths)

def collect_data():
    ''' returns:
    np.array (x, y, track_paths)
    -> x is a matrix containing extracted features
    -> y is a one-hot matrix of genre labels
    -> track_paths is a dict of absolute track paths indexed by row indices in the x and y matrices
    '''
    
    # Load metadata and features.
    tracks = pd.read_csv('fma_metadata/tracks.csv', index_col=0, header=[0,1])
    
    # only take balanced 8000 tracks
    tracks_small = tracks[tracks[('set', 'subset')] == 'small']
    
    # labels already split in dataset (80-20 train-test split)
    y_train = tracks_small[tracks[('set', 'split')] == 'training'][('track', 'genre_top')]
    y_test = tracks_small[tracks[('set', 'split')] == 'test'][('track', 'genre_top')]
    
    X_train, train_track_paths = get_features(y_train)
    X_test, test_track_paths = get_features(y_test)
    track_paths = train_track_paths
    track_paths.update(test_track_paths)
    
    return (X_train, y_train, X_test, y_test, track_paths)


####################### PUT IN INFO ON DATASET ##############################
data = collect_data()

# save data
file = open('data/fma_dataset.pkl', "wb")
pickle.dump(data, file)
