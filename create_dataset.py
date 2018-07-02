from helper_files import load_track
import sys
import numpy as np
import os


def collect_data(dataset_path, track_count):
    ''' returns:
    np.array (x, y, track_paths)
    -> x is a matrix containing extracted features
    -> y is a one-hot matrix of genre labels
    -> track_paths is a dict of absolute track paths indexed by row indices in the x and y matrices
    '''

    def_features = load_track("".join([dataset_path, 'blues/blues.00000.au']))
    default_shape = def_features.shape

    x = np.zeros((track_count,) + default_shape, dtype=np.float32)
    y = np.zeros((track_count, len(genres)), dtype=np.float32) # label
    track_paths = {}

    for (genre_index, genre_name) in enumerate(genres):
        for i in range(track_count // len(genres)):
            file_name = '{}/{}.000{}.au'.format(genre_name, genre_name, str(i).zfill(2))
            print ('Processing', file_name)
            path = "".join([dataset_path, file_name])
            track_index = genre_index  * (track_count // len(genres)) + i
            x[track_index] = load_track(path, default_shape)
            y[track_index, genre_index] = 1
            track_paths[track_index] = os.path.abspath(path)

    return (x, y, track_paths)


####################### PUT IN INFO ON DATASET ##############################
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
track_count = 1000 # size of Dataset (needs to be split evenly)
dataset_path = '/home/jan/Downloads/genres/'
data = collect_data(dataset_path, track_count)

# save data
save_destination = '/home/jan/Documents/uni/deep_learning/Project/data'
np.save(save_destination,data)
