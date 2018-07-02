import matplotlib
matplotlib.use('Agg')
import numpy as np
import librosa as lbr


############## SETTINGS FOR MEL ####################################
window_size = 2048 # smallest audible period (ca. 10ms at 22050HZ/2048)
window_stride = window_size // 2
n_mels = 128

mel_kwargs = {
    'n_fft': window_size,
    'hop_length': window_stride,
    'n_mels': n_mels
}
####################################################################




def load_track(filename, default_shape=None):
    new_input, sample_rate = lbr.load(filename, mono=True)
    features = lbr.feature.melspectrogram(new_input, **mel_kwargs).T # create mel spectogram

    # make sure the shape is the same for all recordings
    if default_shape is not None:
        if features.shape[0] < default_shape[0]:
            delta_shape = (default_shape[0] - features.shape[0], default_shape[1])
            features = np.append(features, np.zeros(delta_shape), axis=0) # if too short, add zeros
        elif features.shape[0] > default_shape[0]:
            features = features[: default_shape[0], :] # if too long, leave out the end
    features[features == 0] = 1e-6 # to avoid errors with log
    return np.log(features)



def pick_batch(sample_size, batch_size):
    idxs = np.arange(sample_size) # pick idxs
    np.random.shuffle(idxs) # shuffle the order
    n = 0

    while True:
        if batch_size*(n+1) > sample_size:
            np.random.shuffle(idxs) # reshuffle the order
            n=0
        else:
            yield idxs[(batch_size*n): (batch_size*(n+1))]
            n+=1
