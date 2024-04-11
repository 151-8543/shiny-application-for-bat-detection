import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom, gaussian_filter1d
import spectrogram as sp
from scipy.io import wavfile
import nms_slow as nms
import tensorflow as tf
import keras
from keras import layers



class NeuralNet:

    def __init__(self, params_):
        self.params = params_
        self.model = None

    def train(self, positions, class_labels, files, durations):
        feats = []
        labs = []
        for ii, file_name in enumerate(files):
            if positions[ii].shape[0] > 0:
                local_feats = self.create_or_load_features(file_name)
                positions_ratio = positions[ii] / durations[ii]
                train_inds = (positions_ratio * float(local_feats.shape[0])).astype('int')
                feats.append(local_feats[train_inds, :, :, :])
                labs.append(class_labels[ii])

        features = np.vstack(feats)
        labels = np.vstack(labs).astype(np.uint8)[:, 0]
        print(('train size'), features.shape)

        input_shape = features.shape[1:]
        self.model = build_cnn(input_shape, self.params.net_type)

        self.model.compile(
            optimizer=keras.optimizers.legacy.SGD(learning_rate=self.params.learn_rate, momentum=self.params.moment),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model.fit(features, labels, epochs=self.params.num_epochs, batch_size=self.params.batchsize, shuffle=True)

    def test(self, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):
        features = self.create_or_load_features(file_name, audio_samples, sampling_rate)
        y_prediction = self.model.predict(features)[:, 1]

        if self.params.smooth_op_prediction:
            y_prediction = gaussian_filter1d(y_prediction, self.params.smooth_op_prediction_sigma, axis=0)

        pos, prob = nms.nms_1d(y_prediction.astype(np.float32), self.params.nms_win_size, file_duration)

        return pos, prob, y_prediction

    def create_or_load_features(self, file_name=None, audio_samples=None, sampling_rate=None):
        if file_name is None:
            features = compute_features(audio_samples, sampling_rate, self.params)
        else:
            if self.params.load_features_from_file:
                features = np.load(self.params.feature_dir + file_name + '.npy')
            else:
                sampling_rate, audio_samples = wavfile.read(self.params.audio_dir + file_name.decode('utf-8') + '.wav') #statt file_name
                features = compute_features(audio_samples, sampling_rate, self.params)

        return features

    def save_features(self, files):
        for file_name in files:
            sampling_rate, audio_samples = wavfile.read(self.params.audio_dir + file_name.decode('utf-8') + '.wav')
            features = compute_features(audio_samples, sampling_rate, self.params)
            np.save(self.params.feature_dir + file_name, features)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def build_cnn(input_shape, net_type):
    model = tf.keras.Sequential()

    if net_type == 'big':
        model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
    elif net_type == 'small':
        model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
    else:
        raise ValueError('Error: network not defined')

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2, activation='softmax'))

    return model


def compute_features(audio_samples, sampling_rate, params):
    spectrogram = sp.gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)
    spectrogram = sp.process_spectrogram(spectrogram, denoise_spec=params.denoise, mean_log_mag=params.mean_log_mag,
                                         smooth_spec=params.smooth_spec)

    spec_win = view_as_windows(spectrogram, (spectrogram.shape[0], params.window_width))[0]
    spec_win = zoom(spec_win, (1, 0.5, 0.5), order=1)
    spec_width = spectrogram.shape[1]

    features = np.zeros((spec_width, 1, spec_win.shape[1], spec_win.shape[2]), dtype=np.float32)
    features[:spec_win.shape[0], 0, :, :] = spec_win

    return features
