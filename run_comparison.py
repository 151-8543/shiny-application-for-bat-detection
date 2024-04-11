import numpy as np
import matplotlib.pyplot as plt
import os
import evaluate as evl
import create_results as res
from data_set_params import DataSetParams
import classifier as clss
import pandas as pd
import pickle5 as pickle


if __name__ == '__main__':

    test_set = 'bulgaria'  # can be one of: bulgaria, uk, norfolk, x
    data_set = 'data/train_test_split/test_set_' + test_set + '.npz'
    raw_audio_dir = 'data/wav/'
    result_dir = 'results/'
    model_dir = 'data/models/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print(('test set:') + test_set)
    plt.close('all')

    # train and test_pos are in units of seconds
    loaded_data_tr = np.load(data_set, allow_pickle=True, encoding='latin1') #allow_pickle=True, encoding='latin1'
    train_pos = loaded_data_tr['train_pos']
    train_files = loaded_data_tr['train_files']
    train_durations = loaded_data_tr['train_durations']
    test_pos = loaded_data_tr['test_pos']
    test_files = loaded_data_tr['test_files']
    test_durations = loaded_data_tr['test_durations']

    # load parameters
    params = DataSetParams()
    params.audio_dir = raw_audio_dir

    #
    # CNN
    print ('\ncnn')
    params.classification_model = 'cnn'
    model = clss.Classifier(params)
    # train and test
    model.train(train_files, train_pos, train_durations)
    nms_pos, nms_prob = model.test_batch(test_files, test_pos, test_durations, False, '')
    # compute precision recall
    precision, recall = evl.prec_recall_1d(nms_pos, nms_prob, test_pos, test_durations, model.params.detection_overlap, model.params.window_size)
    res.plot_prec_recall('cnn', recall, precision, result_dir, nms_prob)
    #plot epochs
    res.plot_epochs(model, result_dir)
    # save CNN model to file
    pickle.dump(model, open(model_dir + 'test_set_' + test_set + '.mod', 'wb'))

    
    # save results
    plt.savefig(result_dir + test_set + '_results.png')
    plt.savefig(result_dir + test_set + '_results.pdf')


