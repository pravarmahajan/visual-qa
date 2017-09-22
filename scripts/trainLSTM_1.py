import numpy as np
import scipy.io
import sys
import argparse

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout, Reshape, Merge
from keras.layers import LSTM, Bidirectional, Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint, RemoteMonitor

from sklearn.externals import joblib
from sklearn import preprocessing

from utils import grouper, selectFrequentAnswers
from features import get_images_matrix, get_answers_matrix, get_embeddings, get_categorical

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-num_hidden_units_mlp', type=int, default=1024)
    parser.add_argument('-num_hidden_units_lstm', type=int, default=512)
    parser.add_argument('-num_hidden_layers_mlp', type=int, default=3)
    parser.add_argument('-num_hidden_layers_lstm', type=int, default=1)
    parser.add_argument('-dropout', type=float, default=0.5)
    parser.add_argument('-activation_mlp', type=str, default='tanh')
    parser.add_argument('-num_epochs', type=int, default=100)
    parser.add_argument('-model_save_interval', type=int, default=5)
    parser.add_argument('-batch_size', type=int, default=128)
    # TODO Feature parser.add_argument('-resume_training', type=str)
    # TODO Feature parser.add_argument('-language_only', type=bool, default= False)
    args = parser.parse_args()

    word_vec_dim = 300
    img_dim = 4096
    img_out_dim = 1024
    max_len = 15
    nb_classes = 1000
    max_answers = nb_classes
    # get the data
    questions_train = open('../data/preprocessed/questions_sample2014.txt',
                           'r').read().decode('utf-8').splitlines()
    questions_lengths_train = open(
        '../data/preprocessed/questions_lengths_sample2014.txt', 'r').read().decode('utf8').splitlines()
    answers_train = open('../data/preprocessed/answers_sample2014_modal.txt',
                         'r').read().decode('utf8').splitlines()
    images_train = open('../data/preprocessed/images_sample2014.txt',
                        'r').read().decode('utf8').splitlines()
    vgg_model_path = '../features/coco/vgg_feats.mat'

    questions_train, answers_train, images_train = selectFrequentAnswers(
        questions_train, answers_train, images_train, max_answers)
    questions_lengths_train, questions_train, answers_train, images_train = (list(t) for t in zip(
        *sorted(zip(questions_lengths_train, questions_train, answers_train, images_train))))

    word_to_id, embedding_matrix = get_embeddings('../data/glove.6B.300d.txt', word_vec_dim)
    print 'loaded word2vec features...'

    # encode the remaining answers
    labelencoder = preprocessing.LabelEncoder()
    labelencoder.fit(answers_train)
    joblib.dump(labelencoder, '../models/labelencoder.pkl')

    image_model = Sequential()
    image_model.add(Dense(img_out_dim, input_shape=(img_dim,),
                          activation=args.activation_mlp))
    image_model.add(Dropout(args.dropout))

    language_model = Sequential()
    language_model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                 trainable=False,
                                 weights=[embedding_matrix],
                                 input_length=max_len))

    language_model.add(Conv1D(15, 3, strides=1, padding="same"))
    if args.num_hidden_layers_lstm == 1:
        language_model.add(Bidirectional(LSTM(args.num_hidden_units_lstm,
                                return_sequences=False,
                                unroll=True)))
        language_model.add(Dropout(args.dropout))
    else:
        language_model.add(Bidirectional(LSTM(args.num_hidden_units_lstm,
                                            return_sequences=True,
                                            unroll=True)))
        language_model.add(Dropout(args.dropout))

        for i in xrange(args.num_hidden_layers_lstm - 2):
            language_model.add(Bidirectional(LSTM(args.num_hidden_units_lstm,
                                    return_sequences=True, unroll=True)))
            language_model.add(Dropout(args.dropout))
            language_model.add(Dropout(args.dropout))

        language_model.add(Bidirectional(LSTM(args.num_hidden_units_lstm,
                                return_sequences=False, unroll=True)))
        language_model.add(Dropout(args.dropout))

    model = Sequential()
    model.add(Merge([language_model, image_model], mode='mul',
                    concat_axis=1))
    for i in xrange(args.num_hidden_layers_mlp):
        model.add(Dense(args.num_hidden_units_mlp,
                        kernel_initializer='uniform'))
        model.add(Activation(args.activation_mlp))
        model.add(Dropout(args.dropout))

    model.add(Dense(nb_classes, activation='softmax'))

    json_string = model.to_json()
    model_file_name = '../models/lstm_1_num_hidden_units_lstm_' + str(args.num_hidden_units_lstm) + \
        '_num_hidden_units_mlp_' + str(args.num_hidden_units_mlp) + '_num_hidden_layers_mlp_' + \
        str(args.num_hidden_layers_mlp) + \
        '_num_hidden_layers_lstm_' + str(args.num_hidden_layers_lstm)
    open(model_file_name + '.json', 'w').write(json_string)

    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    print 'Compilation done'

    features_struct = scipy.io.loadmat(vgg_model_path)
    VGGfeatures = features_struct['feats']
    print 'loaded vgg features'
    image_ids = open('../features/coco_vgg_IDMap.txt').read().splitlines()
    img_map = {}
    for ids in image_ids:
        id_split = ids.split()
        img_map[id_split[0]] = int(id_split[1])

    # training
    print 'Training started...'
    for k in xrange(args.num_epochs):

        progbar = generic_utils.Progbar(len(questions_train))

        for qu_batch, an_batch, im_batch in zip(grouper(questions_train, args.batch_size, fillvalue=questions_train[-1]),
                                                grouper(answers_train, args.batch_size, fillvalue=answers_train[-1]),
                                                grouper(images_train, args.batch_size, fillvalue=images_train[-1])):
            #X_q_batch = get_questions_tensor_timeseries(qu_batch, nlp, max_len)
            X_q_batch = get_categorical(qu_batch, word_to_id, max_len)
            X_i_batch = get_images_matrix(im_batch, img_map, VGGfeatures)
            Y_batch = get_answers_matrix(an_batch, labelencoder)
                                         
            loss = model.train_on_batch([X_q_batch, X_i_batch], Y_batch)
            progbar.add(args.batch_size, values=[("train loss", loss)])

    model.save_weights(model_file_name + '_epoch_{:03d}.hdf5'.format(k + 1))


if __name__ == "__main__":
    main()
