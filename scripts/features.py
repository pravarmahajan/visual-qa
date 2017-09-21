import numpy as np
from keras.utils import np_utils


def get_questions_tensor_timeseries(questions, nlp, timesteps):
    '''
    Returns a time series of word vectors for tokens in the question

    Input:
    questions: list of unicode objects
    nlp: an instance of the class English() from spacy.en
    timesteps: the number of 

    Output:
    A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
    '''
    assert not isinstance(questions, basestring)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
    for i in xrange(len(questions)):
        tokens = nlp(questions[i])
        for j in xrange(len(tokens)):
            if j < timesteps:
                questions_tensor[i, j, :] = tokens[j].vector

    return questions_tensor


def get_questions_matrix_sum(questions, nlp):
    '''
    Sums the word vectors of all the tokens in a question

    Input:
    questions: list of unicode objects
    nlp: an instance of the class English() from spacy.en

    Output:
    A numpy array of shape: (nb_samples, word_vec_dim)	
    '''
    assert not isinstance(questions, basestring)
    nb_samples = len(questions)
    word_vec_dim = nlp(questions[0])[0].vector.shape[0]
    questions_matrix = np.zeros((nb_samples, word_vec_dim))
    for i in xrange(len(questions)):
        tokens = nlp(questions[i])
        for j in xrange(len(tokens)):
            questions_matrix[i, :] += tokens[j].vector

    return questions_matrix


def get_answers_matrix(answers, encoder):
    '''
    Converts string objects to class labels

    Input:
    answers:	a list of unicode objects
    encoder:	a scikit-learn LabelEncoder object

    Output:
    A numpy array of shape (nb_samples, nb_classes)
    '''
    assert not isinstance(answers, basestring)
    y = encoder.transform(answers)  # string to numerical class
    nb_classes = encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_classes)
    return Y


def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
    '''
    Gets the 4096-dimensional CNN features for the given COCO
    images

    Input:
    img_coco_ids: 	A list of strings, each string corresponding to
                                    the MS COCO Id of the relevant image
    img_map: 		A dictionary that maps the COCO Ids to their indexes 
                                    in the pre-computed VGG features matrix
    VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)

    Ouput:
    A numpy matrix of size (nb_samples, nb_dimensions)
    '''
    assert not isinstance(img_coco_ids, basestring)
    nb_samples = len(img_coco_ids)
    nb_dimensions = VGGfeatures.shape[0]
    image_matrix = np.zeros((nb_samples, nb_dimensions))
    for j in xrange(len(img_coco_ids)):
        image_matrix[j, :] = VGGfeatures[:, img_map[img_coco_ids[j]]]

    return image_matrix


def get_embeddings(embedding_file, embedding_dim):
    lines = open(embedding_file).readlines()
    words = {}
    vectors = np.zeros((len(lines) + 1, embedding_dim))
    for line_num, line in enumerate(lines):
        parts = line.split()
        words[parts[0]] = line_num + 1
        vectors[line_num + 1] = parts[1:]
    return words, vectors


def get_categorical(sentences, word_to_id, max_len):
    sentence_ids = np.empty((len(sentences), max_len))
    for i, sentence in enumerate(sentences):
        words = sentence.split()
        word_ids = [None] * len(words)
        for j, w in enumerate(words):
            lower = w.lower()
            if lower in word_to_id:
                word_ids[j] = word_to_id[lower]
            else:
                word_ids[j] = word_to_id['<unk>']
        if len(word_ids) < max_len:
            word_ids = word_ids + [0] * (max_len - len(word_ids))
        else:
            word_ids = word_ids[:max_len]

        sentence_ids[i] = word_ids

    return sentence_ids
