import os 
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB 
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from keras.layers import Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical

from time import time

input_dir = './classified_chat/mix/'
#input_dir = './clean_classified_chat/mix/'
output_data = "date.txt"
output_target = "target.txt"
emote_sentiment_convert = "complete_emote_sentiment.txt"

class Data:
    def __init__(self):
        self.data = list()
        self.target = list()
        self.target_name = list()

def emote_to_sentiment(targets):
    with open (emote_sentiment_convert, "r") as e:
        to_sentiment = {}
        for line in e:
            l = line.split()
            sentiment = l[0]
            l = l[1:]
            for word in l:
                word = word.rstrip("\n")
                word = word.rstrip()
                to_sentiment[word] = sentiment

        for i in range(len(targets)):
            targets[i] = to_sentiment.get(targets[i], "undefined")
    return targets

def create_dataset():
    _data = []
    _targets = []
    _targets_name = set()
    #data_output = open(output_data, 'w')
    #target_output = open(output_target, 'w')
    for filename in os.listdir(input_dir):
        if '.log' not in filename:
            continue
        file_input = open(input_dir + filename, 'r')
        for line in file_input:
            line = line.rstrip()
            sp = line.rfind(' ')
            datum = line[:sp]
            target = line[sp+1:]
            _data.append(datum)
            _targets.append(target)
            _targets_name.add(target)
            
            #data_output.write(line[:sp] + '\n' )
            #target_output.write(line[sp+1:] + '\n')
        file_input.close()
    #data_output.close()
    #target_output.close()
     
    return _data, _targets, sorted(_targets_name)

def test_LSTM(train, test):
    model = Sequential()
    model.add(Embedding(10000, 100, input_length = train.data.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(train.target.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    #model.summary()
    model.fit(train.data, train.target, epochs = 15, batch_size = 64, verbose=2)
    scores = model.evaluate(test.data, test.target, verbose = 0)
    print(scores)
    model.save('lstm_trained_mr_all.h5')

def eval_LSTM(train, test):
    model = load_model('lstm_trained_mr_all.h5')
    _pred = model.predict(test.data, verbose = 0)
    cor = 0
    pred = []
    targ = []
    for i in range(_pred.shape[0]):
        pred.append(np.argmax(_pred[i]))
        targ.append(np.argmax(test.target[i]))

    (precision, recall, f1, support) = metrics.precision_recall_fscore_support(targ, pred, average='micro')
    print("precision: %0.3f" % precision)
    print("recall: %0.3f" % recall)
    print("f1: %0.3f" % f1)
    scores = metrics.precision_recall_fscore_support(targ, pred)
    print("class precision recall f1 support")
    for i in range(_pred.shape[1]):
        print(train.target_name[i], "%.3f" % scores[0][i],
        "%.3f" % scores[1][i], "%.3f" % scores[2][i], scores[3][i])
    print()

names = ["Linear SVM", "Multinomial Naive Bayes" ''',"Multi-layer Perception"''']

classifiers = [
        LinearSVC(),
        MultinomialNB(),
        '''
        MLPClassifier(hidden_layer_sizes = (5,), learning_rate_init = 0.01, 
            alpha = 0.0001, max_iter = 100, verbose = True, early_stopping = False)
        '''
        ]

def test_classifier(clf, train, test):
    t0 = time()
    clf.fit(train.data, train.target)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)
    t0 = time()
    pred = clf.predict(test.data)
    test_time = time() - t0
    print("test time: %0.3fs" % test_time)
    (precision, recall, f1, support) = metrics.precision_recall_fscore_support(test.target, pred, average='micro')
    print("precision: %0.3f" % precision)
    print("recall: %0.3f" % recall)
    print("f1: %0.3f" % f1)
    print()
    scores = metrics.precision_recall_fscore_support(test.target, pred, labels = train.target_name)
    print("class precision recall f1 support")
    for i in range(len(train.target_name)):
        print(train.target_name[i], "%.3f" % scores[0][i],
        "%.3f" % scores[1][i], "%.3f" % scores[2][i], scores[3][i])
    print()
    return clf

def train_and_test(vc, train, test):
    # Test classifiers from sklearn
    for name, clf in zip(names, classifiers):
        print(name, ":")
        clf = test_classifier(clf, train, test)   
        if name == names[0]:
            print("top features:")
            print_top_features(vc, clf, clf.classes_)

def print_top_features(vectorizer, clf, labels):
    feature_names = vectorizer.get_feature_names()
    for i, label in enumerate(labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s:\n[%s]\n" % (label,
            "] [".join(feature_names[j] for j in top10)))
    print()

def main():
    _data, _target, _target_name = create_dataset()
    _target = emote_to_sentiment(_target)
    _target_name = sorted(set(_target))
    print('data creation finished')
    print("emote class names: ", _target_name)
    print()
    train = Data();
    test = Data();
    test.target_name = train.target_name = _target_name

    '''
    # Sequential words feature for LSTM
    print('Using Sequantial data features')
    tokenizer = Tokenizer(num_words = 10000, split=' ', lower = True)
    tokenizer.fit_on_texts(_data)
    data_vectors = pad_sequences(tokenizer.texts_to_sequences(_data))
    target_vectors = pd.get_dummies(_target).values
    train.data, test.data, train.target, test.target = train_test_split(
            data_vectors,  target_vectors, test_size = 0.4, random_state = 0)
    print("training set shape: ", train.data.shape, train.target.shape)
    print("testing set shape: ", test.data.shape, test.target.shape)

    test_LSTM(train, test)
    print("Evaluating:")
    eval_LSTM(train, test)
    print()
    '''
    # TFIDF-Bag of Word Features
    print('Using TFIDF-Bag of word features')
    vectorizer = TfidfVectorizer(lowercase = True, ngram_range = (1,2), max_features = None)
    data_vectors = vectorizer.fit_transform(_data)
    train.data, test.data, train.target, test.target = train_test_split(
            data_vectors, _target, test_size = 0.4, random_state = 0)
    print("training set shape: ", train.data.shape, len(train.target))
    print("testing set shape: ", test.data.shape, len(test.target))
    #print(_target_name)
    print()
    clf = train_and_test(vectorizer, train, test) 

if __name__ == '__main__':
    main()

