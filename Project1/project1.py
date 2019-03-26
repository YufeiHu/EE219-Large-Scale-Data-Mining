from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.extmath import randomized_svd
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import text
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from string import punctuation
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import random
import collections
from nltk import pos_tag
import nltk
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from string import punctuation

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier


np.random.seed(42)
random.seed(42)


def load_data_sub(subset, category):
    data_sub = fetch_20newsgroups(subset=subset,
                                  categories=[category],
                                  shuffle=True)
    doc_sub = ""
    for data in data_sub.data:
        doc_sub = doc_sub + data
    return data_sub, doc_sub


def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'


def lemmatize_sent_demo(text):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(nltk.word_tokenize(text))]


def lemmatize_sent(list_word):
    # Text input is string, returns array of lowercased strings(words).
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(list_word)]


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


def stem_rmv_punc(doc):
    return (word for word in lemmatize_sent(analyzer(doc)) if word not in combined_stopwords and not word.isdigit())


"""Initial Setup"""
""" for stopwords"""
stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_en = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_en),set(punctuation),set(stop_words_skt))

analyzer = CountVectorizer().build_analyzer()
wnl = nltk.wordnet.WordNetLemmatizer()


def vectorize_text(text_array_train, text_array_test, min_df, max_df=1.0):
    # vectorizer = CountVectorizer(min_df = min_df, stop_words = "english",token_pattern=r'\b[^\d\W]+\b',analyzer=stem_rmv_punc)
    # t1 = vectorizer.fit_transform(text_array_train.data)
    # t2 = vectorizer.fit_transform(text_array_test.data)
    # tfidf_transformer = TfidfTransformer()
    # tfidf_array_train = tfidf_transformer.fit_transform(t1)
    # tfidf_array_test = tfidf_transformer.fit_transform(t2)

    vectorizer = CountVectorizer(min_df=min_df, stop_words="english", token_pattern=r'\b[^\d\W]+\b', analyzer=stem_rmv_punc)
    text_array_train = vectorizer.fit_transform(text_array_train.data)
    text_array_test = vectorizer.transform(text_array_test.data)
    tfidf_transformer = TfidfTransformer()
    tfidf_array_train = tfidf_transformer.fit_transform(text_array_train)
    tfidf_array_test = tfidf_transformer.transform(text_array_test)

    return tfidf_array_train, tfidf_array_test, vectorizer


def reduce_dim_LSI(tfidf_train, tfidf_test, k):
    U, Sigma, VT = randomized_svd(tfidf_train, n_components=k)
    tfidf_train_hat = U.dot(np.diag(Sigma)).dot(VT)
    distance_train = np.linalg.norm(tfidf_train - tfidf_train_hat, 'fro') ** 2

    model = TruncatedSVD(n_components=k)
    tfidf_train_low = model.fit_transform(tfidf_train)
    tfidf_test_low = model.transform(tfidf_test)
    return tfidf_train_low, tfidf_test_low, distance_train


def reduce_dim_NMF(tfidf_train, tfidf_test, k):
    model = NMF(n_components=k, init='random')
    W_train = model.fit_transform(tfidf_train)
    H_train = model.components_
    tfidf_train_hat = W_train.dot(H_train)
    distance_train = np.linalg.norm(tfidf_train - tfidf_train_hat, 'fro') ** 2

    W_test = model.transform(tfidf_test)
    return W_train, W_test, distance_train


def get_labels(labels):
    labels_bin = list()
    for label in labels:
        if label <= 3:
            labels_bin.append(0)
        else:
            labels_bin.append(1)
    return np.array(labels_bin)


def plot_roc(fpr, tpr, title):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.title(title)
    plt.show()


def print_performance(y_true, y_pred, y_prob, title):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
    confusion = metrics.confusion_matrix(y_true, y_pred)

    plot_roc(fpr, tpr, title)
    print("Confusion Matrix:")
    print(confusion)
    print("Accuracy: {:.4f}".format(metrics.accuracy_score(y_true, y_pred)))
    print("Recall: {:.4f}".format(metrics.recall_score(y_true, y_pred)))
    print("Precision: {:.4f}".format(metrics.precision_score(y_true, y_pred)))
    print("F1 Score: {:.4f}".format(metrics.f1_score(y_true, y_pred)))


def train_SVM(data_train, labels_train, data_test, labels_test, gamma, title):
    model = SVC(C=gamma, kernel='linear', probability=True)
    model.fit(data_train, labels_train)

    y_predict = model.predict(data_test)
    y_prob = model.predict_proba(data_test)

    print_performance(y_true=labels_test,
                      y_pred=y_predict,
                      y_prob=y_prob,
                      title=title)
    
def train_LR(data_train, labels_train, data_test, labels_test, reg, gamma, title):
    model = LogisticRegression(penalty = reg, C = gamma)
    model.fit(data_train, labels_train)

    y_predict = model.predict(data_test)
    y_prob = model.predict_proba(data_test)

    print_performance(y_true=labels_test,
                      y_pred=y_predict,
                      y_prob=y_prob,
                      title=title)


def train_Bayes(data_train, labels_train, data_test, labels_test, title):
    model = GaussianNB()
    model.fit(data_train, labels_train)

    y_predict = model.predict(data_test)
    y_prob = model.predict_proba(data_test)

    print_performance(y_true=labels_test,
                      y_pred=y_predict,
                      y_prob=y_prob,
                      title=title)


def get_optimal_gamma(data_train, labels_train):
    gamma_range = 10.0 ** np.arange(-3, 4)
    kf = StratifiedKFold(n_splits=5)
    performance_best = 0
    gamma_best = 0
    for gamma in gamma_range:
        model = SVC(C=gamma, kernel='linear', probability=True)
        performance = 0
        for train_index, val_index in kf.split(data_train, labels_train):
            data_train_fold = data_train[train_index]
            data_val_fold = data_train[val_index]
            labels_train_fold = labels_train[train_index]
            labels_val_fold = labels_train[val_index]

            model.fit(data_train_fold, labels_train_fold)
            y_predict = model.predict(data_val_fold)
            performance += metrics.accuracy_score(labels_val_fold, y_predict)
        performance /= 5.0
        if performance > performance_best:
            performance_best = performance
            gamma_best = gamma
    return gamma_best

def get_optimal_gamma_LR(data_train, labels_train, reg):
    gamma_range = 10.0 ** np.arange(-3, 4)
    kf = StratifiedKFold(n_splits=5)
    performance_best = 0
    gamma_best = 0
    for gamma in gamma_range:
        model = LogisticRegression(penalty = reg, C = gamma)
        performance = 0
        for train_index, val_index in kf.split(data_train, labels_train):
            data_train_fold = data_train[train_index]
            data_val_fold = data_train[val_index]
            labels_train_fold = labels_train[train_index]
            labels_val_fold = labels_train[val_index]

            model.fit(data_train_fold, labels_train_fold)
            y_predict = model.predict(data_val_fold)
            performance += metrics.accuracy_score(labels_val_fold, y_predict)
        performance /= 5.0
        if performance > performance_best:
            performance_best = performance
            gamma_best = gamma
    return gamma_best

def print_performance_noroc(y_true, y_pred):
    #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
    confusion = metrics.confusion_matrix(y_true, y_pred)

    #plot_roc(fpr, tpr, title)
    print("Confusion Matrix:")
    print(confusion)
    print("Accuracy: {:.4f}".format(metrics.accuracy_score(y_true, y_pred)))
    print("Recall: {:.4f}".format(metrics.recall_score(y_true, y_pred,average='weighted')))
    print("Precision: {:.4f}".format(metrics.precision_score(y_true, y_pred,average='weighted')))
    print("F1 Score: {:.4f}".format(metrics.f1_score(y_true, y_pred,average='weighted')))
    
def get_optimal_gamma_multi(data_train, labels_train, classtype):
    gamma_range = 10.0 ** np.arange(-3, 4)
    kf = StratifiedKFold(n_splits=5)
    performance_best = 0
    gamma_best = 0
    for gamma in gamma_range:
        if classtype == 'OneVsOne':
            model = OneVsOneClassifier(LinearSVC(C=gamma, random_state=42))
        else:
            model = OneVsRestClassifier(LinearSVC(C=gamma, random_state=42))
        performance = 0
        for train_index, val_index in kf.split(data_train, labels_train):
            data_train_fold = data_train[train_index]
            data_val_fold = data_train[val_index]
            labels_train_fold = labels_train[train_index]
            labels_val_fold = labels_train[val_index]

            model.fit(data_train_fold, labels_train_fold)
            y_predict = model.predict(data_val_fold)
            performance += metrics.accuracy_score(labels_val_fold, y_predict)
        performance /= 5.0
        if performance > performance_best:
            performance_best = performance
            gamma_best = gamma
    return gamma_best
    
def dumb_stem(words):
    return [word for word in words]


if __name__ == "__main__":

    # Load data of all the categories and store it into the dictionary
    categories_total = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                        'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                        'rec.sport.hockey', 'misc.forsale', 'soc.religion.christian', 'alt.atheism',
                        'comp.windows.x', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
                        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


    # Load data into corresponding dictionaries
    data_train_total = dict()
    doc_train_total = dict()
    for category in categories_total:
        data_train_sub, doc_train_sub = load_data_sub("train", category)
        data_train_total[category] = data_train_sub
        doc_train_total[category] = doc_train_sub


    # Q1: plot histogram
    train_all = fetch_20newsgroups(subset='train',  # choose which subset of the dataset to use; can be 'train', 'test', 'all'
                                   categories=None, # choose the categories to load; if is `None`, load all categories
                                   shuffle=True,
                                   random_state=42  # set the seed of random number generator when shuffling to make the outcome repeatable across different runs
    )
    result = collections.Counter()
    for i in range(len(train_all.data)):
        if train_all.target[i] not in result:
            result[train_all.target[i]] = 1
        else:
            result[train_all.target[i]] += 1
    result_key = list()
    result_val = list()
    for k, v in result.items():
        result_key.append(k)
        result_val.append(v)
        
    plt.title("Historgram of 20 training categories")
    plt.bar(result.keys(), result.values(), color='g')
    plt.xlabel("Categories")
    plt.ylabel("Number of training documents")
    plt.show()


    # Q2: extract features
    # Load the data that we are interested in
    categories_interest = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                           'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball',
                           'rec.sport.hockey']
    data_train_interest = fetch_20newsgroups(subset="train",
                                             categories=categories_interest,
                                             shuffle=True,
                                             random_state=42)
    data_test_interest = fetch_20newsgroups(subset="test",
                                            categories=categories_interest,
                                            shuffle=True,
                                            random_state=42)

    tfidf_train, tfidf_test, vectorizer = vectorize_text(text_array_train=data_train_interest,
                                                         text_array_test=data_test_interest,
                                                         min_df=3)

    print("training set has size of ", tfidf_train.shape)
    print("testing set has size of ", tfidf_test.shape)


    # Q3: dimensionality reduction
    tfidf_train_LSI, tfidf_test_LSI, distance_train_LSI = reduce_dim_LSI(tfidf_train, tfidf_test, 50)
    tfidf_train_NMF, tfidf_test_NMF, distance_train_NMF = reduce_dim_NMF(tfidf_train, tfidf_test, 50)
    print("Training distance of LSI is: {}".format(distance_train_LSI))
    print("Training distance of NMF is: {}".format(distance_train_NMF))


    # Q4: SVM classifier
    labels_train = get_labels(data_train_interest.target)
    labels_test = get_labels(data_test_interest.target)

    title = "LSI, gamma = 1000"
    print("\n\n{}".format(title))
    train_SVM(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, 1000, title)

    title = "LSI, gamma = 0.0001"
    print("\n\n{}".format(title))
    train_SVM(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, 0.0001, title)


    # Q4: cross validation
    gamma_optimal = get_optimal_gamma(tfidf_train_LSI, labels_train)
    title = "LSI, best gamma = {}".format(gamma_optimal)
    print("\n\n{}".format(title))
    train_SVM(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, gamma_optimal, title)

    # Q5: Linear Regression
    title = "LR"
    train_LR(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, "l2", 1e10, title)

    gamma_optimal_l1 = get_optimal_gamma_LR(tfidf_train_LSI, labels_train, "l1")
    title = "LR with l1, best gamma = {}".format(gamma_optimal_l1)
    print("\n\n{}".format(title))
    train_LR(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, "l1", gamma_optimal_l1, title)

    gamma_optimal_l2 = get_optimal_gamma_LR(tfidf_train_LSI, labels_train, "l2")
    title = "LR with l2, best gamma = {}".format(gamma_optimal_l2)
    print("\n\n{}".format(title))
    train_LR(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, "l2", gamma_optimal_l2, title)

    # Q6: Bayes
    title = "Bayes"
    train_Bayes(tfidf_train_LSI, labels_train, tfidf_test_LSI, labels_test, title)
    
    #Q7: gridsearch
    for i in range(2):
        if i==0:
            twenty_train = fetch_20newsgroups(subset="train",
                                                 categories=categories_interest,
                                                 shuffle=True,
                                                 random_state=42)
        else:
            twenty_train = fetch_20newsgroups(subset="train",
                                                 categories=categories_interest,
                                                 shuffle=True,
                                                 random_state=42,
                                                 remove=('headers', 'footers'))
        twenty_train_target = get_labels(twenty_train.target)

        cachedir = mkdtemp()
        memory = Memory(cachedir=cachedir, verbose=10)

        pipeline = Pipeline([
            ('vect', CountVectorizer(stop_words='english')),
            ('tfidf', TfidfTransformer()),
            ('reduce_dim', TruncatedSVD(random_state=0)),
            ('clf', GaussianNB()),
        ],
        memory=memory
        )


        param_grid = [
            {
                'vect__min_df': [3,5],
                'vect__analyzer': [stem_rmv_punc,dumb_stem],
                'reduce_dim': [TruncatedSVD(), NMF()],  
                'reduce_dim__n_components':[50],
                'clf': [LinearSVC()],
                'clf__C': [10],
            },
            {
                'vect__min_df': [3,5],
                'vect__analyzer': [stem_rmv_punc,dumb_stem],
                'reduce_dim': [TruncatedSVD(), NMF()],
                'reduce_dim__n_components':[50],        
                'clf': [GaussianNB()]
            },
            {
                'vect__min_df': [3,5],
                'vect__analyzer': [stem_rmv_punc,dumb_stem],
                'reduce_dim': [TruncatedSVD(), NMF()],
                'reduce_dim__n_components':[50], 
                'clf': [LogisticRegression()],
                'clf__penalty': ['l1'],
                'clf__C': [10]
            },
            {
                'vect__min_df': [3,5],
                'vect__analyzer': [stem_rmv_punc,dumb_stem],
                'reduce_dim': [TruncatedSVD(), NMF()],
                'reduce_dim__n_components':[50], 
                'clf': [LogisticRegression()],
                'clf__penalty': ['l2'],
                'clf__C': [100]      
            }
        ]

        grid = GridSearchCV(pipeline, cv=5, n_jobs=1, param_grid=param_grid, scoring='accuracy')
        grid.fit(twenty_train.data, twenty_train_target)
        rmtree(cachedir)
        pd.DataFrame(grid.cv_results_)

    #Q8:multiclass classification
    categories_m = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
                    'misc.forsale','soc.religion.christian']

    data_train_m = fetch_20newsgroups(subset='train', categories=categories_m, shuffle=True,
                                    random_state=42)
    data_test_m = fetch_20newsgroups(subset='test', categories=categories_m, shuffle=True,
                                    random_state=42)
    tfidf_train_m, tfidf_test_m, vectorizer_m = vectorize_text(text_array_train=data_train_m,
                                                         text_array_test=data_test_m,
                                                         min_df=3)
    dim_red_method = ['LSI','NMF']
    for j in range(2):
        if dim_red_method[j] == 'LSI':
            xtrain, xtest, distance_train_LSI_m = reduce_dim_LSI(tfidf_train_m, tfidf_test_m, 50)
        else:
            xtrain, xtest, distance_train_NMF_m = reduce_dim_NMF(tfidf_train_m, tfidf_test_m, 50)
    
        clf = []
        classifier = ['GaussianNB','OneVsOneSVM','OneVsRestSVM']
        clf.append( GaussianNB().fit(xtrain, data_train_m.target) )
        gamma_best_OneVsOne = get_optimal_gamma_multi(xtrain, data_train_m.target, 'OneVsOne')
        clf.append( OneVsOneClassifier(LinearSVC(C=gamma_best_OneVsOne, random_state=42)).fit(xtrain, data_train_m.target) ) #'OneOne'
        gamma_best_OneVsRest = get_optimal_gamma_multi(xtrain, data_train_m.target, 'OneVsRest')
        clf.append( OneVsRestClassifier(LinearSVC(C=gamma_best_OneVsRest, random_state=42)).fit(xtrain, data_train_m.target) ) #'OneRest'

        for i in range(3):
            pred = clf[i].predict(xtest)
            print('Use '+dim_red_method[j]+' to reduce dimension, '+classifier[i]+' classification')
            print_performance_noroc(data_test_m.target, pred)
        
