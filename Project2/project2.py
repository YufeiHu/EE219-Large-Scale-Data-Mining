import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (confusion_matrix, homogeneity_score, completeness_score, 
v_measure_score, adjusted_rand_score, adjusted_mutual_info_score)

np.random.seed(42)
random.seed(42)


def get_contingency_table(label_true, label_pred):
    cmat = confusion_matrix(label_true, label_pred)
    print("Confusion Matrix:")
    print(cmat)


def reduce_dimension_SVD(data, k):
    model = TruncatedSVD(n_components=k, random_state=0)
    data_reduced = model.fit_transform(data)
    return data_reduced


def reduce_dimension_NMF(data, k):
    model = NMF(n_components=k, init='random', random_state=0)
    data_reduced = model.fit_transform(data)
    return data_reduced


def print_performance(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    TN, FP = confusion[0, 0], confusion[0, 1]
    FN, TP = confusion[1, 0], confusion[1, 1]

    print ("Confusion Matrix is: ")
    print(confusion)

    print ("Homogeneity score is :" + str(homogeneity_score(y_true, y_pred)))
    print ("Completeness score is :" + str(completeness_score(y_true, y_pred)))
    print ("V measure score is :" + str(v_measure_score(y_true, y_pred)))
    print ("Adjusted rand score is :" + str(adjusted_rand_score(y_true, y_pred)))
    print ("Adjusted mutual info score is :" + str(adjusted_mutual_info_score(y_true, y_pred)))


def plot_visualization_results(data_2D, label_true, label_pred, title):
    plt.figure()
    ax = plt.subplot(1, 2, 1)
    ax.set_title("Clustering label ({})".format(title))
    ax.scatter(x=data_2D[:, 0], y=data_2D[:, 1], c=label_pred)
    ax = plt.subplot(1, 2, 2)
    ax.set_title("True label ({})".format(title))
    ax.scatter(x=data_2D[:, 0], y=data_2D[:, 1], c=label_true)
    plt.tight_layout()
    plt.show()


def visualize_results(data_reduced, label_true, title, flip):
    print("\n\nClustering Result for {}".format(title))
    model_kmeans_reduced = KMeans(n_clusters=2, random_state=0)
    model_kmeans_reduced.fit(data_reduced)
    label_pred_reduced = model_kmeans_reduced.labels_
    print_performance(label_true, label_pred_reduced)

    # data_2D = reduce_dimension_SVD(data_reduced, 2)
    model_kmeans_2D = KMeans(n_clusters=2, random_state=0)
    model_kmeans_2D.fit(data_reduced)
    label_pred_2D = model_kmeans_2D.labels_
    label_pred = list(label_pred_2D)
    if flip:
        for i, l in enumerate(label_pred):
            if l == 0:
                label_pred[i] = 1
            elif l == 1:
                label_pred[i] = 0
            else:
                raise ValueError("Impossible!")
    plot_visualization_results(data_reduced, label_true, label_pred, title)


def transform_scaling(data):
    return preprocessing.scale(data)


def transform_nonLinear(data, c):
    signs = np.sign(data)
    tmp = np.log(np.abs(data) + c) - np.log(c)
    return signs * tmp


def plot_variance (data):
    svd = TruncatedSVD(n_components=1000)
    svd.fit(data)
    
    variances = []
    sum = 0
    for r in svd.explained_variance_ratio_:
        sum += r
        variances.append(sum)
    plt.figure()
    plt.plot(variances, color='darkorange', lw=2)
    plt.xlabel('r', fontsize=15)
    plt.ylabel('The percent of variance', fontsize=15)
    plt.title('The percent of variance of the top principle components')
    plt.show()


def plot_r_choice (data, labels, method):
    if method == 'SVD':
        data_svd = reduce_dimension_SVD(data, 1000)
    rs = [1,2,3,5,10,20,50,100,300]
    H = []
    C = []
    V = []
    Ar = []
    Am = []
    for r in rs:
        km = KMeans(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
        if method == 'NMF':
            km.fit(reduce_dimension_NMF(data, r))
        else:
            km.fit(data_svd[:,:r])
        H.append(homogeneity_score(labels, km.labels_))
        C.append(completeness_score(labels, km.labels_))
        V.append(v_measure_score(labels, km.labels_))
        Ar.append(adjusted_rand_score(labels, km.labels_))
        Am.append(adjusted_mutual_info_score(labels, km.labels_))
        
    plt.figure()
    plt.plot(rs, H, color='darkorange', lw=2, label='homogeneity')
    plt.plot(rs, C, color='darkmagenta', lw=2, label='completeness')
    plt.plot(rs, V, color='green', lw=2, label='V-measure')
    plt.plot(rs, Ar, color='navy', lw=2, label='adjusted rand index')
    plt.plot(rs, Am, color='red', lw=2, label='adjusted mutual information')
    plt.xlabel('r', fontsize=15)
    plt.ylabel('Score', fontsize=15)
    plt.title('The measure scores of the dimension for ' + method)
    plt.legend()
    plt.show()
    
    return rs[np.argmax(np.array(Ar))]


if __name__ == "__main__":

    categories = ['comp.sys.ibm.pc.hardware', 'comp.graphics','comp.sys.mac.hardware', 'comp.os.ms-windows.misc','rec.autos', 'rec.motorcycles','rec.sport.baseball', 'rec.sport.hockey']
    
    # Q1: dimensions of the TF-IDF matrix
    dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)
    labels = [int(x / 4) for x in dataset.target]
    vectorizer = CountVectorizer(min_df=3, stop_words="english")
    dataset_array = vectorizer.fit_transform(dataset.data)
    tfidf_transformer = TfidfTransformer()
    dataset_tfidf = tfidf_transformer.fit_transform(dataset_array)
    print("p1: dimensions of the TF-IDF matrix is: ", dataset_tfidf.shape)
    

    # Q2: contingency table of clustering result
    km = KMeans(n_clusters=2, random_state=0, max_iter=1000, n_init=30)
    km.fit(dataset_tfidf)
    get_contingency_table(labels, km.labels_)


    # Q3: 5 measures
    print("Homogeneity: %0.4f" % homogeneity_score(labels, km.labels_))
    print("Completeness: %0.4f" % completeness_score(labels, km.labels_))
    print("V-measure: %0.4f" % v_measure_score(labels, km.labels_))
    print("Adjusted Rand Index: %.4f" % adjusted_rand_score(labels, km.labels_))
    print("Adjusted mutual info score: %.4f" % adjusted_mutual_info_score(labels, km.labels_))


    # Q4: plot variance
    plot_variance(dataset_tfidf)


    # Q5,6: SVD and NMF
    best_r_svd = plot_r_choice (dataset_tfidf, labels, "SVD")
    print("The best r for SVD is " + str(best_r_svd))

    best_r_nmf = plot_r_choice (dataset_tfidf, labels, "NMF")
    print("The best r for NMF is " + str(best_r_nmf))


    # Q7: visualize clustering results
    r_best = 2
    data_original = dataset_tfidf
    label_true = labels

    data_reduced_SVD = reduce_dimension_SVD(data_original, r_best)
    data_reduced_NMF = reduce_dimension_NMF(data_original, r_best)

    visualize_results(data_reduced_SVD, label_true, "SVD", False)
    visualize_results(data_reduced_NMF, label_true, "NMF", True)


    # Q8 - Q10: visualize transformed data (SVD)
    data_reduced_SVD_s = transform_scaling(data_reduced_SVD)
    visualize_results(data_reduced_SVD_s, label_true, "SVD S.", True)

    data_reduced_SVD_n = transform_nonLinear(data_reduced_SVD, c=0.01)
    visualize_results(data_reduced_SVD_n, label_true, "SVD N.", True)

    data_reduced_SVD_sn = transform_nonLinear(data_reduced_SVD_s, c=0.01)
    visualize_results(data_reduced_SVD_sn, label_true, "SVD S.N.", False)

    data_reduced_SVD_ns = transform_scaling(data_reduced_SVD_n)
    visualize_results(data_reduced_SVD_ns, label_true, "SVD N.S.", True)


    # Q8 - Q10: visualize transformed data (NMF)
    data_reduced_NMF_s = transform_scaling(data_reduced_NMF)
    visualize_results(data_reduced_NMF_s, label_true, "NMF S.", False)

    data_reduced_NMF_n = transform_nonLinear(data_reduced_NMF, c=0.01)
    visualize_results(data_reduced_NMF_n, label_true, "NMF N.", False)

    data_reduced_NMF_sn = transform_nonLinear(data_reduced_NMF_s, c=0.01)
    visualize_results(data_reduced_NMF_sn, label_true, "NMF S.N.", False)

    data_reduced_NMF_ns = transform_scaling(data_reduced_NMF_n)
    visualize_results(data_reduced_NMF_ns, label_true, "NMF N.S.", True)

    
    #Q11: Repeat the following for 20 categories using the same parameters as in 2-class case
    print("Q11: For the dataset with 20 categories: \n")
    dataset_all = fetch_20newsgroups(subset='all', categories=None, shuffle=True, random_state=42)
    labels_all = dataset_all.target
    dataset_array_all = vectorizer.fit_transform(dataset_all.data)
    dataset_tfidf_all = tfidf_transformer.fit_transform(dataset_array_all)
    km_all = KMeans(n_clusters = 20, random_state = 0)
    km_all.fit(dataset_tfidf_all)
    get_contingency_table(labels_all, km_all.labels_)
    print("Homogeneity: %0.4f" % homogeneity_score(labels_all, km_all.labels_))
    print("Completeness: %0.4f" % completeness_score(labels_all, km_all.labels_))
    print("V-measure: %0.4f" % v_measure_score(labels_all, km_all.labels_))
    print("Adjusted Rand Index: %.4f" % adjusted_rand_score(labels_all, km_all.labels_))
    print("Adjusted mutual info score: %.4f" % adjusted_mutual_info_score(labels_all, km_all.labels_))
    #Q12 different dimensions for both truncated SVD and NMF dimensionality reduction techniques
    import time
    # rs = [1,2,3,5,10,20,50,100,300]
    rs = [2]
    methods = ["SF","NL","SFNL","NLSF"]
    reductions = ["SVD","NMF"]
    scores = dict()

    for x,r in enumerate(rs):
        scores[r] = {}
        for y,m in enumerate(methods):
            scores[r][m] = {}
            for z,red in enumerate(reductions):
                t = time.clock()
                if red == "SVD":
                    reduced_data_all = reduce_dimension_SVD(dataset_tfidf_all,r)
                else:
                    reduced_data_all = reduce_dimension_NMF(dataset_tfidf_all,r)

                if m == "SF":
                    rdata = transform_scaling(reduced_data_all)
                elif m == "NL":
                    rdata = transform_nonLinear(reduced_data_all,0.01)
                elif m == "SFNL":
                    rdata = transform_scaling(reduced_data_all)
                    rdata = transform_nonLinear(rdata,0.01)
                elif m == "NLSF":
                    rdata = transform_nonLinear(reduced_data_all,0.01)
                    rdata = transform_scaling(rdata)

                km_all = KMeans(n_clusters = 20,random_state = 0)
                km_all.fit(rdata)
                scores[r][m][red] = [homogeneity_score(labels_all, km_all.labels_),completeness_score(labels_all, km_all.labels_),
                                  v_measure_score(labels_all, km_all.labels_),adjusted_rand_score(labels_all, km_all.labels_),
                                  adjusted_mutual_info_score(labels_all, km_all.labels_)]

                print(scores[r][m][red])
                print("Methods {0}, Reductions {1}, and r {2} cost time {3}".format(m,red,r,time.clock()-t))
                t = time.clock()