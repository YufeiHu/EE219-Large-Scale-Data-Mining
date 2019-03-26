import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from surprise import Reader
from surprise import Dataset
from surprise import KNNWithMeans
from surprise.model_selection import cross_validate

from surprise.model_selection import KFold
from surprise import accuracy
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
from surprise.model_selection import train_test_split
from sklearn import metrics

from collections import defaultdict

class Data:
    def __init__(self):
        self.user_IDs = None
        self.movie_IDs = None
        self.ratings = None
        self.titles = None
        self.genres = None

    def load_ratings(self, path):
        dataset = pd.read_csv(path)
        self.user_IDs = dataset['userId'].tolist()
        self.movie_IDs = dataset['movieId'].tolist()
        self.ratings = dataset['rating'].tolist()

    def load_movies(self, path):
        dataset = pd.read_csv(path)
        self.movie_IDs = dataset['movieId'].tolist()
        self.titles = dataset['title'].tolist()
        self.genres = dataset['genres'].tolist()


def load_movie_ratings(path):
    data_ratings = Data()
    data_ratings.load_ratings(path)
    return data_ratings


def load_data_movies(path):
    data_movies = Data()
    data_movies.load_movies(path)
    return data_movies

def trimMovies():
    pop, unpop, highVar = set(), set(), set()

    movies = list()
    movies_ratings = list()
    ratings_all = list()
    movies_counts = list()

    ID_ratings = sorted(zip(movie_IDs, ratings))
    movies.append(ID_ratings[0][0])
    count1=0
    for i in range(len(ID_ratings)):
        if ID_ratings[i][0] not in movies:
            movies.append(ID_ratings[i][0])
            variance = np.var(ratings_all)
            ratings_all = list()
            movies_ratings.append(variance)
            movies_counts.append(count1)
            count1=0
        ratings_all.append(ID_ratings[i][1])
        count1=count1+1
    variance = np.var(ratings_all)
    movies_ratings.append(variance)
    movies_counts.append(count1)
    
    for i in range(len(movies)):
        if movies_counts[i] <= 2:
            unpop.add(movies[i])
        if movies_counts[i] > 2:
            pop.add(movies[i])
        if movies_counts[i] >= 5 and movies_ratings[i] >= 2.0:
            highVar.add(movies[i])
    return pop, unpop, highVar



def trim_performance(qNum,maxk=0): 
    pop, unpop, highVar = trimMovies()
    
    if maxk == 0:
        if 12 <= qNum <= 14:
            maxk = 100
        elif 19 <= qNum <= 21:
            maxk = 50

    trim_Model = {
        12: (pop, 'KNNWithMeans'),
        13: (unpop, 'KNNWithMeans'),
        14: (highVar, 'KNNWithMeans'),
        19: (pop, 'NMF'),
        20: (unpop, 'NMF'),
        21: (highVar, 'NMF'),
    }
    trimSet, modelName = trim_Model[qNum]
    
    kf = KFold(n_splits=10)
    RMSE = [] 
    for k in range(2, maxk + 1, 2):
        print('-' * 20 + 'k = ' + str(k) + ' ' + '-' * 20)
        
        if modelName == 'KNNWithMeans':
            model = KNNWithMeans(k=k, sim_options={'name': 'pearson'})
        elif modelName == 'NMF':
            model = NMF(n_factors=k)

        subRMSE = [] 
        temp = 1
        for trainSet, testSet in kf.split(data):
            model.fit(trainSet)
            testSet = list(filter(lambda x: int(x[1]) in trimSet, testSet))
            print("Split " + str(temp) + ": test set size after trimming: %d", len(testSet))
            temp += 1
            predictions = model.test(testSet)
            subRMSE.append(accuracy.rmse(predictions, verbose=True))
        RMSE.append(np.mean(subRMSE))

    plt.figure()
    plt.plot(list(range(2, maxk+1, 2)), RMSE)
    plt.xlabel("k")
    plt.ylabel("Average RMSE")
    plt.title("Q"+str(qNum)+": Average RMSE Along k")
    plt.show()
    print(min(RMSE))
    return min(RMSE)

def plot_ROC(qNum, k, thresh=[2.5,3,3.5,4]):
    range = 5.0
    trainset, testset = train_test_split(data, test_size=0.1)
    if qNum == 15:
        model = KNNWithMeans(k=k, sim_options={'name': 'pearson'})
    model.fit(trainset)
    predictions = model.test(testset)
    
    for thrs in thresh:
        y = np.array([])
        scores = np.array([])
        for u, i, t, est, d in predictions:
            if t >= thrs:
                t = 1
            else:
                t = 0
            y = np.append(y, t)
            scores = np.append(scores, est/range)
        
        fpr, tpr, thresholds = metrics.roc_curve(y, scores)
        roc_auc = metrics.auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Threshold = '+str(thrs))
        plt.show()
        print("auc = "+str(roc_auc))

def plot_all_ROC():
    rang = 5.0
    sim_options = {
        'name': 'pearson_baseline',
        'shrinkage': 0  # no shrinkage
    }
    trainset, testset = train_test_split(data, test_size=0.1)
    knn = KNNWithMeans(22, sim_options=sim_options)
    nmf = NMF(n_factors=18)
    svd = SVD(n_factors=8)
    fp = {}
    tp = {}
    area = np.array([])
    for model, key in zip([knn, nmf, svd], ['KNN','NNMF','SVD']):
        model.fit(trainset)
        pred = model.test(testset)
        np_true = np.array([])
        np_score = np.array([])
        for _, _, t, p, _ in pred:
            if t >= 3:
                t = 1
            else:
                t = 0
            np_true = np.append(np_true, t)
            np_score = np.append(np_score, p/rang)
        fpr, tpr, thresholds = metrics.roc_curve(np_true, np_score)
        print(fpr.shape, tpr.shape)
        roc_auc = metrics.auc(fpr, tpr)
        fp[key] = fpr
        tp[key] = tpr
        area = np.append(area, roc_auc)
    plt.figure()
    lw = 2
    for mod, f, t, roc_auc in zip(['k-NN','NNMF','MF'], fp, tp, area):
        fpr = fp[f]
        tpr = tp[t]
        plt.plot(fpr, tpr, lw=lw, label='%s'%mod)
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.show()

def precision_recall (predictions, t):
    threshold = 3
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_t = sum((est >= threshold) for (est, _) in user_ratings[:t])
        n_rel_and_rec_t = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:t])
        precisions[uid] = n_rel_and_rec_t / n_rec_t if n_rec_t != 0 else 1
        recalls[uid] = n_rel_and_rec_t / n_rel if n_rel != 0 else 1

    return precisions, recalls

def rank_predictions(model_name):

    k_KNN = 22 
    k_NNMF = 20
    k_MF = 26

    if model_name == 'KNN':
        sim_options = {
            'name': 'pearson_baseline',
            'shrinkage': 0
        }
        model = KNNWithMeans(k_KNN, sim_options=sim_options)
    elif model_name == 'NNMF':
        model = NMF(n_factors= k_NNMF)
    else:
        model = SVD(n_factors = k_MF)

    precision_arr = []
    recall_arr = []
    for t in range (1,26):
        kf = KFold(n_splits=10)
        print(t)
        p = []
        r = []
        for trainSet, testSet in kf.split(data):
            model.fit(trainSet)
            predictions = model.test(testSet)
            precisions, recalls = precision_recall (predictions, t)
            p.append(sum(prec for prec in precisions.values()) / len(precisions))
            r.append(sum(rec for rec in recalls.values()) / len(recalls))
            
        precision_arr.append(np.mean(np.array(p)))
        recall_arr.append(np.mean(np.array(r)))

    # precision vs t
    plt.plot(list(range (1,26)), precision_arr)
    plt.xlabel("Size")
    plt.ylabel("Precision")
    plt.title("The average precision plot using " + model_name)
    plt.show()
    
    # recall vs t
    plt.plot(list(range (1,26)), recall_arr)
    plt.xlabel("Size")
    plt.ylabel("Recall")
    plt.title("The average recall plot using MF " + model_name)
    plt.show()
    
    # precision vs recall 
    plt.plot(recall_arr, precision_arr)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("The average precision and recall plot using " + model_name)
    plt.show()


    return precision_arr, recall_arr 


if __name__ == "__main__":

    # Q1:
    dataset_ratings = load_movie_ratings('./data/ratings.csv')
    user_IDs = dataset_ratings.user_IDs
    movie_IDs = dataset_ratings.movie_IDs
    ratings = dataset_ratings.ratings

    dataset_movies = load_data_movies('./data/movies.csv')
    movie_dict = dict(zip(dataset_movies.movie_IDs, dataset_movies.genres))

    num_users = max(user_IDs)
    movie_set = set()
    for movie_ID in movie_IDs:
        movie_set.add(movie_ID)
    num_movies = len(movie_set)

    print("Sparsity = {:.3f}".format(float(len(ratings) / (num_users * num_movies))))


    # Q2:
    rating_bins = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    plt.figure()
    plt.hist(ratings, bins=rating_bins)
    plt.title("Q2: Frequency of Rating Values")
    plt.show()


    # Q3:
    ID_ratings = sorted(zip(movie_IDs, ratings))
    ID_list = list()
    ratings_list = list()
    idx = -1
    for i in range(len(ID_ratings)):
        if ID_ratings[i][0] not in ID_list:
            ID_list.append(ID_ratings[i][0])
            ratings_list.append(0)
            idx += 1
        ratings_list[idx] += 1
    ratings_IDs = sorted(zip(ratings_list, ID_list))[::-1]
    ratings_list, IDs_list = zip(*ratings_IDs)

    plt.figure()
    plt.plot(ratings_list)
    plt.xlabel("Movies")
    plt.ylabel("Number of Movie Ratings")
    plt.title("Q3: Distribution of Ratings Among Movies")
    plt.show()


    # Q4:
    users = list()
    users_movies = list()
    for i in user_IDs:
        if i not in users:
            users.append(i)
            users_movies.append(user_IDs.count(i))
    result = sorted(zip(users_movies, users))[::-1]
    users_movies.sort(reverse=True)

    plt.figure()
    plt.plot(users_movies)
    plt.xlabel("Users")
    plt.ylabel("Number of Movie Ratings")
    plt.title("Q4: Distribution of Ratings Among Users")
    plt.show()


    # Q6
    movies = list()
    movies_ratings = list()
    ratings_all = list()

    ID_ratings = sorted(zip(movie_IDs, ratings))
    movies.append(ID_ratings[0][0])
    for i in range(len(ID_ratings)):
        if ID_ratings[i][0] not in movies:
            movies.append(ID_ratings[i][0])
            variance = np.var(ratings_all)
            ratings_all = list()
            movies_ratings.append(variance)
        ratings_all.append(ID_ratings[i][1])
    variance = np.var(ratings_all)
    movies_ratings.append(variance)

    plt.figure()
    upper = math.floor(max(movies_ratings) + 1)
    bins = np.arange(0, upper, 0.5)
    plt.xlim(0, upper)
    plt.hist(movies_ratings, bins=bins, alpha=0.5)
    plt.xlabel("Variance")
    plt.ylabel("Number of Movie Ratings")
    plt.title("Q6: Distribution of Ratings Along Variance")
    plt.show()


    # Q10:
    path = os.path.expanduser('./data/ratings.csv')
    reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0, 5), skip_lines=1)
    data = Dataset.load_from_file(path, reader=reader)

    rmse_list = list()
    mae_list = list()
    k_range = range(2, 101, 2)
    rmse_min = float('inf')
    mae_min = float('inf')
    k_min = 0
    for k in k_range:
        model = KNNWithMeans(k=k, sim_options={'name': 'pearson'})
        result_cur = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=10)
        rmse = sum(result_cur['test_rmse']) / 10.0
        mae = sum(result_cur['test_mae']) / 10.0
        if rmse < rmse_min and mae < mae_min:
            rmse_min = rmse
            mae_min = mae
            k_min = k
        rmse_list.append(rmse)
        mae_list.append(mae)

    plt.figure()
    plt.plot(k_range, rmse_list)
    plt.plot(k_range, mae_list)
    plt.xlabel("k")
    plt.ylabel("Average Error")
    plt.legend(['RMSE', 'MAE'])
    plt.title("Q10: Average RMSE and MAE Along k")
    plt.show()
    
    #Q12-14:
    trim_performance(12)
    trim_performance(13)
    trim_performance(14)
    
    #Q15:
    plot_ROC(15, 22)
    
    #Q17:
    rmse_list = list()
    mae_list = list()
    k_range = range(2, 51, 2)
    rmse_min = float('inf')
    mae_min = float('inf')
    k_min = [0,0]
    for k in k_range:
        model = NMF(n_factors=k)
        result_cur = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=10)
        rmse = sum(result_cur['test_rmse']) / 10.0
        mae = sum(result_cur['test_mae']) / 10.0
        if rmse < rmse_min:
            rmse_min = rmse
            k_min[0] = k
        if mae < mae_min:
            mae_min = mae
            k_min[1] = k
        rmse_list.append(rmse)
        mae_list.append(mae)

    plt.figure()
    plt.plot(k_range, rmse_list)
    plt.plot(k_range, mae_list)
    plt.xlabel("k")
    plt.ylabel("Average Error")
    plt.legend(['RMSE', 'MAE'])
    plt.title("Q17: Average RMSE and MAE against k")
    plt.show()
    print('Optimal numbers of latent factors that gives the minimum average RMSE and the minimum average MAE are '+str(k_min[0])+', '+str(k_min[1])+' ,respectively. The minimum average RMSE and MAE are'+str(rmse_min)+', '+str(mae_min)+',respectively.')
    
    #Q19-21:
    trim_performance(19)
    trim_performance(20)
    trim_performance(21)

    #Q22:
    print('Question 22:')
    plot_ROC(22,20)

    #Q23:
    print('Question 23:')
    #buildup rating matrix
    df = pd.read_csv('./data/ratings.csv',usecols=['userId','movieId','rating'])
    reader = Reader(line_format='user item rating', sep='\t', rating_scale=(0.5,5.0))
    data = Dataset.load_from_df(df, reader)
    optimal_k = 20
    trainset = data.build_full_trainset()
    algo = NMF(n_factors=optimal_k, random_state=42)
    algo.fit(trainset)
    #get U and V
    U_user_latent = algo.pu
    V_movie_latent = algo.qi
    #sort according to column of V
    index_sort = np.argsort(V_movie_latent,axis = 0)
    index = []
    for i in range(10):
        index.append(index_sort[9065-i])
    #report the performance
    x = pd.read_csv('./data/movies.csv')
    for i in range(20):
        for j in range(10):
            idx = trainset.to_raw_iid(index[j][i])
            print('The {0}th movieId for {1}th column is {2}, the genres are {3}'.format(j,i+1,idx,x.loc[x.movieId == idx,'genres'].values[0]))
        print('-'*70)

    #24
    print("Question 24:")
    rmse_list = list()
    mae_list = list()
    k_range = range(2, 51, 2)
    rmse_min = float('inf')
    mae_min = float('inf')
    k_min = [0,0]
    for k in k_range:
        model = SVD(n_factors = k, biased = True)
        result_cur = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=10)
        rmse = sum(result_cur['test_rmse']) / 10.0
        mae = sum(result_cur['test_mae']) / 10.0
        if rmse < rmse_min:
            rmse_min = rmse
            k_min[0] = k
        if mae < mae_min:
            mae_min = mae
            k_min[1] = k
        rmse_list.append(rmse)
        mae_list.append(mae)

    plt.figure()
    plt.plot(k_range, rmse_list)
    plt.plot(k_range, mae_list)
    plt.xlabel("k")
    plt.ylabel("Average Error")
    plt.legend(['RMSE', 'MAE'])
    plt.title("Q24: Average RMSE and MAE against k")
    plt.show()
    print('Optimal numbers of latent factors that gives the minimum average RMSE and the minimum average MAE are '+str(k_min[0])+', '+str(k_min[1])+' ,respectively. The minimum average RMSE and MAE are'+str(rmse_min)+', '+str(mae_min)+',respectively.')

    #Q26-28
    print("Question 26:")
    trim_performance(26)
    print("Question 27:")
    trim_performance(27)
    print("Question 28:")
    trim_performance(28)
    #Q29
    print('Question 29:')
    plot_ROC(29,26)

    #Q30
    df = pd.read_csv('./data/ratings.csv',usecols=['userId','movieId','rating'])
    reader = Reader(line_format='user item rating', sep='\t', rating_scale=(0.5,5.0))
    data = Dataset.load_from_df(df, reader)

    pop, unpop, highVar = trimMovies()
    Naive_Collective(None)
    #Q31
    Naive_Collective(pop)
    #Q32
    Naive_Collective(unpop)

    #Q32
    Naive_Collective(highVar)

    #Q34
    plot_all_ROC()

    #Q36
    precision_knn, recall_knn = rank_predictions("KNN")

    #Q37
    precision_nmf, recall_nmf = rank_predictions("NNMF")

    #Q38
    precision_svd, recall_svd = rank_predictions("MF")

    #Q39
    plt.plot(recall_knn, precision_knn, label = "KNN")
    plt.plot(recall_nmf, precision_nmf, label = "NNMF")
    plt.plot(recall_svd, precision_svd, label = "MF")
    xlabel = "Recall"
    ylabel = "Precision"
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title("The average precision and recall curve for KNN, NNMF and MF")
    plt.show()



