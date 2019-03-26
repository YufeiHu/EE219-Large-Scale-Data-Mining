def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)


import os
import json
import math
import random
import datetime
import pytz
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import statsmodels.api as stats_api
from statsmodels.regression.linear_model import OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import statsmodels.api as sm
from pprint import pprint
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.utils.extmath import randomized_svd
from sklearn import metrics
from string import punctuation
from nltk import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_curve, auc
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def Q1Q2():
    hashtag_set = ['#GoHawks', '#GoPatriots', '#NFL', '#Patriots', '#SB49', '#SuperBowl']
    hashtag_filename = {'#GoHawks': 'tweets_#gohawks.txt',
                        '#GoPatriots': 'tweets_#gopatriots.txt',
                        '#NFL': 'tweets_#nfl.txt',
                        '#Patriots': 'tweets_#patriots.txt',
                        '#SB49': 'tweets_#sb49.txt',
                        '#SuperBowl': 'tweets_#superbowl.txt'}

    for hashtag in hashtag_set:
        onefile = open(hashtag_filename[hashtag])

        time_stamps = []
        num_retweets = []
        num_followers = []

        for line in onefile:
            onetweet = json.loads(line)
            time_stamps.append(onetweet['citation_date'])
            num_retweets.append(onetweet['metrics']['citations']['total'])
            num_followers.append(onetweet['author']['followers'])

        onefile.close()

        total_hours = float(max(time_stamps) - min(time_stamps)) / 3600.0
        total_num_tweets = len(time_stamps)
        total_num_retweets = sum(num_retweets)
        total_num_followers = sum(num_followers)

        # Q1
        print('-' * 20)
        print('Statistics for', hashtag)
        print('Average number of tweets per hour:', total_num_tweets / total_hours)
        print('Average number of followers of users posting the tweets per tweet:',
              total_num_followers / total_num_tweets)
        print('Average number of retweets per tweet:', total_num_retweets / total_num_tweets)

        # Q2
        if hashtag in ['#NFL', '#SuperBowl']:
            tweet_per_hour = [0] * (int(total_hours) + 1)
            start_time = min(time_stamps)
            for time_stamp in time_stamps:
                tweet_per_hour[int((time_stamp - start_time) / 3600)] += 1

            plt.figure()
            plt.bar(range(1, len(tweet_per_hour) + 1), tweet_per_hour)
            plt.xlabel('Time(hour)')
            plt.ylabel('Number of tweets')
            plt.title('Number of tweets over time (in hour) for ' + hashtag)
            plt.show()


def Q3():
    hashtag_set = ['#GoHawks', '#GoPatriots', '#NFL', '#Patriots', '#SB49', '#SuperBowl']
    hashtag_filename = {'#GoHawks': 'tweets_#gohawks.txt',
                        '#GoPatriots': 'tweets_#gopatriots.txt',
                        '#NFL': 'tweets_#nfl.txt',
                        '#Patriots': 'tweets_#patriots.txt',
                        '#SB49': 'tweets_#sb49.txt',
                        '#SuperBowl': 'tweets_#superbowl.txt'}

    for hashtag in hashtag_set:
        onefile = open(hashtag_filename[hashtag])
        time_stamps = []
        for line in onefile:
            onetweet = json.loads(line)
            time_stamps.append(onetweet['citation_date'])
        start_time = min(time_stamps) // 3600 * 3600
        total_hours = (max(time_stamps) - start_time) // 3600 + 1
        onefile.close()

        onefile = open(hashtag_filename[hashtag])

        tweetCount = [0] * total_hours
        retweetCount = [0] * total_hours
        followerCount = [0] * total_hours
        max_followers = [0] * total_hours
        time_of_day = [0] * total_hours

        for line in onefile:
            onetweet = json.loads(line)
            current_hour = (onetweet['citation_date'] - start_time) // 3600
            tweetCount[current_hour] += 1
            retweetCount[current_hour] += onetweet['metrics']['citations']['total']
            followerCount[current_hour] += onetweet['author']['followers']
            max_followers[current_hour] = max(max_followers[current_hour], onetweet['author']['followers'])

        i = 0
        while i < total_hours:
            time_of_day[i] = i % 24
            i += 1

        onefile.close()

        featureset = np.array([tweetCount, retweetCount, followerCount, max_followers, time_of_day])
        featureset = featureset.transpose()
        X = featureset[:-1, 0:5]
        y = featureset[1:, 0]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        result = stats_api.OLS(y, X).fit()

        print('hashtag: ', hashtag)
        print('mse: ', mse)
        print('t-test results:')
        print(result.summary())
        print('P values: ')
        print(result.pvalues)


def Q4():
    hashtag_set = ['#GoHawks', '#GoPatriots', '#NFL', '#Patriots', '#SB49', '#SuperBowl']
    hashtag_filename = {'#GoHawks': 'tweets_#gohawks.txt',
                        '#GoPatriots': 'tweets_#gopatriots.txt',
                        '#NFL': 'tweets_#nfl.txt',
                        '#Patriots': 'tweets_#patriots.txt',
                        '#SB49': 'tweets_#sb49.txt',
                        '#SuperBowl': 'tweets_#superbowl.txt'}

    for hashtag in hashtag_set:
        onefile = open(hashtag_filename[hashtag])
        time_stamps = []
        for line in onefile:
            onetweet = json.loads(line)
            time_stamps.append(onetweet['citation_date'])
        start_time = min(time_stamps) // 3600 * 3600
        total_hours = (max(time_stamps) - start_time) // 3600 + 1
        onefile.close()

        onefile = open(hashtag_filename[hashtag])

        tweetCount = [0] * total_hours
        retweetCount = [0] * total_hours
        followerCount = [0] * total_hours
        max_followers = [0] * total_hours
        time_of_day = [0] * total_hours

        url_citations = [0] * total_hours
        num_authors = [0] * total_hours
        hourly_author_set = [0] * total_hours
        for i in range(total_hours):
            hourly_author_set[i] = set()
        num_mentions = [0] * total_hours
        ranking_scores = [0] * total_hours
        num_hashtags = [0] * total_hours

        for line in onefile:
            onetweet = json.loads(line)
            current_hour = (onetweet['citation_date'] - start_time) // 3600
            tweetCount[current_hour] += 1
            retweetCount[current_hour] += onetweet['metrics']['citations']['total']
            followerCount[current_hour] += onetweet['author']['followers']
            max_followers[current_hour] = max(max_followers[current_hour], onetweet['author']['followers'])

            url_citations[current_hour] += len(onetweet['tweet']['entities']['urls'])
            hourly_author_set[current_hour].add(onetweet['author']['nick'])
            num_mentions[current_hour] += len(onetweet['tweet']['entities']['user_mentions'])
            ranking_scores[current_hour] += onetweet['metrics']['ranking_score']
            num_hashtags[current_hour] += onetweet['title'].count('#')

        for i in range(0, len(hourly_author_set)):
            num_authors[i] = len(hourly_author_set[i])

        i = 0
        while i < total_hours:
            time_of_day[i] = i % 24
            i += 1

        onefile.close()

        featureset = np.array(
            [tweetCount, retweetCount, followerCount, max_followers, time_of_day, url_citations, num_authors,
             num_mentions, ranking_scores, num_hashtags])
        featureset = featureset.transpose()
        X = featureset[:-1, 0:10]
        y = featureset[1:, 0]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)

        result = stats_api.OLS(y, X).fit()

        print('hashtag: ', hashtag)
        print('mse: ', mse)
        print('t-test results:')
        print(result.summary())
        print('P values: ')
        print(result.pvalues)


def min_three(p_values):
    p_index = list()
    for i in range(3):
        min_value = min(p_values)
        a = p_values.index(min_value)
        p_values[a] = 1
        p_index.append(a)
    return p_index


def Q5():
    data_path = "./"
    feature_list = ["number of tweets",
                    "sum of favourites_count",
                    "max number of favourite_count",
                    "ranking_score",
                    "sum of friends_count"]

    for file_name in os.listdir(data_path):
        number_of_tweets = 0
        time_dict = dict()
        max_time = 0
        min_time = 100000000
        if file_name.endswith(".txt"):
            with open(os.path.join(data_path, file_name), encoding="utf8") as text:
                print(file_name)
                for i, line in enumerate(text):
                    json_object = json.loads(line)
                    number_of_tweets += 1
                    time = json_object['citation_date']
                    pst_tz = pytz.timezone('US/Pacific')
                    time = datetime.datetime.fromtimestamp(time, pst_tz)
                    tmp = (time.month - 1) * 31 * 24 + (time.day - 1) * 24 + time.hour
                    if max_time < tmp:
                        max_time = tmp
                    if min_time > tmp:
                        min_time = tmp
                    if tmp not in time_dict:
                        time_dict.setdefault(tmp, [0, 0, 0, 0, 0])
                    if tmp in time_dict:
                        time_dict[tmp][0] += 1
                        time_dict[tmp][1] += json_object['tweet']['user']['favourites_count']
                        time_dict[tmp][3] += json_object['metrics']['ranking_score']
                        time_dict[tmp][4] += json_object['tweet']['user']['friends_count']
                        if json_object['tweet']['user']['favourites_count'] > time_dict[tmp][2]:
                            time_dict[tmp][2] = json_object['tweet']['user']['favourites_count']
                for i in range(min_time, max_time + 1):
                    if i not in time_dict.keys():
                        time_dict.setdefault(i, [0, 0, 0, 0, 0])

                feature = [value for key, value in time_dict.items()]
                del feature[len(feature) - 1]
                label = [value[0] for key, value in time_dict.items()]
                del label[0]
                x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, random_state=42)
                lr = linear_model.LinearRegression()
                lr.fit(x_train, y_train)
                test_predict = lr.predict(x_test)

                print("RMSE = {:.3f}".format(math.sqrt(mean_squared_error(test_predict, y_test))))

                model = OLS(label, feature)
                results = model.fit()
                print("p_values: {}",format(results.pvalues))

                feature_select = min_three(results.pvalues.tolist())
                print("Three most important features: {}, {}, {}".format(feature_list[feature_select[0]],
                                                                         feature_list[feature_select[1]],
                                                                         feature_list[feature_select[2]]))

                x_test = np.array(x_test)
                fig, axarr = plt.subplots(3, 1)
                for i in range(3):
                    axarr[i].scatter(x_test[:, feature_select[i]], test_predict)
                    axarr[i].set_title("{}: {}".format(file_name, feature_list[feature_select[i]]))
                plt.setp([a.get_yticklabels() for a in axarr], visible=False)
                plt.show()

                print("")


def get_linear_regression_performance(feature, label):
    percent = 0.2
    indices = list(range(len(label)))
    random.shuffle(indices)

    train_index = indices[: int((1 - percent) * len(label))]
    test_index = indices[int((1 - percent) * len(label)) :]
    features_train = [feature[i] for i in train_index]
    labels_train = [label[i] for i in train_index]
    features_test = [feature[i] for i in test_index]
    labels_test = [label[i] for i in test_index]

    if not labels_train or not labels_test:
        return

    # Linear Regression
    lr = linear_model.LinearRegression()
    lr.fit(features_train, labels_train)
    test_predict = lr.predict(features_test)
    mse = mean_squared_error(test_predict, labels_test)
    r2score = r2_score(test_predict, labels_test)
    print("MSE: {:.3f}".format(mse))
    print("R2 Score: {:.3f}\n".format(r2score))


def Q6Q7Q8():
    debug_cnt = 0
    data_path = "./"
    time_dict_aggregate = [dict(), dict(), dict()]

    for file_name in os.listdir(data_path):
        time_dict = [dict(), dict(), dict()]

        if file_name.endswith(".txt"):
            with open(os.path.join(data_path, file_name), encoding="utf8") as text:
                print("=============== {} ===============".format(file_name))

                for i, line in enumerate(text):
                    json_object = json.loads(line)
                    time = json_object['citation_date']
                    pst_tz = pytz.timezone('US/Pacific')
                    time = datetime.datetime.fromtimestamp(time, pst_tz)
                    time_cnt_hour = (time.month - 1) * 31 * 24 + (time.day - 1) * 24 + time.hour
                    time_cnt_5min = ((time.month - 1) * 31 * 24 * 60 + (time.day - 1) * 24 * 60 + time.hour * 60 + time.minute) // 5

                    # judge time period
                    if time_cnt_hour <= 752:
                        idx = 0
                        time_cnt = time_cnt_hour
                    elif 752 < time_cnt_hour <= 764:
                        idx = 1
                        time_cnt = time_cnt_5min
                    else:
                        idx = 2
                        time_cnt = time_cnt_hour

                    # aggregate
                    if time_cnt_hour not in time_dict_aggregate[idx]:
                        time_dict_aggregate[idx].setdefault(time_cnt_hour, [0, 0, 0, 0, 0, 0, 0, 0])
                    else:
                        time_dict_aggregate[idx][time_cnt_hour][0] += 1
                        time_dict_aggregate[idx][time_cnt_hour][1] += json_object['tweet']['user']['favourites_count']
                        if json_object['tweet']['user']['favourites_count'] > time_dict_aggregate[idx][time_cnt_hour][2]:
                            time_dict_aggregate[idx][time_cnt_hour][2] = json_object['tweet']['user']['favourites_count']
                        time_dict_aggregate[idx][time_cnt_hour][3] += json_object['tweet']['user']['friends_count']
                        time_dict_aggregate[idx][time_cnt_hour][4] += json_object['metrics']['ranking_score']
                        time_dict_aggregate[idx][time_cnt_hour][5] += json_object['metrics']['citations']['total']
                        time_dict_aggregate[idx][time_cnt_hour][6] += json_object['author']['followers']

                    # selected features for GO_Hawks, Go_Patriots, Patriots, Superbowl
                    if file_name == 'tweets_#gohawks.txt' or file_name == 'tweets_#gopatriots.txt' or file_name == 'tweets_#patriots.txt' or file_name == 'tweets_#superbowl.txt':
                        if time_cnt not in time_dict[idx]:
                            time_dict[idx].setdefault(time_cnt, [0, 0, 0, 0, 0, 0, 0, 0])
                        else:
                            time_dict[idx][time_cnt][0] += 1
                            time_dict[idx][time_cnt][1] += json_object['tweet']['user']['favourites_count']
                            if json_object['tweet']['user']['favourites_count'] > time_dict[idx][time_cnt][2]:
                                time_dict[idx][time_cnt][2] = json_object['tweet']['user']['favourites_count']
                            time_dict[idx][time_cnt][3] += json_object['tweet']['user']['friends_count']
                            time_dict[idx][time_cnt][4] += json_object['metrics']['ranking_score']
                            time_dict[idx][time_cnt][5] += json_object['metrics']['citations']['total']
                            time_dict[idx][time_cnt][6] += json_object['author']['followers']

                    # selected features for nfl
                    elif file_name == 'tweets_#nfl.txt':
                        if time_cnt not in time_dict[idx]:
                            time_dict[idx].setdefault(time_cnt, [0, 0, 0, 0, 0, 0, 0])
                            time_dict[idx][time_cnt][3] = time.hour
                        else:
                            time_dict[idx][time_cnt][0] += 1
                            time_dict[idx][time_cnt][1] += json_object['metrics']['ranking_score']
                            time_dict[idx][time_cnt][2] += json_object['tweet']['user']['friends_count']
                            time_dict[idx][time_cnt][4] += json_object['tweet']['user']['favourites_count']
                            time_dict[idx][time_cnt][5] += json_object['metrics']['citations']['total']
                            time_dict[idx][time_cnt][6] += json_object['author']['followers']

                    # selected features for sb49
                    elif file_name == 'tweets_#sb49.txt':
                        if time_cnt not in time_dict[idx]:
                            time_dict[idx].setdefault(time_cnt, [0, 0, 0, 0, 0, 0, 0])
                        else:
                            time_dict[idx][time_cnt][0] += 1
                            time_dict[idx][time_cnt][1] += json_object['metrics']['ranking_score']
                            if json_object['tweet']['user']['favourites_count'] > time_dict[idx][time_cnt][2]:
                                time_dict[idx][time_cnt][2] = json_object['tweet']['user']['favourites_count']
                            time_dict[idx][time_cnt][3] += json_object['tweet']['user']['friends_count']
                            time_dict[idx][time_cnt][4] += json_object['tweet']['user']['favourites_count']
                            time_dict[idx][time_cnt][5] += json_object['metrics']['citations']['total']
                            time_dict[idx][time_cnt][6] += json_object['author']['followers']

                    # if debug_cnt >= 1000:
                    #     debug_cnt = 0
                    #     break
                    # else:
                    #     debug_cnt += 1

            feature1 = [value for key, value in time_dict[0].items()]
            feature2 = [value for key, value in time_dict[1].items()]
            feature3 = [value for key, value in time_dict[2].items()]
            if feature1: del feature1[len(feature1) - 1]
            if feature2: del feature2[len(feature2) - 1]
            if feature3: del feature3[len(feature3) - 1]

            label1 = [value[0] for key, value in time_dict[0].items()]
            label2 = [value[0] for key, value in time_dict[1].items()]
            label3 = [value[0] for key, value in time_dict[2].items()]
            if label1: del label1[0]
            if label2: del label2[0]
            if label3: del label3[0]

            print("Before 02/01/8:00")
            get_linear_regression_performance(feature1, label1)
            print("02/01/8:00 to 8:00 PM")
            get_linear_regression_performance(feature2, label2)
            print("After 02/01/8:00 PM")
            get_linear_regression_performance(feature3, label3)

    feature_aggregate1 = [value for key, value in time_dict_aggregate[0].items()]
    feature_aggregate2 = [value for key, value in time_dict_aggregate[1].items()]
    feature_aggregate3 = [value for key, value in time_dict_aggregate[2].items()]
    if feature_aggregate1: del feature_aggregate1[len(feature_aggregate1) - 1]
    if feature_aggregate2: del feature_aggregate2[len(feature_aggregate2) - 1]
    if feature_aggregate3: del feature_aggregate3[len(feature_aggregate3) - 1]

    label_aggregate1 = [value[0] for key, value in time_dict_aggregate[0].items()]
    label_aggregate2 = [value[0] for key, value in time_dict_aggregate[1].items()]
    label_aggregate3 = [value[0] for key, value in time_dict_aggregate[2].items()]
    if label_aggregate1: del label_aggregate1[0]
    if label_aggregate2: del label_aggregate2[0]
    if label_aggregate3: del label_aggregate3[0]

    print("Before 02/01/8:00 (Aggregated)")
    get_linear_regression_performance(feature_aggregate1, label_aggregate1)
    print("02/01/8:00 to 8:00 PM (Aggregated)")
    get_linear_regression_performance(feature_aggregate2, label_aggregate2)
    print("After 02/01/8:00 PM (Aggregated)")
    get_linear_regression_performance(feature_aggregate3, label_aggregate3)

    feature_aggregate = feature_aggregate1 + feature_aggregate2 + feature_aggregate3
    label_aggregate = label_aggregate1 + label_aggregate2 + label_aggregate3
    param_grid = {'max_depth': [10, 20, 40, 60, 80, 100, 200, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    print("\n=========== RandomForestRegressor ===========")
    clf = GridSearchCV(RandomForestRegressor(),
                       param_grid,
                       cv=KFold(5, shuffle=True),
                       scoring='neg_mean_squared_error')
    clf.fit(feature_aggregate, label_aggregate)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("\nGrid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print("")

    print("\n=========== GradientBoostingRegressor ===========")
    clf = GridSearchCV(GradientBoostingRegressor(),
                       param_grid,
                       cv=KFold(5, shuffle=True),
                       scoring='neg_mean_squared_error')
    clf.fit(feature_aggregate, label_aggregate)
    print("Best parameters set found on development set:")
    print(clf.best_params_)
    print("\nGrid scores on development set:")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print("")


def aggregated_data():
    data_path = "ECE219_tweet_data/"
    time_dict_aggregate = [dict(), dict(), dict()]

    for file_name in os.listdir(data_path):
        time_dict = [dict(), dict(), dict()]

        if file_name.endswith(".txt"):
            with open(os.path.join(data_path, file_name), encoding="utf8") as text:
                print("=============== {} ===============".format(file_name))

                for i, line in enumerate(text):
                    json_object = json.loads(line)
                    time = json_object['citation_date']
                    pst_tz = pytz.timezone('US/Pacific')
                    time = datetime.datetime.fromtimestamp(time, pst_tz)
                    time_cnt_hour = (time.month - 1) * 31 * 24 + (time.day - 1) * 24 + time.hour
                    time_cnt_5min = ((time.month - 1) * 31 * 24 * 60 + (time.day - 1) * 24 * 60 + time.hour * 60 + time.minute) // 5

                    # judge time period
                    if time_cnt_hour <= 752:
                        idx = 0
                        time_cnt = time_cnt_hour
                    elif 752 < time_cnt_hour <= 764:
                        idx = 1
                        time_cnt = time_cnt_5min
                    else:
                        idx = 2
                        time_cnt = time_cnt_hour

                    # aggregate
                    if time_cnt_hour not in time_dict_aggregate[idx]:
                        time_dict_aggregate[idx].setdefault(time_cnt_hour, [0, 0, 0, 0, 0, 0, 0, 0])
                    else:
                        time_dict_aggregate[idx][time_cnt_hour][0] += 1
                        time_dict_aggregate[idx][time_cnt_hour][1] += json_object['tweet']['user']['favourites_count']
                        if json_object['tweet']['user']['favourites_count'] > time_dict_aggregate[idx][time_cnt_hour][2]:
                            time_dict_aggregate[idx][time_cnt_hour][2] = json_object['tweet']['user']['favourites_count']
                        time_dict_aggregate[idx][time_cnt_hour][3] += json_object['tweet']['user']['friends_count']
                        time_dict_aggregate[idx][time_cnt_hour][4] += json_object['metrics']['ranking_score']
                        time_dict_aggregate[idx][time_cnt_hour][5] += json_object['metrics']['citations']['total']
                        time_dict_aggregate[idx][time_cnt_hour][6] += json_object['author']['followers']

    feature_aggregate1 = [value for key, value in time_dict_aggregate[0].items()]
    feature_aggregate2 = [value for key, value in time_dict_aggregate[1].items()]
    feature_aggregate3 = [value for key, value in time_dict_aggregate[2].items()]
    if feature_aggregate1: del feature_aggregate1[len(feature_aggregate1) - 1]
    if feature_aggregate2: del feature_aggregate2[len(feature_aggregate2) - 1]
    if feature_aggregate3: del feature_aggregate3[len(feature_aggregate3) - 1]

    label_aggregate1 = [value[0] for key, value in time_dict_aggregate[0].items()]
    label_aggregate2 = [value[0] for key, value in time_dict_aggregate[1].items()]
    label_aggregate3 = [value[0] for key, value in time_dict_aggregate[2].items()]
    if label_aggregate1: del label_aggregate1[0]
    if label_aggregate2: del label_aggregate2[0]
    if label_aggregate3: del label_aggregate3[0]
    return [feature_aggregate1, feature_aggregate2, feature_aggregate3],[label_aggregate1, label_aggregate2, label_aggregate3]

def GradientBoosting_period(feature, label):
    param_grid = {'max_depth': [10, 20, 40, 60, 80, 100, 200, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10],
                  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

    for idx in range(3):
        print(idx)
        clf = GridSearchCV(GradientBoostingRegressor(),
                           param_grid,
                           cv=KFold(5, shuffle=True),
                           scoring='neg_mean_squared_error')
        clf.fit(feature[idx], label[idx])
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("\nGrid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print("")

def Neural_network(X, y):
    parameters = [(50,), (100,), (300,), (100, 50), (50,100)]
    error = []
    for param in parameters:

        model = MLPRegressor(hidden_layer_sizes=param)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        print(mse)
        error.append(mse)

    print("best hidden layer")
    print(parameters[np.argmin(error)])
    return parameters[np.argmin(error)]

def Neural_network_scaler(X, y, param):
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    model = MLPRegressor(hidden_layer_sizes=param)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(mse)
    
def Q13(features,labels):
    param_grid = {'hidden_layer_sizes':[(50,), (100,), (300,), (100, 50),(300,100),(300,50)]}
    scaler = StandardScaler()
    for i in range(3):
        clf = GridSearchCV(MLPRegressor(),
                           param_grid,
                           cv=KFold(5, shuffle=True),
                           scoring='neg_mean_squared_error')
        scaler.fit(features[i])
        fs = scaler.transform(features[i])
        clf.fit(fs, labels[i])
        print("Best parameters set found on development set:")
        print(clf.best_params_)
        print("\nGrid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std ** 2, params))
        print("")
        
def Q14(features,labels):
    scaler = StandardScaler()
    scaler.fit(features[0])
    fs0 = scaler.transform(features[0])
    model0 = MLPRegressor(hidden_layer_sizes=(100,50))
    model0.fit(fs0,labels[0])
    
    scaler = StandardScaler()
    scaler.fit(features[1])
    fs1 = scaler.transform(features[1])
    model1 = MLPRegressor(hidden_layer_sizes=(300,100))
    model1.fit(fs1,labels[1])
    
    scaler = StandardScaler()
    scaler.fit(features[2])
    fs2 = scaler.transform(features[2])
    model2 = MLPRegressor(hidden_layer_sizes=(300,100))
    model2.fit(fs2,labels[2])
    
    datapath = "ECE219_tweet_test/"
    for file in os.listdir(datapath):
        text = open(datapath+file)
        new_feature = np.zeros((7,1))
        for i, lines in enumerate(text):
            json_object = json.loads(lines)
            new_feature[0] += 1
            new_feature[1] += json_object['tweet']['user']['favourites_count']
            if json_object['tweet']['user']['favourites_count'] > new_feature[2]:
                new_feature[2] = json_object['tweet']['user']['favourites_count']
            new_feature[3] += json_object['tweet']['user']['friends_count']
            new_feature[4] += json_object['metrics']['ranking_score']
            new_feature[5] += json_object['metrics']['citations']['total']
            new_feature[6] += json_object['author']['followers']
            
        if 'period1' in file:
            y = model0.predict(new_feature.transpose())
        elif 'period2' in file:
            y = model1.predict(new_feature.transpose())
        elif 'period3' in file:
            y = model2.predict(new_feature.transpose())
        print('The predict number of tweets in the next time window is {0} for file: {1}'.format(y[-1],file))
        
def in_washington(location):
    white_list = ["seattle","washington","wa","kirkland"]
    black_list = ["dc","d.c.","d.c."]

    flag = False

    location = location.split()
    for s in white_list:
        if s in location:
            flag = True
            break
    for s in black_list:
        if s in location:
            flag = False
            break
    return flag

def in_mas(location):
    white_list = ["ma","massachusetts","boston","worcester","salem","plymouth","springfield","arlington","scituate","northampton"]
    black_list = ["ohio",]

    flag = False
    
    location = location.split()
    for s in white_list:
        if s in location:
            flag = True
            break
    for s in black_list:
        if s in location:
            flag = False
            break
    return flag

def preprocess(doc):
    analyzer = CountVectorizer().build_analyzer()
    doc=re.sub(r'[^A-Za-z]', " ", doc)
    return (stemmer.stem(w) for w in analyzer(doc) if w not in combined_stopwords)

def print_statistics(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    recall = recall_score(actual, predicted)
    precision = precision_score(actual, predicted)
    print ('accuracy score is:', accuracy)
    print ('recall score is:', recall)
    print ('precision score is:', precision)
    
    #print("Confusion Matrix is ", smet.confusion_matrix(actual, predicted))
def plot_confusion_matrix(cm, classes, normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "%.4f"%cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_roc(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkgreen',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def classify(X, Y, classifier, cname):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    #b = 0.85 * X.shape[0]
    #X_train = X[:b, :]
    #Y_train = Y[:b]
    #X_test = X[b:, :]
    #Y_test = Y[b:]

    classifier.fit(X_train, Y_train)
    predicted = classifier.predict(X_test)
    predicted_probs = classifier.predict_proba(X_test)

    print_statistics(Y_test, predicted)
    cnf_matrix = confusion_matrix(Y_test, predicted)
    plot_confusion_matrix(cnf_matrix, ['WA','MA'])
    
    fpr = dict()
    tpr = dict()
    fpr, tpr, thresholds = roc_curve(Y_test, predicted)
    plot_roc(fpr, tpr)

def Q15()
    stop_words_skt = text.ENGLISH_STOP_WORDS
    stemmer = SnowballStemmer('english')
    combined_stopwords = set.union(set(stop_words_skt),set(punctuation))
    tweet_content = []
    tweet_labels = []

    input_file = open('ECE219_tweet_data/tweets_#superbowl.txt',encoding='utf-8')
    for (line, index) in zip(input_file, range(0, 1348767)):
        data = json.loads(line)
        location=data.get('tweet').get('user').get('location')
        if in_washington(location):
            tweet_content.append(data.get("title"))
            tweet_labels.append(0)
        elif in_mas(location):
            tweet_content.append(data.get("title"))
            tweet_labels.append(1)

    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([('CountVectorizer', CountVectorizer(min_df=5,analyzer=preprocess,stop_words=combined_stopwords)),
                          ('tf-idf', TfidfTransformer()),('svd', TruncatedSVD(n_components=50))])

    X = pipeline.fit_transform(tweet_content)
    Y = np.array(tweet_labels)
    print("Statistics of SVM classifier:")
    classify(X, Y, svm.SVC(kernel='linear', probability=True), "SVM")

    print("Statistics of LogisticRegression:")
    classify(X, Y, LogisticRegression(), "LogisticRegression")

    print("Statistics of Naive Bayes Classifier:")
    classify(X, Y, BernoulliNB(), "BernoulliNB")    


def get_sent(data,d):           
    
    sum_pos = 0
    sum_neg = 0
    ratio_pos = 0
    ratio_neg = 0
    neg = 0
    pos = 0
    num = 0
    pos_val = []
    neg_val = []
    sid = SentimentIntensityAnalyzer()
    
    for tweet in data:
        num += 1
        ss = sid.polarity_scores(tweet['tweet']['text'])
        for k in ss:
            if k == 'neg':
                neg = ss[k]
            elif k == 'pos':
                pos = ss[k]
        pos_val.append(pos)
        neg_val.append(neg)
        if pos > neg:
            sum_pos += 1
        elif neg > pos:
            sum_neg += 1
            
    data_number = range(num)
    
    area = np.pi*2 
    
    print('######## Scatter Figure of Data-Time set',d,'########')
    print()
    plt.scatter(data_number, pos_val, s=area, label='Positive')
    plt.scatter(data_number, neg_val, s=area, label='Negative')
    plt.title('Scatter of Positive and Negative Value of Sentiments vs Datapoints')
    plt.xlabel('Data Point Number')
    plt.ylabel('Value of Sentiments')
    plt.legend()
    plt.show()

    ratio_pos = float(sum_pos/(sum_pos + sum_neg))
    ratio_neg = float(sum_neg/(sum_pos + sum_neg))
    
    return ratio_pos, ratio_neg, sum_pos, sum_neg, num



def Q16():
    nltk.download('vader_lexicon')
    data_dir = 'ECE219_tweet_data'      # loading dataset

    hashtags = ['gopatriots','gohawks']
    data = {}
    for hashtag in hashtags:
        file_name = data_dir + '/tweets_#' + hashtag + '.txt' 
        with open(file_name, 'r') as f:
            tweets = []
            for i, l in enumerate(f):
                tweet = json.loads(l)
                tweets.append(tweet)
            data[hashtag] = tweets




    pst_tz = pytz.timezone('US/Pacific')

    data_time_0 = []
    data_time_1 = []    # separate tweets in to time periods within two goals
    data_time_2 = []
    data_time_3 = []
    data_time_4 = []
    data_time_5 = []
    data_time_6 = []
    data_time_7 = []
    data_time_8 = []

    for hashtag in ['gopatriots']:
        for tweet in data[hashtag]:

            time = datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz)
            time_in_minute = datetime.datetime(time.year, time.month, time.day,time.hour, time.minute, 0)

            if (time_in_minute >= datetime.datetime(2015,2,1,15,30)) & (time_in_minute < datetime.datetime(2015,2,1,15,58)):
                    data_time_0.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,15,58)) & (time_in_minute < datetime.datetime(2015,2,1,16,10)):
                    data_time_1.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,10)) & (time_in_minute < datetime.datetime(2015,2,1,16,20)):
                    data_time_2.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,20)) & (time_in_minute < datetime.datetime(2015,2,1,16,29)):
                    data_time_3.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,29)) & (time_in_minute < datetime.datetime(2015,2,1,16,36)):
                    data_time_4.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,36)) & (time_in_minute < datetime.datetime(2015,2,1,16,47)):
                    data_time_5.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,45)) & (time_in_minute < datetime.datetime(2015,2,1,17,10)):
                    data_time_6.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,17,10)) & (time_in_minute < datetime.datetime(2015,2,1,17,30)):
                    data_time_7.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,17,30)) & (time_in_minute < datetime.datetime(2015,2,1,17,35)):
                    data_time_8.append(tweet)

    ratio = np.zeros((9,2))
    sum_pos = np.zeros((1,9))
    sum_neg = np.zeros((1,9))
    num = np.zeros((1,9))


    ## Part(1): Show the value of sentiment analysis for each tweet in each time periods in ‘#gopatriots’ and ‘#gohawks’ in a figure.

    ratio[0][0], ratio[0][1], sum_pos[0][0], sum_neg[0][0], num[0][0] = get_sent(data_time_0,0)
    ratio[1][0], ratio[1][1], sum_pos[0][1], sum_neg[0][1], num[0][1] = get_sent(data_time_1,1)
    ratio[2][0], ratio[2][1], sum_pos[0][2], sum_neg[0][2], num[0][2] = get_sent(data_time_2,2)
    ratio[3][0], ratio[3][1], sum_pos[0][3], sum_neg[0][3], num[0][3] = get_sent(data_time_3,3)
    ratio[4][0], ratio[4][1], sum_pos[0][4], sum_neg[0][4], num[0][4] = get_sent(data_time_4,4)
    ratio[5][0], ratio[5][1], sum_pos[0][5], sum_neg[0][5], num[0][5] = get_sent(data_time_5,5)
    ratio[6][0], ratio[6][1], sum_pos[0][6], sum_neg[0][6], num[0][6] = get_sent(data_time_6,6)
    ratio[7][0], ratio[7][1], sum_pos[0][7], sum_neg[0][7], num[0][7] = get_sent(data_time_7,7)
    ratio[8][0], ratio[8][1], sum_pos[0][8], sum_neg[0][8], num[0][8] = get_sent(data_time_8,8)

    ## Part(2):Show the number of positive and negative tweets in each time periods as a plot, and also show the ratio of positive and negative accordingly in a plot.

    t = range(9)
    plt.plot(t, sum_pos[0], label='Positive')
    plt.plot(t, sum_neg[0], label='Negative')
    plt.title('Number of Tweets in gopatriots with Potive & Negative Sentiments vs Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Tweets')
    plt.legend()
    plt.show()

    R1 = []
    R2 = []
    for i in t:
        R1.append(ratio[i][0])
        R2.append(ratio[i][1])  
    plt.plot(t, R1, label='Positive')
    plt.plot(t, R2, label='Negative')
    plt.title('Ratio of Positive Tweets to Negative Tweets in gopatriots vs Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('Ratio of Positive to Negative')
    plt.legend()
    plt.show()




    data_time_0 = []
    data_time_1 = []
    data_time_2 = []
    data_time_3 = []
    data_time_4 = []
    data_time_5 = []
    data_time_6 = []
    data_time_7 = []
    data_time_8 = []

    for hashtag in ['gohawks']:
        for tweet in data[hashtag]:

            time = datetime.datetime.fromtimestamp(tweet['citation_date'], pst_tz)
            time_in_minute = datetime.datetime(time.year, time.month, time.day,time.hour, time.minute, 0)

            if (time_in_minute >= datetime.datetime(2015,2,1,15,30)) & (time_in_minute < datetime.datetime(2015,2,1,15,58)):
                    data_time_0.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,15,58)) & (time_in_minute < datetime.datetime(2015,2,1,16,10)):
                    data_time_1.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,10)) & (time_in_minute < datetime.datetime(2015,2,1,16,20)):
                    data_time_2.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,20)) & (time_in_minute < datetime.datetime(2015,2,1,16,29)):
                    data_time_3.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,29)) & (time_in_minute < datetime.datetime(2015,2,1,16,36)):
                    data_time_4.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,36)) & (time_in_minute < datetime.datetime(2015,2,1,16,45)):
                    data_time_5.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,16,50)) & (time_in_minute < datetime.datetime(2015,2,1,17,10)):
                    data_time_6.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,17,10)) & (time_in_minute < datetime.datetime(2015,2,1,17,40)):
                    data_time_7.append(tweet)
            elif (time_in_minute >= datetime.datetime(2015,2,1,17,40)) & (time_in_minute < datetime.datetime(2015,2,1,17,55)):
                    data_time_8.append(tweet)

    ratio_test = np.zeros((9,2))
    sum_pos_test = np.zeros((1,9))
    sum_neg_test = np.zeros((1,9))
    num_test = np.zeros((1,9))

    ## Part(1): Show the value of sentiment analysis for each tweet in each time periods in ‘#gopatriots’ and ‘#gohawks’ in a figure.

    ratio_test[0][0], ratio_test[0][1], sum_pos_test[0][0], sum_neg_test[0][0], num_test[0][0] = get_sent(data_time_0,0)
    ratio_test[1][0], ratio_test[1][1], sum_pos_test[0][1], sum_neg_test[0][1], num_test[0][1] = get_sent(data_time_1,1)
    ratio_test[2][0], ratio_test[2][1], sum_pos_test[0][2], sum_neg_test[0][2], num_test[0][2] = get_sent(data_time_2,2)
    ratio_test[3][0], ratio_test[3][1], sum_pos_test[0][3], sum_neg_test[0][3], num_test[0][3] = get_sent(data_time_3,3)
    ratio_test[4][0], ratio_test[4][1], sum_pos_test[0][4], sum_neg_test[0][4], num_test[0][4] = get_sent(data_time_4,4)
    ratio_test[5][0], ratio_test[5][1], sum_pos_test[0][5], sum_neg_test[0][5], num_test[0][5] = get_sent(data_time_5,5)
    ratio_test[6][0], ratio_test[6][1], sum_pos_test[0][6], sum_neg_test[0][6], num_test[0][6] = get_sent(data_time_6,6)
    ratio_test[7][0], ratio_test[7][1], sum_pos_test[0][7], sum_neg_test[0][7], num_test[0][7] = get_sent(data_time_7,7)
    ratio_test[8][0], ratio_test[8][1], sum_pos_test[0][8], sum_neg_test[0][8], num_test[0][8] = get_sent(data_time_8,8)

    ## Part(2):Show the number of positive and negative tweets in each time periods as a plot, and also show the ratio of positive and negative accordingly in a plot.

    t = range(9)
    plt.plot(t, sum_pos_test[0], label='Positive')
    plt.plot(t, sum_neg_test[0], label='Negative')
    plt.title('Number of Tweets in gohawks with Potive & Negative Sentiments vs Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('Number of Tweets')
    plt.legend()
    plt.show()

    R1 = []
    R2 = []
    for i in t:
        R1.append(ratio_test[i][0])
        R2.append(ratio_test[i][1])

    plt.plot(t, R1, label='Positive')
    plt.plot(t, R2, label='Negative')
    plt.title('Ratio of Positive Tweets to Negative Tweets in gohawks vs Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('Ratio of Positive to Negative')
    plt.legend()  
    plt.show()




    ## (3)Based on the statistics for the sentiment analysis of tweets for two opponent teams above and the time of goal 
    ##.   in this game, build a linear regression model based on tweets to predict which team will win finally.


    from sklearn import linear_model
    print('##### Result for patriots #####')
    y = np.array([[0,0],
                 [1,0],
                 [0,0],
                 [1,0],
                 [0,0],
                 [0,1],
                 [0,1],
                 [0,1],
                 [1,0]])
    print(y)
    print()

    LR = linear_model.LinearRegression()
    LR.fit(ratio, y)
    y_pred = LR.predict(ratio_test)
    print('##### Prediction Value for Seahawks #####')
    print(y_pred)
    print()

    win_pred = np.zeros((1,9))
    win_true = np.array([1,0,1,0,1,2,2,2,0])
    for i in range(9):
        if (y_pred[i][0] - y_pred[i][1]) >= 0.05:
            win_pred[0][i] = 2
        elif (y_pred[i][1] - y_pred[i][0]) >= 0.05:
            win_pred[0][i] = 0
        else:
            win_pred[0][i] = 1
    print('##### Prediction Result for Seahawks #####')
    print(win_pred)
    print()
    print('##### True Result for Seahawks #####')
    print(win_true)
    print()

    sum = 0
    for i in range(9):
        if win_pred[0][i] == win_true[i]:
            sum += 1

    Accuracy = sum/9
    print('Accuracy of the prediction=')
    print(Accuracy)
    print()

    t = range(9)

    W = []
    for i in t:
        W.append(win_pred[0][i])
    plt.plot(t, win_true, label='True Result')
    plt.plot(t, W, label='Prediction')
    plt.title('Winning Team Prediction and True Result in gohawks vs Time Period')
    plt.xlabel('Time Period')
    plt.ylabel('Winning Team')
    plt.legend()  
    plt.show()


    
if __name__ == '__main__':
    Q1Q2()
    Q3()
    Q4()

    Q5()
    Q6Q7Q8()
    feature, label = aggregated_data()
    GradientBoosting_period(feature, label)
    feature_all = feature[0] + feature[1] + feature[2]
    label_all = label[0] + label[1] + label[2]
    param = Neural_network(feature_all, label_all)
    Neural_network_scaler(feature_all, label_all, param)
    
    Q13(feature,label)
    Q14(feature,label)
    Q15()
    Q16()


