def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression, mutual_info_regression, SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures


def progressBar(start, end, bar_length=20):
    percent = float(start) / end
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r[{0}] {1}% [{2}/{3}]".format(arrow + spaces, int(round(percent * 100)), start, end))
    sys.stdout.flush()


def get_dataset3(path):
    dataset_frame = pd.read_csv(path)
    X = dataset_frame.iloc[:, :6].values
    y = dataset_frame.iloc[:, 6].values

    X[:, 3] = one_hot_encode(X[:, 3])
    X[:, 4] = one_hot_encode(X[:, 4])
    X[:, 5] = one_hot_encode(X[:, 5])

    return X, y


def encode_dataset(X, should_standardize=False):
    X_encoded = list()
    for i in range(len(X)):
        row = list()
        for feature in X[i]:
            if isinstance(feature, list):
                row.extend(feature)
            else:
                row.append(feature)
        X_encoded.append(row)

    X_encoded = np.array(X_encoded)
    if should_standardize:
        x_scaler = StandardScaler()
        X_encoded[:, :3] = x_scaler.fit_transform(X_encoded[:, :3])
    return X_encoded


def discretize_dataset(X):
    X_encoded = list()
    for i in range(len(X)):
        row = list()
        if X[i][0] < 30:
            row.append(1)
        elif 30 <= X[i][0] <= 50:
            row.append(2)
        else:
            row.append(3)
        row.append(X[i][1])
        row.append(X[i][2])
        row.extend(X[i][3])
        row.extend(X[i][4])
        row.extend(X[i][5])
        X_encoded.append(row)

    X_encoded = np.array(X_encoded)
    x_scaler = StandardScaler()
    X_encoded[:, 1:3] = x_scaler.fit_transform(X_encoded[:, 1:3])
    return X_encoded


def categorize_dataset(path):
    dataset_frame = pd.read_csv(path)
    X = dataset_frame.iloc[:, :6].values
    X[:, 3] = categorize_encode(X[:, 3])
    X[:, 4] = categorize_encode(X[:, 4])
    X[:, 5] = categorize_encode(X[:, 5])
    return X


def categorize_encode(data):
    cnt = 0
    mapping = dict()
    for category in set(data):
        mapping[category] = cnt
        cnt += 1

    data_encoded = list()
    for d in data:
        data_encoded.append(mapping[d])
    return data_encoded


def one_hot_encode(data):
    cnt = 0
    mapping = dict()
    for category in set(data):
        mapping[category] = cnt
        cnt += 1

    data_encoded = [[0] * cnt for _ in range(len(data))]
    for i, d in enumerate(data):
        idx = mapping[d]
        data_encoded[i][idx] = 1
    return data_encoded


def print_performance(model, X, y, title, shuffle=False, verbose=True):
    kf = KFold(n_splits=10, shuffle=shuffle)

    SRE_test = np.array([])
    SRE_train = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_test_hat = model.predict(X_test)
        y_train_hat = model.predict(X_train)
        SRE_test_curr = np.power(np.array(y_test) - np.array(y_test_hat), 2)
        SRE_test = np.append(SRE_test, SRE_test_curr)
        SRE_train_curr = np.power(np.array(y_train) - np.array(y_train_hat), 2)
        SRE_train = np.append(SRE_train, SRE_train_curr)

    RMSE_test = sqrt(np.sum(SRE_test) / SRE_test.size)
    RMSE_train = sqrt(np.sum(SRE_train) / SRE_train.size)
    if verbose:
        print("========= {} =========".format(title))
        print("RMSE training: {:.3f}".format(RMSE_train))
        print("RMSE testing: {:.3f}".format(RMSE_test))

    model.fit(X, y)
    y_hat = model.predict(X)
    if verbose:
        plt.figure()
        plt.scatter(list(range(len(y_hat))), y_hat)
        plt.scatter(list(range(len(y))), y)
        plt.legend(["Fitted charges", "True charges"])
        plt.title(title + ": Fitted charges against true charges")
        plt.xlabel("Data Index")
        plt.ylabel("Charges")
        plt.grid()
        plt.show()

    y_residual = y - y_hat
    if verbose:
        plt.figure()
        plt.scatter(list(range(len(y_hat))), y_hat)
        plt.scatter(list(range(len(y_residual))), y_residual)
        plt.legend(["Residual charges", "Fitted charges"])
        plt.title(title + ": Residual charges against fitted charges")
        plt.xlabel("Data Index")
        plt.ylabel("Charges")
        plt.grid()
        plt.show()

    return RMSE_test


def print_performance_log(model, X, y, title, shuffle=False, verbose=True):
    y_log = np.log(y)
    kf = KFold(n_splits=10, shuffle=shuffle)

    SRE_test = np.array([])
    SRE_train = np.array([])
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train_log, y_test_log = y_log[train_index], y_log[test_index]
        model.fit(X_train, y_train_log)
        y_test_hat = model.predict(X_test)
        y_train_hat = model.predict(X_train)
        SRE_test_curr = np.power(np.array(y_test) - np.exp(np.array(y_test_hat)), 2)
        SRE_test = np.append(SRE_test, SRE_test_curr)
        SRE_train_curr = np.power(np.array(y_train) - np.exp(np.array(y_train_hat)), 2)
        SRE_train = np.append(SRE_train, SRE_train_curr)

    RMSE_test = sqrt(np.sum(SRE_test) / SRE_test.size)
    RMSE_train = sqrt(np.sum(SRE_train) / SRE_train.size)
    if verbose:
        print("========= {} =========".format(title))
        print("RMSE training: {:.3f}".format(RMSE_train))
        print("RMSE testing: {:.3f}".format(RMSE_test))

    model.fit(X, y_log)
    y_hat = model.predict(X)
    if verbose:
        plt.figure()
        plt.scatter(list(range(len(y_hat))), np.exp(np.array(y_hat)))
        plt.scatter(list(range(len(y))), y)
        plt.legend(["Fitted charges", "True charges"])
        plt.title(title + ": Fitted charges against true charges")
        plt.xlabel("Data Index")
        plt.ylabel("Charges")
        plt.grid()
        plt.show()

    y_residual = y - np.exp(np.array(y_hat))
    if verbose:
        plt.figure()
        plt.scatter(list(range(len(y_hat))), np.exp(np.array(y_hat)))
        plt.scatter(list(range(len(y_residual))), y_residual)
        plt.legend(["Residual charges", "Fitted charges"])
        plt.title(title + ": Residual charges against fitted charges")
        plt.xlabel("Data Index")
        plt.ylabel("Charges")
        plt.grid()
        plt.show()

    return RMSE_test


if __name__ == "__main__":
    # Dataset 3/1/(a)
    X, y = get_dataset3("./data/insurance_data.csv")
    X_1a = encode_dataset(X, should_standardize=False)
    model = linear_model.LinearRegression(normalize=True)
    print_performance(model, X_1a, y, "Dataset3/1/(a)")

    # (b)
    X_1b = encode_dataset(X, should_standardize=True)
    model = linear_model.LinearRegression(normalize=False)
    print_performance(model, X_1b, y, "Dataset3/1/(b)")

    # (c)
    X_1c = discretize_dataset(X)
    model = linear_model.LinearRegression(normalize=False)
    print_performance(model, X_1c, y, "Dataset3/1/(c)")

    # 2/(a)
    X_2a = categorize_dataset("./data/insurance_data.csv")
    F, p = f_regression(X_2a, y)
    mi = mutual_info_regression(X_2a, y)
    print("========= {} =========".format("Dataset3/2/(a)"))
    print("F: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(*F))
    print("Mutual information: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(*mi))

    X_reg = SelectKBest(f_regression, k=2).fit_transform(X_2a, y)
    X_mi = SelectKBest(mutual_info_regression, k=2).fit_transform(X_2a, y)
    print("Feature 1 and 5 are selected")

    # (b)
    plt.figure()
    X_2b_1 = list()
    y_2b_1 = list()
    X_2b_2 = list()
    y_2b_2 = list()
    for f2, f5, y_cur in zip(X_2a[:, 1], X_2a[:, 4], y):
        if f5 == 0:
            X_2b_1.append(f2)
            y_2b_1.append(y_cur)
        else:
            X_2b_2.append(f2)
            y_2b_2.append(y_cur)
    plt.scatter(X_2b_1, y_2b_1)
    plt.scatter(X_2b_2, y_2b_2)
    plt.legend(["When feature 5 equals 0", "When feature 5 equals 1"])
    plt.title("Dataset3/2/(b): Charges against feature 2 based on feature 5")
    plt.xlabel("Feature Values")
    plt.ylabel("Charges")
    plt.grid()
    plt.show()

    # (c)
    plt.figure()
    X_2c_1 = list()
    y_2c_1 = list()
    X_2c_2 = list()
    y_2c_2 = list()
    for f1, f5, y_cur in zip(X_2a[:, 0], X_2a[:, 4], y):
        if f5 == 0:
            X_2c_1.append(f1)
            y_2c_1.append(y_cur)
        else:
            X_2c_2.append(f1)
            y_2c_2.append(y_cur)
    plt.scatter(X_2c_1, y_2c_1)
    plt.scatter(X_2c_2, y_2c_2)
    plt.legend(["When feature 5 equals 0", "When feature 5 equals 1"])
    plt.title("Dataset3/2/(c): Charges against feature 1 based on feature 5")
    plt.xlabel("Feature Values")
    plt.ylabel("Charges")
    plt.grid()
    plt.show()

    # 3/(a)
    model = linear_model.LinearRegression(normalize=False)
    print_performance_log(model, X_1b, y, "Dataset3/3/(a)")

    # (b)
    y_3b = np.log(y)
    F, p = f_regression(X_2a, y_3b)
    mi = mutual_info_regression(X_2a, y_3b)
    print("========= {} =========".format("Dataset3/3/(b)"))
    print("F: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(*F))
    print("Mutual information: {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(*mi))

    X_reg = SelectKBest(f_regression, k=2).fit_transform(X_2a, y_3b)
    X_mi = SelectKBest(mutual_info_regression, k=2).fit_transform(X_2a, y_3b)
    print("Feature 1 and 5 are selected")

    plt.figure()
    X_3b_1 = list()
    y_3b_1 = list()
    X_3b_2 = list()
    y_3b_2 = list()
    for f2, f5, y_cur in zip(X_2a[:, 1], X_2a[:, 4], y):
        if f5 == 0:
            X_3b_1.append(f2)
            y_3b_1.append(y_cur)
        else:
            X_3b_2.append(f2)
            y_3b_2.append(y_cur)
    plt.scatter(X_3b_1, y_3b_1)
    plt.scatter(X_3b_2, y_3b_2)
    plt.legend(["When feature 5 equals 0", "When feature 5 equals 1"])
    plt.title("Dataset3/3/(b): Charges against feature 2 based on feature 5")
    plt.xlabel("Feature Values")
    plt.ylabel("Charges")
    plt.grid()
    plt.show()

    plt.figure()
    X_3c_1 = list()
    y_3c_1 = list()
    X_3c_2 = list()
    y_3c_2 = list()
    for f1, f5, y_cur in zip(X_2a[:, 0], X_2a[:, 4], y):
        if f5 == 0:
            X_3c_1.append(f1)
            y_3c_1.append(y_cur)
        else:
            X_3c_2.append(f1)
            y_3c_2.append(y_cur)
    plt.scatter(X_3c_1, y_3c_1)
    plt.scatter(X_3c_2, y_3c_2)
    plt.legend(["When feature 5 equals 0", "When feature 5 equals 1"])
    plt.title("Dataset3/3/(b): Charges against feature 1 based on feature 5")
    plt.xlabel("Feature Values")
    plt.ylabel("Charges")
    plt.grid()
    plt.show()

    # 4/(a)
    poly = PolynomialFeatures(degree=2, interaction_only=False)
    X_4a_new = poly.fit_transform(X_2a[:, [0, 4]])[:, 3:]
    X_4a = np.column_stack((X_1a, X_4a_new))
    model = linear_model.LinearRegression(normalize=False)
    print_performance(model, X_4a, y, "Dataset3/4/(a)")

    # (b)
    estimators = range(1, 201, 5)
    max_features = range(1, 6, 2)
    e_best = None
    f_best = None
    RMSE_test_best = float('inf')
    for i, e in enumerate(estimators):
        progressBar(i + 1, len(estimators))
        for f in max_features:
            model_forest = RandomForestRegressor(n_estimators=e,
                                                 max_depth=4,
                                                 bootstrap=True,
                                                 max_features=f,
                                                 oob_score=True)
            RMSE_test_cur = print_performance(model_forest, X_1b, y, "Dataset3/4/(b)/RF", verbose=False)
            if RMSE_test_cur < RMSE_test_best:
                RMSE_test_best = RMSE_test_cur
                e_best = e
                f_best = f
    print()
    model_forest = RandomForestRegressor(n_estimators=e_best,
                                         max_depth=4,
                                         bootstrap=True,
                                         max_features=f_best,
                                         oob_score=True)
    print_performance(model_forest, X_1b, y, "Dataset3/4/(b)/RF")
    print("Best number of estimators: {}".format(e_best))
    print("Best number of max features: {}".format(f_best))

    activation = ['relu']
    num_hidden = [5, 10, 250]
    a_best = None
    h_best = None
    RMSE_test_best = float('inf')
    for i, h in enumerate(num_hidden):
        progressBar(i + 1, len(num_hidden))
        for a in activation:
            model_neural = MLPRegressor((h,), activation=a)
            RMSE_test_cur = print_performance_log(model_neural, X_1b, y, "Dataset3/4/(b)/Neural", verbose=False)
            if RMSE_test_cur < RMSE_test_best:
                RMSE_test_best = RMSE_test_cur
                a_best = a
                h_best = h
    print()
    model_neural = MLPRegressor((h_best,), activation=a_best)
    print_performance_log(model_neural, X_1b, y, "Dataset3/4/(b)/Neural")
    print("Best activation: {}".format(a_best))
    print("Best number of hidden layers: {}".format(h_best))

    neighbors = range(1, 51)
    n_best = None
    RMSE_test_best = float('inf')
    for i, n in enumerate(neighbors):
        progressBar(i + 1, len(neighbors))
        model_knn = KNeighborsRegressor(n_neighbors=n)
        RMSE_test_cur = print_performance(model_knn, X_1b, y, "Dataset3/4/(b)/KNN", verbose=False)
        if RMSE_test_cur < RMSE_test_best:
            RMSE_test_best = RMSE_test_cur
            n_best = n
    print()
    model_knn = KNeighborsRegressor(n_neighbors=n_best)
    print_performance(model_knn, X_1b, y, "Dataset3/4/(b)/KNN")
    print("Best number of neighbors: {}".format(n_best))


