import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, KFold
from sklearn import linear_model
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.feature_selection import f_regression


def cross_val( X, y, output=False):
    kf = KFold(n_splits=10)

    # squre root errors sr_test and sr_train
    err_test = np.array([])
    err_train = np.array([])
    best_test_rmse=999
    for train_index, test_index in kf.split(X):
        model = linear_model.LinearRegression()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        t1=sqrt(mean_squared_error(y_test, y_pred_test))
        t2=sqrt(mean_squared_error(y_train, y_pred_train))
        
        err_test = np.append(err_test, t1)
        err_train = np.append(err_train, t2)
        if t1<best_test_rmse:
            best_test_rmse=t1
            best_model=model

    rmse_test = np.sum(err_test)/err_test.size
    rmse_train = np.sum(err_train)/err_train.size
    print('Average test RMSE: %s ; Average training RMSE: %s' % (rmse_test, rmse_train))
    return rmse_test, rmse_train, best_model

# model.coef_

def cross_val1(model, X, y, output=True):
    kf = KFold(n_splits=10)

    # squre root errors sr_test and sr_train
    err_test = np.array([])
    err_train = np.array([])
    best_test_rmse=999
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)
        t1=sqrt(mean_squared_error(y_test, y_pred_test))
        t2=sqrt(mean_squared_error(y_train, y_pred_train))
        
        err_test = np.append(err_test, t1)
        err_train = np.append(err_train, t2)
        if t1<best_test_rmse:
            best_test_rmse=t1
            best_coeff=model.coef_

    rmse_test = np.sum(err_test)/err_test.size
    rmse_train = np.sum(err_train)/err_train.size
    if output:
        print('Average test RMSE: %s ; Average training RMSE: %s' % (rmse_test, rmse_train))
    return rmse_test, rmse_train, best_coeff

def scatter_plot(x, y, xlabel=None, ylabel=None, title=None):
    #plt.scatter(x, y, s=1, marker='.')
    plt.scatter(x, y)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.show()
    
def find_best_para(regu, alpha, ratio=[-1]):
    min_test = 999
    min_train = 999 
    best_ar = 999 
    best_ra = 999 
    best_coef = [] 
    if len(ratio) == 1:
        ratio_list = [1]
    else:
        ratio_list = ratio
    for ra in ratio_list:
        for ar in alpha:
            if (regu == 'Ridge'):
                model = Ridge(alpha = ar)
            elif (regu == 'Lasso'):
                model = Lasso(alpha = ar)
            elif (regu == 'Elastic Net'):
                model = ElasticNet(alpha = ar, l1_ratio = ra)

            rmse_test, rmse_train, coeff = cross_val1(model, X, y, False)
            #print(ar,rmse_test)

            if rmse_test < min_test:
                min_test = rmse_test 
                min_train = rmse_train
                best_ar = ar 
                best_ra = ra 
                best_coef = coeff 
    print ('Regularization method: '+ regu)
    print ('Best parameter Test RMSE: ',min_test)
    print ('Best parameter Train RMSE: ',min_train)
    print ('Best alpha: ', best_ar)
    if len(ratio)!=1:
        print ('Best l1_Ratio: ',best_ra)
        print ('Best lambda1: ',best_ar*best_ra)
        print ('Best lambda2: ',0.5*best_ar*(1-best_ra))
    print ('Estimated coefficients: ', best_coef)
    return min_test, best_ar

if __name__ == "__main__":
    #2
    data1 = pd.read_csv("housing_data.csv")
    X = data1.iloc[:,0:13].values
    y = data1.iloc[:,13].values

    F, p = f_regression(X,y)
    print('F', F)
    print('p',p)
    #lr = linear_model.LinearRegression()
    rmse_test_, rmse_train_, best_model = cross_val( X, y)

    y_predict = best_model.predict(X)

    title = 'fitted values against true values'
    scatter_plot(y, y_predict,'true values','fitted values', title=title)

    y_residual = y - y_predict
    title = 'residuals versus fitted values'
    scatter_plot(y_predict, y_residual, 'fitted values','residuals',title=title)

    #3
    alpha = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    ratio = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    regu = 'Ridge'
    find_best_para(regu, alpha)
    regu = 'Lasso'
    find_best_para(regu, alpha)
    regu = 'Elastic Net'
    find_best_para(regu, alpha, ratio)
    print('Estimated coefficients of unregularized best model: ', best_model.coef_)
