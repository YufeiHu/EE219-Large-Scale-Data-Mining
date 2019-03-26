import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn import tree
import graphviz
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image 
import warnings

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from math import sqrt
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

class Data:
    
    def __init__(self,path = None,usecols = []):
        if not path:
            self.data = None
        else:
            if not usecols:
                self.data = pd.read_csv(path)
            else:
                self.data = pd.read_csv(path,usecols=usecols)
    
def find_days(week,day):
    if day == 'Monday':
        return week*7 + 1
    elif day == 'Tuesday':
        return week*7 + 2
    elif day == 'Wednesday':
        return week*7 + 3
    elif day == 'Thursday':
        return week*7 + 4
    elif day == 'Friday':
        return week*7 + 5
    elif day == 'Saturday':
        return week*7 + 6
    else:
        return week*7 + 7

def day_to_one_hot(day):
    if day == 'Monday':
        return [1,0,0,0,0,0,0]
    elif day == 'Tuesday':
        return [0,1,0,0,0,0,0]
    elif day == 'Wednesday':
        return [0,0,1,0,0,0,0]
    elif day == 'Thursday':
        return [0,0,0,1,0,0,0]
    elif day == 'Friday':
        return [0,0,0,0,1,0,0]
    elif day == 'Saturday':
        return [0,0,0,0,0,1,0]
    else:
        return [0,0,0,0,0,0,1]
    
def day_to_one_value(day):
    if day == 'Monday':
        return 1
    elif day == 'Tuesday':
        return 2
    elif day == 'Wednesday':
        return 3
    elif day == 'Thursday':
        return 4
    elif day == 'Friday':
        return 5
    elif day == 'Saturday':
        return 6
    else:
        return 7

def generate_feature(data, t = 'value'):
    data_feature = []
    if t == 'value':
        for i in range(len(data.data)):
            feature = []
            d = data.data.iloc[i]
            feature += [day_to_one_value(d['Day of Week'])]
            feature += [d['Backup Start Time - Hour of Day']]
            feature += [int(d['Work-Flow-ID'][-1])]
            feature += [int(d['File Name'][-1])] if len(d['File Name']) == 6 else [int(d['File Name'][5:])]
            feature += [d['Week #']]
            data_feature.append(feature)
    elif t == 'onehot':
        for i in range(len(data.data)):
            feature = []
            d = data.data.iloc[i]
            feature += day_to_one_hot(d['Day of Week'])
            feature += [d['Backup Start Time - Hour of Day']]
            feature += [int(d['Work-Flow-ID'][-1])]
            feature += [int(d['File Name'][-1])] if len(d['File Name']) == 6 else [int(d['File Name'][5:])]
            feature += [d['Week #']]
            data_feature.append(feature)
    return np.array(data_feature)

def find_best_model(X,y,model = LinearRegression()):
    kf = KFold(n_splits=10, shuffle=True)
    min_test_rmse = float("inf")
    kf_train_rmse = []
    kf_test_rmse = []
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lm = model
        lm.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(lm.predict(X_train), y_train))
        test_rmse = np.sqrt(mean_squared_error(lm.predict(X_test), y_test))
        kf_train_rmse.append(train_rmse)
        kf_test_rmse.append(test_rmse)
        
        if test_rmse < min_test_rmse:
            min_test_rmse = test_rmse
            best_model = lm
    return best_model, kf_train_rmse, kf_test_rmse

def plot_scatter(targets, best_model_target):
    plt.scatter(targets, best_model_target, s=5)
    plt.xlabel("True values")
    plt.ylabel("Fitted values")
    plt.title("Fitted values vs True values")
    plt.show()

    plt.scatter(best_model_target,targets-best_model_target,  s=5)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals values")
    plt.title("Residual values vs Fitted values")
    plt.show()


if __name__ == '__main__':    
    
    file = Data('network_backup_dataset.csv')   
    
    """1. Load The Data"""
    #(a) 20 days' data
    i, w, dw, lis = 0, 3, 'Sunday', []
    while i < len(file.data):
        data = file.data.iloc[i]
        if data['Week #'] == w and data['Day of Week'] == dw:
            break
        days = find_days(data['Week #'], data['Day of Week'])
        lis.append([days+data['Backup Start Time - Hour of Day']/6,data['Size of Backup (GB)'], int(data['Work-Flow-ID'][-1])])
        i += 1
    df = pd.DataFrame(lis,columns=['Days','Size','WF'])
    print('Scatter plot for 20-day period:')
    plt.figure(figsize=(30, 10))
    colormap = cm.viridis
    for i in range(len(set(df['WF']))):

        x = df['Days'][df['WF'] == i]
        y = df['Size'][df['WF'] == i]

        plt.scatter(x = x, y = y,label = i, linewidth=0.1, c=colors.rgb2hex(colormap(i * 50)))
    plt.legend(prop={'size':15})
    plt.show()
    
    
    #(b) 105 days' data
    i, w, dw, lis = 0, 16, 'Monday', []
    while i < len(file.data):
        data = file.data.iloc[i]
        if data['Week #'] == w and data['Day of Week'] == dw:
            break
        days = find_days(data['Week #'], data['Day of Week'])
        lis.append([days+data['Backup Start Time - Hour of Day']/6,data['Size of Backup (GB)'], int(data['Work-Flow-ID'][-1])])
        i += 1
    df = pd.DataFrame(lis,columns=['Days','Size','WF'])
    print('Scatter plot for 105-day period:')
    plt.figure(figsize=(30, 10))
    colormap = cm.viridis
    for i in range(len(set(df['WF']))):

        x = df['Days'][df['WF'] == i]
        y = df['Size'][df['WF'] == i]

        plt.scatter(x = x, y = y,label = i, linewidth=0.1, c=colors.rgb2hex(colormap(i * 50)))
    plt.legend(prop={'size':15})
    plt.show()

    """2. Predict"""

    data = Data('network_backup_dataset.csv',usecols=['Day of Week','Backup Start Time - Hour of Day','Work-Flow-ID','File Name',
                                              'Week #'])
    label = Data('network_backup_dataset.csv',usecols=['Size of Backup (GB)'])
    warnings.filterwarnings('ignore')
    #(a)Linear Regression
    data_feature = generate_feature(data=data, t = 'value')
    targets = np.array(label.data['Size of Backup (GB)'].tolist())
    reg = LinearRegression()
    # result = cross_validate(reg, X = data_feature, y= targets, scoring= ['neg_mean_squared_error'],cv= 10,return_train_score=True)
    best_model, kf_train_rmse, kf_test_rmse = find_best_model(data_feature, targets, reg)
    print('The Root Mean Squared Error for training across ten fold is',kf_train_rmse)
    print('The Root Mean Squared Error for testing across ten fold is',kf_test_rmse)

    best_model_target = best_model.predict(data_feature)
    plt.scatter(targets, best_model_target, s=5)
    plt.xlabel("True values")
    plt.ylabel("Fitted values")
    plt.title("Fitted values vs True values")
    plt.show()

    plt.scatter(best_model_target,targets-best_model_target,  s=5)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals values")
    plt.title("Residual values vs Predicted values")
    plt.show()

    #(b)Random Forest Regression
    #(i) 
    rf_train_rmse = []
    rf_test_rmse = []
    rf_oob = []
    forest =RandomForestRegressor(n_estimators=20,
                    oob_score=True,bootstrap = True,
                                   max_features=5,max_depth=4,
                                   random_state=42)
    kf = KFold(n_splits=10, shuffle=True)
    #kfold validation
    for train_index, test_index in kf.split(data_feature):

        X_train, X_test = data_feature[train_index], data_feature[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        forest.fit(X_train, y_train)
        train_rmse = np.sqrt(mean_squared_error(forest.predict(X_train), y_train))
        test_rmse = np.sqrt(mean_squared_error(forest.predict(X_test), y_test))
        rf_train_rmse.append(train_rmse)
        rf_test_rmse.append(test_rmse)
        rf_oob.append(1-forest.oob_score_)

    #find average testin rmse
    ave_test_rmse =np.sqrt(np.sum(np.square(rf_test_rmse)/10))
    print('The training error is',train_rmse)
    print('The average testing error is',ave_test_rmse)
    print('The Out of Bag error is',rf_oob)

    #(ii)
    i = 0
    oob = np.zeros((200,5))
    ave_test_error = np.zeros((200,5))
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(data_feature):
        i += 1
        X_train, X_test = data_feature[train_index], data_feature[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        for n1 in range(200):
            for n2 in range(5):
                forest1 = RandomForestRegressor(n_estimators=n1+1,
                    oob_score=True,bootstrap = True,
                                   max_features=n2+1,max_depth=4)
                forest1.fit(X_train,y_train)
                rmse_test = mean_squared_error(forest1.predict(X_test), y_test)
                ave_test_error[n1,n2] += rmse_test
                oob[n1,n2] += 1-forest1.oob_score_
    #     print('Number of trees {0}, Number of maximum features {1} is done.'.format(n1+1,n2+1))
        print('Cross Validation {} is done!'.format(i))


    #plot average test error
    ave_test_error = np.sqrt(ave_test_error/10)
    for i in range(5):
        plt.plot(ave_test_error[:,i])
    plt.xlabel('Number of Trees')
    plt.ylabel('Average testing error')
    plt.title('Average testing Error vs Number of Trees')
    plt.legend(('Max num of features = 1','Max num of features = 2','Max num of features = 3','Max num of features = 4','Max num of features = 5'))
    plt.show()


    #plot out of bag error
    oob /= 10
    for i in range(5):
        plt.plot(oob[:,i])
    plt.xlabel('Number of Trees')
    plt.ylabel('Out Of Bag Error')
    plt.title('Out Of Bag Error vs Number of Trees')
    plt.legend(('Max num of features = 1','Max num of features = 2','Max num of features = 3','Max num of features = 4','Max num of features = 5'))
    plt.show()

    #(iii)
    print('Sweep for depth of each tree from 1 to 30 by using number of trees = 25 and maximum num of features = 3')
    oob = np.zeros((30,))
    ave_test_error = np.zeros((30,))
    kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in kf.split(data_feature):

        X_train, X_test = data_feature[train_index], data_feature[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        for n1 in range(30):
            forest1 = RandomForestRegressor(n_estimators=25,
                oob_score=True,bootstrap = True,
                               max_features=3,max_depth=n1+1)
            forest1.fit(X_train,y_train)
            rmse_test = mean_squared_error(forest1.predict(X_test), y_test)
            ave_test_error[n1] += rmse_test
            oob[n1] += 1-forest1.oob_score_

        #plot average test error
    ave_test_error = np.sqrt(ave_test_error/10)
    plt.plot(ave_test_error)
    plt.xlabel('Maximum depth of trees')
    plt.ylabel('Average testing error')
    plt.title('Average testing Error vs Maximum depth of Trees')
    plt.show()


    #plot out of bag error
    oob /= 10
    plt.plot(oob)
    plt.xlabel('Maximum depth of trees')
    plt.ylabel('Out Of Bag Error')
    plt.title('Out Of Bag Error vs Maximum depth of Trees')
    plt.show()

    print('When maximum depth of trees = {0}, it has the lowest average testing error.'.format(np.argmin(ave_test_error)+1))

    #(iv)
    print('Using hyperparameter as following: \nNumber of trees: 25\nMaximum number of features: 3\nMaximum depth of trees: 7')
    forest = RandomForestRegressor(n_estimators=25,
        oob_score=True,bootstrap = True,
                       max_features=3,max_depth=7)
    best_model, rf_train_rmse, rf_test_rmse = find_best_model(data_feature, targets, forest)
    print('The feature importance is ',best_model.feature_importances_)

    print('The Root Mean Squared Error for training across ten fold is',rf_train_rmse)
    print('The Root Mean Squared Error for testing across ten fold is',rf_test_rmse)

    best_model_target = best_model.predict(data_feature)
    plt.scatter(targets, best_model_target, s=5)
    plt.xlabel("True values")
    plt.ylabel("Fitted values")
    plt.title("Fitted values vs True values")
    plt.show()

    plt.scatter(best_model_target,targets-best_model_target,  s=5)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals values")
    plt.title("Residual values vs Predicted values")
    plt.show()
    
    #(v)
    dtree=DecisionTreeRegressor(max_depth=4,max_features=3)
    dtree.fit(data_feature,targets)
    dot_data = StringIO()

    export_graphviz(dtree, out_file=dot_data,  
                    feature_names=['Day of Week','Backup Start Time - Hour of Day','Work-Flow-ID','File Name',
                                              'Week #'], 
                                    class_names='Size of Backup (GB)',   filled=True, rounded=True,  special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    Image(graph.create_png())

    ## c
    data_feature = generate_feature(data=data, t = 'onehot')
    targets = np.array(label.data['Size of Backup (GB)'].tolist())

    hiddens = [2, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    activations = ['relu', 'logistic', 'tanh']
    nn_test = { 'relu': [],
                  'logistic': [],
                  'tanh': []
                }
    nn_train = { 'relu': [],
                   'logistic': [],
                   'tanh': []
                 }
    best_nn = { 'relu': [],
               'logistic': [],
               'tanh': []
             }
    for a in activations:
        for n in hiddens:
            print(a, n)
            nn = MLPRegressor((n,), activation=a)
            best_model, nn_train_rmse, nn_test_rmse = find_best_model(data_feature, targets, nn)
            best_nn[a].append(best_model)
            nn_test[a].append(np.sum(nn_test_rmse)/len(nn_test_rmse))
            nn_train[a].append(np.sum(nn_train_rmse)/len(nn_train_rmse))

    for a in activations:
        plt.plot(hiddens, nn_test[a], label = a)

    plt.xlabel("hidden units")
    plt.ylabel("test-RMSE")
    plt.legend()
    plt.title("The test-RMSE vs the number of hidden units for different activation function")
    plt.show()

    best_idx = np.argmin(nn_test['relu'])
    best_h = hiddens[best_idx]
    print(best_h)
    print(nn_test['relu'][best_idx])
    print(nn_train['relu'][best_idx])

    best_model = best_nn['relu'][best_idx]
    best_model_target = best_model.predict(data_feature)
    plot_scatter(targets, best_model_target)

    ## d(1)
    data_feature = generate_feature(data=data, t = 'value')
    targets = np.array(label.data['Size of Backup (GB)'].tolist())

    X = data_feature
    y = targets
    Xy = np.concatenate((X,np.array([y]).T),axis=1)
    group_by_workflow = {}
    for x in Xy:
        wf = x[2]
        if wf not in group_by_workflow:
            group_by_workflow[wf] = np.array([x])
        else:
            group_by_workflow[wf] = np.concatenate(
            (group_by_workflow[wf], [x]),axis=0)
    for key in group_by_workflow:
        print(key)
        Xy_wf = group_by_workflow[key]
        data_feature = Xy_wf[:,[0,1,3,4]]
        targets = Xy_wf[:,5]
        reg = LinearRegression()
        best_model, kf_train_rmse, kf_test_rmse = find_best_model(data_feature, targets, reg)
        print(np.sum(kf_train_rmse)/len(kf_train_rmse))
        print(np.sum(kf_test_rmse)/len(kf_test_rmse))

        best_model_target = best_model.predict(data_feature)
        plot_scatter(targets, best_model_target)

    ## d(2)
    ks = range(4,11)
    for key in group_by_workflow:
        print(key)
        Xy_wf = group_by_workflow[key]
        data_feature = Xy_wf[:,[0,1,3,4]]
        targets = Xy_wf[:,5]
        ply_test = []
        ply_train = []
        best_ply = []
        
        for k in ks:
            ply = make_pipeline(PolynomialFeatures(k), Ridge())
            best_model, ply_train_rmse, ply_test_rmse = find_best_model(data_feature, targets, ply)
            best_ply.append(best_model)
            ply_test.append(np.sum(ply_test_rmse)/len(ply_test_rmse))
            ply_train.append(np.sum(ply_train_rmse)/len(ply_train_rmse))
            
        plt.plot(ks, ply_test, label = "test")
        plt.plot(ks, ply_train, label = "train")
        plt.xlabel("degree")
        plt.ylabel("RMSE")
        plt.legend()
        plt.title("The RMSE vs the number of degree")
        plt.show()
        
        best_idx = np.argmin(ply_test)
        best_k = ks[best_idx]
        print(best_k)
        print(ply_test[best_idx])
        print(ply_train[best_idx])

        best_model = best_ply[best_idx]
        best_model_target = best_model.predict(data_feature)
        plot_scatter(targets, best_model_target)

    ## e
    data_feature = generate_feature(data=data, t = 'value')
    targets = np.array(label.data['Size of Backup (GB)'].tolist())

    ks = range(1,101)
    kn_test = []
    kn_train = []
    best_kn = []

    for k in ks:
        clf = KNeighborsRegressor(n_neighbors = k)
        best_model, kn_train_rmse, kn_test_rmse = find_best_model(data_feature, targets, clf)
        best_kn.append(best_model)
        kn_test.append(np.sum(kn_test_rmse)/len(kn_test_rmse))
        kn_train.append(np.sum(kn_train_rmse)/len(kn_train_rmse))
        

    plt.plot(ks, kn_train, label = "train")
    plt.plot(ks, kn_test, label = "test")
    plt.xlabel("the number of neighbors")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title("The RMSE vs the number of neighbors")
    plt.show()

    best_idx = np.argmin(kn_test)
    best_k = ks[best_idx]
    print(best_k)
    print(kn_test[best_idx])
    print(kn_train[best_idx])

    best_model = best_kn[best_idx]
    best_model_target = best_model.predict(data_feature)
    plot_scatter(targets, best_model_target)
    
