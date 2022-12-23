import os
import sys
import time
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection._univariate_selection import SelectKBest
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif, VarianceThreshold, mutual_info_classif, \
    RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, CategoricalNB, ComplementNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from dask.distributed import Client

from ddos_utils import *


def readfile(filename):
    print("Reading file "+ filename + " from filesystem.")
    df = pd.read_csv(filename, low_memory=False)
    print("Loaded file shape: "+ str(df.shape))
    return df
def savefile(df, filename):
    print("Saving file "+ filename + " from filesystem.")
    df.to_csv(filename, index=False)
    print("file saved.")

#encoding a column with string classes into integers for ML
def encode_df(dataframe, columnNametoEncode, newColumnName):
    encoder = LabelEncoder()
    dataframe[newColumnName] = encoder.fit_transform(dataframe[columnNametoEncode])
    return dataframe

def get_feature_names(dataset):
    return list(dataset.columns.values)[:-1]

# getting all the features that are held constant because they are useless.
def get_constant_features(dataset):
    var_threshold = VarianceThreshold(threshold=0)  # threshold = 0 for constant

    var_threshold.fit(dataset)

    # getting a bool list True-> keep feature, false->remove feature
    features_to_remove_results = var_threshold.get_support()

    # list to keep the name of features to remove
    features_to_remove = []
    feature_names = get_feature_names(dataset)

    for bool, feature in zip(features_to_remove_results, feature_names):
        if bool == False:
            features_to_remove.append(feature)
    return features_to_remove

#Finding and removing the highly correlated features
def get_highly_correlated_features(dataset, algorithm, correlationThreshold):
    features_to_remove = []
    print("Running algorithm: "+str(algorithm))
    correlation_matrix = dataset.corr(method = algorithm)
    #getting only the upper triangle(without diagonal as it is the correlation of target with it self) of the table as it is diagonally symmetrical and identical
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool))
    features_to_remove = features_to_remove+([column for column in upper_tri.columns if any(abs(upper_tri[column]) > correlationThreshold)])
    return [*set(features_to_remove)]

def get_KBest_features_matrix_correlation(dataset, algorithm, kbest):
    correlation_matrix = dataset.corr(method=algorithm)
    features = range(len(get_feature_names(dataset)))
    target_correlation = list(zip(list(correlation_matrix["Label"].abs()), features))
    target_correlation = sorted(target_correlation, key=lambda x: x[0], reverse=True)
    values, kBest_features = (zip(*target_correlation))
    return kBest_features[:kbest]

def get_KBestFeatures_Univariate_selection(X, y, algorithm, k):

    model = SelectKBest(score_func=algorithm, k=k).fit(X, y)

    res = model.get_support()
    kBest = []

    for i in range(len(res)):
        if(res[i]):
            kBest.append(i)
    return kBest

def get_KBestFeatures_RFE(model, X, y, k):
    rfe = RFE(estimator=model, n_features_to_select=k)
    fit = rfe.fit(X, y)
    kBest = []
    for i in range(0, len(fit.support_)):
        if(fit.support_[i]):
            kBest.append(i)
    return kBest

def get_KBestFeatures_RandomForest(X, y, k):
    model = RandomForestClassifier()
    model.fit(X, y)
    important_features_dict = {}
    for idx, val in enumerate(model.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)

    return important_features_list[:k]

def get_KBest_ExtraTrees(X, y, k):
    model = ExtraTreesClassifier(n_estimators=10)
    model.fit(X, y)
    important_features_dict = {}
    for idx, val in enumerate(model.feature_importances_):
        important_features_dict[idx] = val

    important_features_list = sorted(important_features_dict,
                                     key=important_features_dict.get,
                                     reverse=True)

    return important_features_list[:k]

def get_KBest_all(model, dataset, X, y, k):
    res = [0 for i in range(len(get_feature_names(dataset)))]

    """
    # ** 2 algorithms on Univariate Selection
    univar_algorithms_names = ['chi2', 'f_classif']
    univar_algorithms = [chi2, f_classif]
    for algorithm, algorithmName in zip(univar_algorithms, univar_algorithms_names):

        print("\nRunning algorithm " + str(algorithmName) + ":")
        results = get_KBestFeatures_Univariate_selection(X, y, algorithm, k)
        print(str(k) + " best features: ")
        print(results)
        res = append_results(res, results)

    #** 2 algorithms on matrix correlation
    mat_correlation_algorithms = ['pearson', 'spearman']
    for algorithm in mat_correlation_algorithms:
        print("\nRunning Matrix Correlation algorithm: " + str(algorithm))
        results = get_KBest_features_matrix_correlation(dataset, algorithm, k)
        print(str(k) + " best features: ")
        print(results)
        res = append_results(res, results)

    # ** Random Forest Classifier
    print("\nRunning Random Forest Classifer algorithm: ")
    results = get_KBestFeatures_RandomForest(X, y, k)
    print(str(k) + " best features: ")
    print(results)

    #** Extra trees Classifier
    print("\nRunning Extra Trees Classifer algorithm: ")
    results = get_KBest_ExtraTrees(X, y, k)
    print(str(k) + " best features: ")
    print(results)
    res = append_results(res, results)
    """

    # ** Logistic Regression
    print("\nRunning RFE with random forest algorithm: ")
    results = get_KBestFeatures_RFE(model, X, y, k)
    print(str(k) + " best features: ")
    print(results)
    res = append_results(res, results)

    return res

def append_results(results_global, results):
    for feature in results:
        results_global[feature]+=1
    return results_global

def PCA(X, components):
    covar_matrix = PCA(n_components=components)
    covar_matrix.fit(X)

    # Calculate variance ratios
    variance = covar_matrix.explained_variance_ratio_
    var = np.cumsum(np.round(covar_matrix.explained_variance_ratio_, decimals=3) * 100)
    plt.ylabel('% Variance Explained')
    plt.xlabel('No. of Features')
    plt.title('PCA Analysis')
    plt.ylim(20, 110)
    plt.xlim(0, 21)
    plt.plot(var)
    plt.show()
    return

def reducing_initial_dataset(dir_to_make, amount_of_non_benign_traffic_per_type, dataset_filenames, testing_set):
    #Code for reducing the initial dataset while keeping 100% of the benign traffic as it is very small. Also not loading TFTP traffic as file is too big to load

    for file in dataset_filenames:
        if(testing_set == 0):
            partial_data = readfile("01-12\\"+file)
        elif(testing_set == 1):
            partial_data = readfile("03-11\\" + file)
        print("Extracting benign traffic.")
        benign_traffic = partial_data[partial_data[' Label'] == "BENIGN"]
        print("benign traffic shape: "+str(benign_traffic.shape))

        print("Extracting non-benign traffic.")
        non_benign_traffic = partial_data[partial_data[' Label'] != "BENIGN"]
        print("Non-benign traffic shape: "+str(non_benign_traffic.shape))

        print("Reducing non-benign traffic to "+str(amount_of_non_benign_traffic_per_type)+" rows.")
        non_benign_traffic_reduced = non_benign_traffic.sample(n = amount_of_non_benign_traffic_per_type, replace=True)
        print("Reduced non-benign traffic shape: " + str(non_benign_traffic_reduced.shape))

        print("Combining reduced non-benign traffic with begnin traffic.")
        traffic_reduced = pd.concat([non_benign_traffic_reduced, benign_traffic], ignore_index=True, axis = 0)
        print("Final reduced dataset shape: "+str(non_benign_traffic_reduced.shape))

        print("Fixing unnecessary space in column names")
        # fix unnecessary space in column names
        traffic_reduced.columns = traffic_reduced.columns.str.replace(' ', '')

        savefile(traffic_reduced, dir_to_make+"\\reduced_"+file)
    return dir_to_make

def normalize_dataset(dataset, cols):
    print("\nNormalizing all the features to the default [0,1] scale.\n")
    #scaler = MinMaxScaler()
    scaler = preprocessing.StandardScaler()

    traffic_normalized = scaler.fit(dataset)
    traffic_normalized = pd.DataFrame(traffic_normalized, columns=cols)
    return traffic_normalized

def merging_partial_datasets(directory_name, dataset_filenames):

    dataset = pd.DataFrame()
    for file in dataset_filenames:
        partial_data = readfile(directory_name+"\\reduced_"+file)
        dataset = pd.concat([dataset, partial_data], ignore_index=True, axis = 0)
        print("Loaded traffic shape: "+str(partial_data.shape)+"\n")

    print(" DDoS class distribution before class encoding:")
    print(dataset.groupby("Label").size())

    # encode the labels because we want everything in numerical value
    print("Encoding labels")
    dataset = encode_df(dataset, 'Label', 'Label')

    print("Final reduced/cleaned dataset shape: "+str(dataset.shape))
    savefile(dataset, directory_name+"\\reduced_dataset.csv")

def clean_reduce_dataset(dataset):
    # shuffling the rows of the dataset
    dataset = dataset.sample(frac=1)

    # dropping null/inf values
    dataset.dropna(inplace=True)

    features_to_remove = get_constant_features(dataset)

    print("\nRemoving constant features: " + str(len(features_to_remove)))
    print(features_to_remove)
    dataset.drop(columns=features_to_remove, axis=1, inplace=True)

    threshold = 0.95
    features_to_drop_pearson = get_highly_correlated_features(dataset, 'pearson', threshold)
    features_to_drop_spearman = get_highly_correlated_features(dataset, 'spearman', threshold)
    features_to_drop = set(features_to_drop_spearman).union(features_to_drop_pearson)

    print("\nRemoving highly correlated features: " + str(len(features_to_drop)))
    print(features_to_drop)
    dataset.drop(columns=features_to_drop, axis=1, inplace=True)
    return dataset

def get_KBestFeatures(model, dataset, k_list, X_train, y_train,dir_to_make=None):
    print("\n Selecting k best features: ")
    features = get_feature_names(dataset)
    for k in k_list:
        kbest = get_KBest_all(model, dataset, X_train, y_train, k)

        best_features = list(zip(list(kbest), features))
        best_features = sorted(best_features, key=lambda x: x[0], reverse=True)

        print("best features: ")
        print(best_features)

        values, k_features = (zip(*best_features))

        k_features = list(k_features[:k])
        k_features.append('Label')

        print(str(k) + " - Best features: ")
        print(k_features)
        k_dataset = dataset[k_features]
        if(dir_to_make != None):
            savefile(k_dataset, dir_to_make + "\\reduced_cleaned_" + str(k) + "Best_dataset.csv")

def run_ML_algorithms(X_train, X_val , y_train, y_val, models, res_table, subset_num, k):

    scoring = 'accuracy'
    # evaluate each model in turn
    results = []
    names = []
    cnt = 0
    for name, model in models:

        #print("Algorithm: " + name)
        start_time = time.time()
        #scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5)
        #print("Training accuracy %0.3f" % (scores.mean()))
        elapsed_time = time.time() - start_time
        names.append(name)

        with joblib.parallel_backend("dask"):
            model.fit(X_train, y_train.values.ravel())
        y_val_preds = model.predict(X_val)
        y_train_preds = model.predict(X_train)

        #print("Validation accuracy: {}".format(precision_score(y_val, y_val_preds, average='micro'),beta=2.0))

        elapsed_time = time.time() - start_time
        names.append(name)

        print("Algorithm: " +name+ ", Training duration: " + str(elapsed_time) + " seconds,  Training accuracy: {}".format(precision_score(y_train, y_train_preds, average='micro'),beta=2.0), ", Validation accuracy: {}".format(precision_score(y_val, y_val_preds, average='micro'),beta=2.0))
        res_table[cnt][0]+=precision_score(y_train, y_train_preds, average='micro')
        res_table[cnt][1]+=precision_score(y_val, y_val_preds, average='micro')
        res_table[cnt][2]+=elapsed_time
        if(precision_score(y_train, y_train_preds, average='micro') > res_table[cnt][3]):
            res_table[cnt][3] = precision_score(y_train, y_train_preds, average='micro')
            res_table[cnt][4] = subset_num
            res_table[cnt][5] = k
        if(precision_score(y_val, y_val_preds, average='micro') > res_table[cnt][6]):
            res_table[cnt][6] = precision_score(y_val, y_val_preds, average='micro')
            res_table[cnt][7] = subset_num
            res_table[cnt][8] = k

        cnt+=1
    return res_table

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
