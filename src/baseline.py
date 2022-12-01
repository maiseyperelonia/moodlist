# import necessary libraries
import numpy as np
import os
import glob
import pandas as pd
import csv
import torch
import torchvision
from torchvision import datasets
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Helper Music_Data Class
# Correct values in Acousticness, Valence, Energy, Danceability, Loudness, Tempo
class Music_Data:
    """Music data set"""
    def __init__(self, path):
        """Music data set"""
        #print(path)
        train_path = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']
        data_orig = []
        num = -1
        #data_orig.append(train_path)
        # loop over the list of .csv files
        for word in train_path:
            total = path + '\\' + word
            #print(total)
            num+=1
            csv_files = glob.glob(os.path.join(total, "*.csv"))
            for file in csv_files:
                # read the csv file
                #print(file)
                df = pd.read_csv(file)
                with open(file) as fd:
                    reader = csv.reader(fd, delimiter=',')
                    count = 0
                    for row in reader:
                        row.append(num)
                        if count > 0 and row[0] != '':
                            #print(row)
                            data_orig.append(row)
                        #else:
                            #print(len(row))
                        count += 1
        self.data = np.array(data_orig, dtype=object)
        #print(num)

    def keep_columns(self, L):
        """Select Features """           
        feature_data = self.data[:,L]
        feature_data = feature_data.astype(float)
        return feature_data

if __name__ == "__main__":
    # call function to prepare data structure
    pathname = os.getcwd()
    train_data = Music_Data(pathname + '\src\input_data\\train') # separate??
    #train_nums = train_data.data[1:]
    test_data = Music_Data(pathname + '\src\input_data\\test')
    #print(train_nums)
    #test_nums = test_data[1:]
    #print(test_data[0])
    title = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'mood']
    # remove unnecessary columns and convert array to float
    train_feature_data = train_data.keep_columns([0, 1, 3, 6, 9, 10, 18]) # dance, energy, loudness, acousticness, valence, tempo, mood
    #print(train_feature_data[0])
    
    test_feature_data = test_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    #print(feature_data[0:5,:])    
    df_train = pd.DataFrame(train_feature_data, columns = title)
    df_test = pd.DataFrame(test_feature_data, columns = title)
    df = df_train.append(df_test, ignore_index = True)
    # print(df_train)
    # print(df_test)
    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']


    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ["mood"]),
                                                            df.mood,
                                                            test_size = 0.2,
                                                            random_state = 1)



    def fit_and_score_model(mdl, X_train, X_test, y_train, y_test, random_state = 1):
        mdl.fit(X_train, y_train)
        train_score = mdl.score(X_train, y_train)
        test_score = mdl.score(X_test, y_test)

        print('Accuracy for \n\t training data: {}'.format(round(train_score,3)))
        print('\ttesting data: {}'.format(round(test_score,3)))
        return train_score, test_score
    print('CART Model:')
    cart_model = DecisionTreeClassifier(random_state= 1, max_depth = 10)
    train_score, test_score = fit_and_score_model(cart_model, X_train, X_test, y_train, y_test)
    
    depths = [7,8,9,10,11,12,13,14,15]
    scores = []
    print('Random Forest:\n')
    for i in depths:
        mdl = RandomForestClassifier(n_estimators = 100, max_depth = i, random_state = 1, bootstrap = True, max_samples = 0.2)
        print(f'max_depth {i}:')
        train_score, test_score = fit_and_score_model(mdl, X_train, X_test, y_train, y_test, random_state = 1)
        scores.append(test_score)

    n_e = [100,150,200,250,300,350]
    scores = []
    print('Random Forest:\n')
    for i in n_e:
        mdl = RandomForestClassifier(n_estimators = i, max_depth = 10, random_state = 1, bootstrap = True, max_samples = 0.2)
        print(f'{i} n_estimators:')
        train_score, test_score = fit_and_score_model(mdl, X_train, X_test, y_train, y_test, random_state = 1)
        scores.append(test_score)

    # confusion matrix
    
