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
                    #df[['loudness','tempo']] = (df[['loudness','tempo']] - df[['loudness','tempo']].min())/(df[['loudness','tempo']].max() - df[['loudness','tempo']].min())
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
    #val_data = Music_Data(pathname + '\src\input_data\val')
    test_data = Music_Data(pathname + '\src\input_data\\test')
    #print(train_nums)
    #test_nums = test_data[1:]
    #print(test_data[0])
    title = ['danceability', 'energy', 'loudness', 'acousticness', 'valence', 'tempo', 'mood']
    # remove unnecessary columns and convert array to float
    train_feature_data = train_data.keep_columns([0, 1, 3, 6, 9, 10, 18]) # dance, energy, loudness, acousticness, valence, tempo, mood
    #print(train_feature_data[0])
    #val_feature_data = val_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    
    test_feature_data = test_data.keep_columns([0, 1, 3, 6, 9, 10, 18])
    #print(feature_data[0:5,:])    
    df_train = pd.DataFrame(train_feature_data, columns = title)
    df_test = pd.DataFrame(test_feature_data, columns = title)
    df = df_train.append(df_test, ignore_index = True)
    # print(df_train)
    # print(df_test)
    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']

    #model = DecisionTreeClassifier(max_depth=10)
    # model = RandomForestClassifier(n_estimators=50)
    # train_labels = train_feature_data[0:,6]
    # train_features = train_feature_data[0:,0:5]
    # #val_labels = val_feature_data[0:,6]
    # #val_features = val_feature_data[0:,0:5]
    # # # # Fit the model to our training data
    # model.fit(train_features, train_labels)
    # training_predicted = model.predict(train_features)
    # score = (1-sum(abs(training_predicted-train_labels))/len(training_predicted)) # LMAOOOO WHAT
    # print('training accuracy:', score)

    # model.fit(val_features, val_labels)
    # #val_predicted = model.predict(val_features)
    # score = (1-sum(abs(val_predicted-val_labels))/len(val_predicted)) # LMAOOOO WHAT
    # print('validation accuracy:', score)
    # # # Make predictions
    # test_labels = test_feature_data[0:,6]
    # test_features = test_feature_data[0:,0:5]
    # testing_predicted = m.predict(test_features)
    # score = (1-sum(abs(testing_predicted-test_labels))/len(testing_predicted)) # LMAOOOO WHAT
    # print('testing accuracy:', score)

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ["mood"]),
                                                            df.mood,
                                                            test_size = 0.2,
                                                            random_state = 1)



    def fit_and_score_model(mdl, X_train, X_test, y_train, y_test, random_state = 1):
        mdl.fit(X_train, y_train)
        train_score = mdl.score(X_train, y_train)
        test_score = mdl.score(X_test, y_test)

        print('the accuracy on the: \n\t training data is {}'.format(round(train_score,3)))
        print('\ttesting data is {}'.format(round(test_score,3)))
        return train_score, test_score
    print('CART Model:')
    cart_model = DecisionTreeClassifier(random_state= 1, max_depth = 10)
    train_score, test_score = fit_and_score_model(cart_model, X_train, X_test, y_train, y_test)
    
    depths = [7,8,9,10,11,12,13,14,15]
    scores = []
    print('Random Forest:\n')
    for i in depths:
        mdl = RandomForestClassifier(n_estimators = 100, max_depth = i, random_state = 1, bootstrap = True, max_samples = 0.2)
        print(f'the accuracy on {i} max depth:')
        train_score, test_score = fit_and_score_model(mdl, X_train, X_test, y_train, y_test, random_state = 1)
        scores.append(test_score)
