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
    #print(df_train)
    #print(df_test)
    moods = ['Angry', 'Calm', 'Happy', 'Love', 'Sad']

    model = DecisionTreeClassifier(max_depth=12)
    #model = RandomForestClassifier(n_estimators=100)
    train_labels = train_feature_data[0:,6]
    train_features = train_feature_data[0:,0:5]
    #val_labels = val_feature_data[0:,6]
    #val_features = val_feature_data[0:,0:5]
    # # # Fit the model to our training data
    model.fit(train_features, train_labels)
    training_predicted = model.predict(train_features)
    score = (1-sum(abs(training_predicted-train_labels))/len(training_predicted)) # LMAOOOO WHAT
    print('training accuracy:', score)

    # model.fit(val_features, val_labels)
    # #val_predicted = model.predict(val_features)
    # score = (1-sum(abs(val_predicted-val_labels))/len(val_predicted)) # LMAOOOO WHAT
    # print('validation accuracy:', score)
    # # # Make predictions
    test_labels = test_feature_data[0:,6]
    test_features = test_feature_data[0:,0:5]
    testing_predicted = model.predict(test_features)
    score = (1-sum(abs(testing_predicted-test_labels))/len(testing_predicted)) # LMAOOOO WHAT
    print('testing accuracy:', score)