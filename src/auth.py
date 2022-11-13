import spotipy as spot
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

import sys
import json
import time
import os
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly as px
#import splitfolders
import requests

def get_spotify_token():
    AUTH_URL = 'https://accounts.spotify.com/api/token'
    auth_response = requests.post(AUTH_URL, {
        'grant_type': 'client_credentials',
        'client_id': 'a205530de70741b89d875ffb7540b000',
        'client_secret': 'a9396ffb47254bf6863c31a221b14c95',
    })

    # convert the response to JSON
    auth_response_data = auth_response.json()

    # save the access token
    access_token = auth_response_data['access_token']
    print("token:", access_token)
    return access_token

headers = {
    'Authorization': 'Bearer {token}'.format(token=get_spotify_token())
}

def spotify_auth():
    scope = "user-library-read"

    scope = "playlist-modify-public"
    
    auth_manager = SpotifyClientCredentials(client_id='a205530de70741b89d875ffb7540b000', client_secret='a9396ffb47254bf6863c31a221b14c95')
    sp = spot.Spotify(auth_manager=auth_manager)
    return sp
    

def print_playlists(sp):
    playlists = sp.user_playlists('spotify')
    while playlists:
        for i, playlist in enumerate(playlists['items']):
            print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
        if playlists['next']:
            playlists = sp.next(playlists)
        else:
            playlists = None
    # info = sp.user_playlist_create(sp.me().user_id, "I didn't make this", public=True, collaborative=False, description='')
    # print(info)


def get_playlists(path):
    filenames = os.listdir(path)
    file_num = 0
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)

            label_f = open("../data_scrape.json")
            label_js = label_f.read()
            label_f.close()
            label_info = json.loads(label_js)

            for playlist in mpd_slice["playlists"]:
                for label in label_info["Labels"]:
                    keyword_arr = label_info["Labels"][label]
                    for keyword in keyword_arr["keywords"]:
                        pName = playlist["name"]
                        
                        if (pName.find(keyword) != -1):
                            if (label == "Love"):
                                insert_songs(label, playlist, file_num)
                                #print(pName," -- ", label)
                            #print_playlist(playlist)
                    file_num += 1

def insert_songs(label, playlist, file_num):
    song_cnt = 0
    total_song_cnt = 0
    for track in playlist["tracks"]:
        track_id = track["track_uri"][14:]
        track_info = get_song_info(track_id).json()

        if(song_cnt == 0):
            #print("creating csv for file: ", file_num)
            create_csv(label, track_info, file_num)

        if(song_cnt == 9):
            total_song_cnt += 1
            if ((playlist["num_tracks"] - total_song_cnt) > 7):
                song_cnt = 0
                file_num += 1
            """else:
                print("remaining: ",(playlist["num_tracks"] - total_song_cnt),"for file num",file_num)"""
        else:  
            song_cnt += 1
            total_song_cnt += 1

        #print("writing track num: ", total_song_cnt)
        write_csv(label, track_info, file_num)

def create_csv(label, track, file_num):
    
    file_name = "../feature_data/" + label + "/" + str(file_num) + ".csv"
    with open(file_name,'w', newline='') as fd:

        writer = csv.writer(fd)
        writer.writerow(track.keys())

def write_csv(label, track, file_num):
    file_name = "../feature_data/" + label + "/" + str(file_num) + ".csv"
    with open(file_name,'a', newline='') as fd:
        writer = csv.writer(fd)
        writer.writerow(track.values())

def get_song_info(track_id):
    # base URL of all Spotify API endpoints
    BASE_URL = 'https://api.spotify.com/v1/'

    # actual GET request with proper header
    response = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    return response

def print_playlist(playlist):
    print("=====", playlist["pid"], "====")
    print("name:          ", playlist["name"])
    ts = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(playlist["modified_at"])
    )

    print("last_modified: ", ts)
    print("num edits: ", playlist["num_edits"])
    print("num followers: ", playlist["num_followers"])
    print("num artists: ", playlist["num_artists"])
    print("num albums: ", playlist["num_albums"])
    print("num tracks: ", playlist["num_tracks"])
    print()
    for i, track in enumerate(playlist["tracks"]):
        print(
            "   %3d %s - %s - %s"
            % (i + 1, track["track_name"], track["album_name"], track["artist_name"])
        )
    print()

def visualize_data():
    directory = "../feature_data/"

    #for name in directory:

    all_labels = pd.DataFrame()
    for label_folder in os.listdir(directory):
        label_folder = label_folder + "/"
        print("next label", label_folder)
        label_list = pd.DataFrame()
        num = 0
        for filename in os.listdir(directory+label_folder):
            df = pd.read_csv(directory+label_folder+filename)
<<<<<<< HEAD
=======
            #mean = df.mean()
>>>>>>> rnn_model
            num+=1
            df[['loudness','tempo']] = (df[['loudness','tempo']] - df[['loudness','tempo']].min())/(df[['loudness','tempo']].max() - df[['loudness','tempo']].min())
            features = df[['valence', 'loudness', 'energy', 'danceability', 'tempo', 'acousticness']].median()
            feature_list = features.to_frame().T
            label_list['index'] = filename
            label_list = pd.concat([label_list,feature_list])
            
            
<<<<<<< HEAD
            pd.concat([label_list, feature_list])
        
        print(label_folder)
        #label_list.plot.scatter(x='valence',y='energy', c='loudness', colormap="viridis")
=======
            # pd.concat([label_list, feature_list])
            # print(label_list)
        
        #new_df = label_list.data.iris()
        # label_list.plot.scatter(x='valence',y='energy', c='loudness', colormap="viridis")
>>>>>>> rnn_model
        label_list.plot.scatter(x='danceability',y='acousticness', c='valence', colormap="viridis")
        plt.show()
        label_list = label_list.median().to_frame().T
        all_labels = pd.concat([all_labels, label_list])
    
    all_labels['mood'] = ['Angry','Calm','Happy','Sad']
    all_labels.plot(x="mood",y=['valence','danceability','tempo','loudness','energy','acousticness'])
    plt.show()
    print(all_labels)
<<<<<<< HEAD

def normalize_data():
    directory = "../feature_data/"

    #for name in directory:

    all_labels = pd.DataFrame()
    for label_folder in os.listdir(directory):
        label_folder = label_folder + "/"
        for filename in os.listdir(directory+label_folder):
            if label_folder == "Love/":
                df = pd.read_csv(directory+label_folder+filename)
                df[['loudness','tempo']] = (df[['loudness','tempo']] - df[['loudness','tempo']].min())/(df[['loudness','tempo']].max() - df[['loudness','tempo']].min())
        
                df.loc['loudness'] = df['loudness']
                df.loc['tempo'] = df['tempo']
                df.to_csv(directory+label_folder+filename, index = False)



def split_data():
    directory = "../feature_data/"
    #splitfolders.ratio(directory, output='input_data', seed=1337, ratio=(0.6, 0.2,0.2)) 
=======
            
>>>>>>> rnn_model
            
if __name__ == "__main__":
    path = "../SpotifyDataset/data"
    auth = spotify_auth()
    #get_playlists(path)
    visualize_data()
    #print_playlists(auth)
    #write_csv()
    #split_data()
    #normalize_data()
    #process_playlists(path)
