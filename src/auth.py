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
    counter = 0
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
            # if(counter > 27000):
            #     print("starting sleep")
            #     time.sleep(30)
            #     print("done sleeping")
            #     counter = 0
            # counter += 1000
            # print(counter)

            for playlist in mpd_slice["playlists"]:
                for label in label_info["Labels"]:
                    keyword_arr = label_info["Labels"][label]
                    for keyword in keyword_arr["keywords"]:
                        pName = playlist["name"]
                        
                        if (pName.find(keyword) != -1):
                            if (label == "Love"):
                                #insert_songs(label, playlist, file_num)
                                if(pName != "spain" and pName != "spain " and pName != "Ambassadors"):
                                    print(pName," -- ", label)
                                counter += 1
                                #print_playlist(playlist)
                    file_num += 1
    print(counter)

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
            #mean = df.mean()
            num+=1
            df[['loudness','tempo']] = (df[['loudness','tempo']] - df[['loudness','tempo']].min())/(df[['loudness','tempo']].max() - df[['loudness','tempo']].min())
            features = df[['valence', 'loudness', 'energy', 'danceability', 'tempo', 'acousticness']].median()
            feature_list = features.to_frame().T
            label_list['index'] = filename
            label_list = pd.concat([label_list,feature_list])
            # pd.concat([label_list, feature_list])
            # print(label_list)
        
        #new_df = label_list.data.iris()
        # label_list.plot.scatter(x='valence',y='energy', c='loudness', colormap="viridis")
        label_list.plot.scatter(x='danceability',y='acousticness', c='valence', colormap="viridis")
        plt.show()
        label_list = label_list.median().to_frame().T
        all_labels = pd.concat([all_labels, label_list])
    
    all_labels['mood'] = ['Angry','Calm','Happy','Sad']
    all_labels.plot(x="mood",y=['valence','danceability','tempo','loudness','energy','acousticness'])
    plt.show()
    print(all_labels)

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

if __name__ == "__main__":
    path = "../SpotifyDataset/data"
    auth = spotify_auth()
    
    #get_playlists(path)
    playlist = [# Anrgy - Harshit
                ["5wQnmLuC1W7ATsArWACrgW","1oTo3ijRbaDAtrjJrGAPSw","3tSmXSxaAnU1EPGKa6NytH",
                "3ZffCQKLFLUvYM59XKLbVm","5hljqZrl0h7RcIhUnOPuve","6ihL9TjfRjadfEePzXXyVF",
                "31q2AsxMNpxHNENkWKG1j0","5iCY0TXNImK4hyKfcplQsg","74kCarkFBzXYXNkkYJIsG0"],
                # Love - Abeer
                ["2eAvDnpXP5W0cVtiI0PUxV","0HYAsQwJIO6FLqpyTeD3l6","39EXZNMxb4RBHlRjnRaOKp",
                 "5gBEdUKVZJgvQwNu8pIQqy","20gAOYKYQ7aFRVRg6LvrLW","6aHCXTCkPiB4zgXKpB7BHS",
                 "1XyzcGhmO7iUamSS94XfqY","1XyzcGhmO7iUamSS94XfqY","3bnVBN67NBEzedqQuWrpP4"],
                # Sad - Anne
                ["5enxwA8aAbwZbf5qCHORXi", "4pNApnaUWAL2J4KO2eqokq", "4nyF5lmSziBAt7ESAUjpbx", 
                "1wXqbn4OVaYBOhgj7Z4did", "2qLyo5FeWquE7HBUbcVnEy", "2I8YAEA1VmCuP1wkJHMpTw", 
                "1BxfuPKGuaTgP7aM0Bbdwr", "3afkJSKX0EAMsJXTZnDXXJ", "3H7oAhHxkEkSf9iomv2mbG"],
                # Love - Anne
                ["1dGr1c8CrMLDpV6mPbImSI","4WHIqhrbdva5vQSrCuet3l","1fzAuUVbzlhZ1lJAx9PtY6",
                "7F5oktn5YOsR9eR5YsFtqb","4A2LfnduSTsE8u0ecYROxE","14LtANuaslKWyYbktUrHBU",
                "2IAR0DziHCjSu16gR4ihvy","22bPsP2jCgbLUvh82U0Z3M","4crdHmkZQmNd2hucSIk7dA"],
                # Angry - Abeer
                ["0H4ugk6rhnXmTl47ayy9O5","3xrn9i8zhNZsTtcoWgQEAd","1wcr8DjnN59Awev8nnKpQ4",
                "31CdkzHnMbvJuKZvtCQfR6","5ghIJDpPoe3CfHMGu71E6T","5cbpoIu3YjoOwbBDGUEp3P",
                "3QTDzwSkRW04FVPo6COm0H","7i9763l5SSfOnqZ35VOcfy","4RVwu0g32PAqgUiJoXsdF8"],
                # Srini
                ["3azJifCSqg9fRij2yKIbWz", "5dewQ7ojISR32NAYNHFYWC","34ZAzO78a5DAVNrYIGWcPm",
                "7kfOEMJBJwdCYqyJeEnNhr","6ZRuF2n1CQxyxxAAWsKJOy","1p3RpKfSJQcVTEwrQWr8q7",
                "2Ch7LmS7r2Gy2kc64wv3Bz","4NFD9ea0uH0MtoC30yNYE1","6ydWxkzjDktHsTzvWcaZ1i"],
                # Happy- Maisey
                ["0QHEIqNKsMoOY5urbzN48u", "1MkFj1ThZZxjYMNkczx9mk","4h9wh7iOZ0GGn8QVp4RAOB",
                "1xzi1Jcr7mEi9K2RfzLOqS","185Q9qHtxrovV2fA09XjAw","0rIAC4PXANcKmitJfoqmVm",
                "5B1TY0oq5I1VeTZxEnkGV8","1magKwGDsyU3RGjpo0BfPe","70khXICDeTTxgYtw3EysKH"],
                # Laurel
                ["5WbfFTuIldjL9x7W6y5l7R","3sNVsP50132BTNlImLx70i","1QxcWlk8PivolUaWcpAoNq",
                 "1bDbXMyjaUIooNwFE9wn0N", "4wFeMmJDlgkAxlQ07PbdGZ","4EDijkJdHBZZ0GwJ12iTAj",
                 "1nsFPTltzIVUJMWynPddpO", "1nsFPTltzIVUJMWynPddpO", "3XCveEutwTaDiekRkkfdp9"]
                ]
    cnt = 0
    track_num = 0
    for playlist_id in range(len(playlist)):
        track_num+=1
        cnt = 0
        print(playlist_id)
        for track_id in playlist[playlist_id]:
            track_info = get_song_info(track_id).json()

            if(cnt == 0):
                create_csv("Angry", track_info, track_num)
                cnt += 1
            #print("writing track num: ", total_song_cnt)
            write_csv("Angry", track_info, track_num)
    #visualize_data()
    #print_playlists(auth)
    #write_csv()
    #split_data()
    #normalize_data()
    #process_playlists(path)
