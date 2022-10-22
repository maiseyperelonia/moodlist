import spotipy as spot
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth

import sys
import json
import time
import os
import csv
import json
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


def get_playlists(path, keywords):
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice["playlists"]:
                for keyword in keywords:
                    pName = playlist["name"]
                    if (pName.find(keyword) != -1):
                        print(pName)

def get_song_info(track_id):

    # base URL of all Spotify API endpoints
    BASE_URL = 'https://api.spotify.com/v1/'

    # actual GET request with proper header
    r = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    print(r.json())

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

def write_csv():
    header = ['names', 'coolness']
    row = [
        ["harshit", "very cool"], 
        ["anne", "eh"]
        ]
    with open("../feature_data/practice_file.csv",'w', newline='') as fd:

        writer = csv.writer(fd)

        writer.writerow(header)
        writer.writerows(row)

if __name__ == "__main__":
    path = sys.argv[1]
    auth = spotify_auth()
    keyword = sys.argv[2].split(',')
    track_id = sys.argv[3]
    get_song_info(track_id)
    #get_playlists(path, keyword)
    #print_playlists(auth)
    write_csv()

    #process_playlists(path)
