import spotipy as spot
from spotipy.oauth2 import SpotifyClientCredentials

import sys
import json
import time
import os
import csv


def spotify_auth():
    scope = "user-library-read"

    auth_manager = SpotifyClientCredentials(client_id='d257b254f2fb4b31b1b67ef23f7333b3', client_secret='5e7c2d68c4964c2b925a789cdd6b4b36')
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

def process_playlists(path):
    filenames = os.listdir(path)
    for filename in sorted(filenames):
        if filename.startswith("mpd.slice.") and filename.endswith(".json"):
            fullpath = os.sep.join((path, filename))
            f = open(fullpath)
            js = f.read()
            f.close()
            mpd_slice = json.loads(js)
            for playlist in mpd_slice["playlists"]:
                print_playlist(playlist)


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
    # print_playlists(auth)
    write_csv()
    #process_playlists(path)
