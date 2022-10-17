import spotipy as spot
from spotipy.oauth2 import SpotifyClientCredentials

scope = "user-library-read"

auth_manager = SpotifyClientCredentials(client_id='d257b254f2fb4b31b1b67ef23f7333b3', client_secret='5e7c2d68c4964c2b925a789cdd6b4b36')
sp = spot.Spotify(auth_manager=auth_manager)


playlists = sp.user_playlists('spotify')
while playlists:
    for i, playlist in enumerate(playlists['items']):
        print("%4d %s %s" % (i + 1 + playlists['offset'], playlist['uri'],  playlist['name']))
    if playlists['next']:
        playlists = sp.next(playlists)
    else:
        playlists = None