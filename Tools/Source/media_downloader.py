# 
# 
# 

from os import path, makedirs, getenv
from sys import argv, exit
from dotenv import load_dotenv

from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
#import spotify_dl
from yt_dlp import YoutubeDL

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
SPOTIFY_CLIENT_ID = getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = getenv("SPOTIFY_CLIENT_SECRET")
if not (SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET):
    raise ValueError("API client ID not found.")


def download_youtube(youtube_url, output_folder):
    """Download a YouTube video as an MP3 file."""
    
    options = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': path.join(output_folder, '%(title)s.%(ext)s'),
    }

    with YoutubeDL(options) as ydl:
        ydl.download([youtube_url])


def download_spotify_tracks(spotify_url, output_folder):
    """Download tracks from a Spotify playlist, album, or track URL."""
    
    # Authenticate with Spotify
    sp = Spotify(client_credentials_manager=SpotifyClientCredentials(
                 client_id=SPOTIFY_CLIENT_ID,
                 client_secret=SPOTIFY_CLIENT_SECRET
    ))

    results = []
    
    if "track" in spotify_url:
        track_info = sp.track(spotify_url)
        results.append(track_info)
    elif "playlist" in spotify_url:
        playlist_info = sp.playlist_tracks(spotify_url)
        results = playlist_info['items']
    elif "album" in spotify_url:
        album_info = sp.album_tracks(spotify_url)
        results = album_info['items']
    else:
        print("Invalid Spotify URL.")
        return

    for item in results:
        if 'track' in item:
            track = item['track']
        else:
            track = item
        
        artist_name = track['artists'][0]['name']
        track_name = track['name']
        search_query = f"{artist_name} {track_name}"
        print(f"Downloading: {artist_name} - {track_name}")
        try:
            # Search YouTube and download MP3
            with YoutubeDL({'quiet': True}) as ydl:
                search_results = ydl.extract_info(f"ytsearch:{search_query}", download=False)['entries']
                if search_results:
                    youtube_url = search_results[0]['webpage_url']
                    download_youtube(youtube_url, output_folder)
                else:
                    print(f"No YouTube results for: {search_query}")
        except Exception as e:
            print(f"Error downloading {search_query}: {e}")
    
    # Set up SpotifyDL configuration
    #config = {
    #    'url': spotify_url,
    #    'output': output_folder,
    #    'format': 'mp3',
    #    'client_id': SPOTIFY_CLIENT_ID,
    #    'client_secret': SPOTIFY_CLIENT_SECRET,
    #    'no_overwrites': True,
    #}

    # Initialize and run SpotifyDL
    #spotify_dl = Spotify_DL(config)
    #spotify_dl.download_songs()

def main():
    if len(argv) < 3:
        print(f"Usage: python media_downloader.py <media url> <output folder>\nYour args ({len(argv)}): {argv}")
        exit(1)

    url = argv[1]
    output_folder = argv[2]

    if not path.exists(output_folder):
        makedirs(output_folder)

    # NOTE: spotify downloads from youtube
    if "spotify.com" in url:
        download_spotify_tracks(url, output_folder)
    elif "youtube.com" in url or "youtu.be" in url:
        download_youtube(url, output_folder)
    else:
        print("Invalid URL. Provide a Spotify or YouTube link.")

if __name__ == "__main__":
    main()