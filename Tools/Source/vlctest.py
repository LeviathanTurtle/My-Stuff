import os
import sys
import ctypes

# Set VLC path manually (ensure it's the 64-bit version)
vlc_path = r"C:\Program Files\VideoLAN\VLC"  # Update this path to where your 64-bit VLC is installed
os.environ["PATH"] += os.pathsep + vlc_path

# Explicitly load the 64-bit libvlc.dll
try:
    ctypes.CDLL(os.path.join(vlc_path, "libvlc.dll"))
except OSError as e:
    print(f"Error loading libvlc.dll: {e}")

# Now try importing vlc
import vlc

# Initialize VLC
instance = vlc.Instance()
media_player = instance.media_player_new()

print("VLC loaded successfully!")
