
# 
# William Wadsworth
# 2.24.2025
# 
# This program searches and deletes most duplicate images and videos. 
# ASSUMPTIONS:
# - each video has a unique first frame
# 
# Usage: python -<video | photo> <path/to/file>
# 

from os import listdir, path, stat, remove
from argparse import ArgumentParser
from json import loads
from subprocess import run, PIPE
from imagehash import phash
from PIL import Image
from collections import defaultdict
from typing import Literal, List, Tuple, Optional

VIDEO_EXTS: str = ('.mp4', '.avi', '.mov', '.mkv')
PHOTO_EXTS: str = ('.jpg', '.jpeg', '.png', '.gif')

# pre-condition: the media path must include the file and be a valid location on the disk, the 
#                media type should either be PHOTO or VIDEO
# post-condition: returns the metadata for a video (duration, resolution) or an image (resolution)
#                 if the file is found, otherwise None
def get_metadata(media_path: str, type: Literal["PHOTO","VIDEO"]) -> Optional[Tuple[float, Tuple[int, int]] | Tuple[int, int]]:
    """Extracts video duration and resolution using ffprobe or resolution for photos."""
    
    try:
        if type == "VIDEO":
            # get video metadata (duration and resolution)
            result = run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                'stream=duration,width,height', '-of', 'json', media_path],
                stdout=PIPE, stderr=PIPE, text=True
            )
            
            # de-jsonify the metadata
            metadata = loads(result.stdout).get('streams', [{}])[0]
            # get the duration and resolution
            duration = float(metadata.get('duration', 0))
            resolution = (int(metadata.get('width', 0)), int(metadata.get('height', 0)))
            
            return duration, resolution
        
        # 
        return Image.open(media_path).size # (width, height)
    except Exception as e:
        print(f"Error extracting metadata for {media_path}: {e}")
        return (None, None) if type == "VIDEO" else None

# pre-condition: the media path must include the file and be a valid location on the disk
# post-condition: returns the image's hash if it is found. If it is not found or the hash fails, it
#                 returns None
def get_image_hash(image_path: str) -> str | None:
    """Generates a perceptual hash for an image."""
    
    try:
        img = Image.open(image_path)
        
        # using phash because it handles duplicates better
        return str(phash(img))
    except Exception as e:
        print(f"Error hashing {image_path}: {e}")
        return None

# pre-condition: the media path must include the file and be a valid location on the disk, the 
#                media type should either be PHOTO or VIDEO
# post-condition: returns a dictionary containing all duplicate media
def find_duplicates(media_path: str, type: Literal["PHOTO","VIDEO"]) -> List[List[str]]:
    """Find duplicates based on file size, duration, and resolution."""
    
    # dict to store duplicate media
    media_info = defaultdict(list)
    extensions = VIDEO_EXTS if type == "VIDEO" else PHOTO_EXTS

    # for each file in the directory
    for filename in listdir(media_path):
        # if the file is one of the supported video formats
        if filename.lower().endswith(extensions):
            # add the file to the file path
            file_path = path.join(media_path, filename)
            # make note of the size for the key
            file_size = stat(file_path).st_size

            metadata = get_metadata(file_path, type)
            
            # store the data we need
            key = (file_size, *metadata)
            if type == "PHOTO":
                key += (get_image_hash(file_path),)  # Only add hash for photos

            # group files with the same key together
            media_info[key].append(file_path)

    # return only groups where multiple files share the same stuff
    return [files for files in media_info.values() if len(files) > 1]
  
# pre-condition: duplicates must be a non-zero entry dictionary containing valid file paths
# post-condition: the non-first media entries of a group are deleted
def delete_duplicates(duplicates: List[List[str]]) -> None:
    """Deletes duplicate files, keeping the first copy."""
    
    for group in duplicates:
        # keep the first file, delete the rest
        for file in group[1:]:
            remove(file)
            print(f"Deleted: {file}")



def main():
    parser = ArgumentParser(description="Find and delete duplicate photos or videos.")
    parser.add_argument("-photo", help="Path to folder containing images", type=str)
    parser.add_argument("-video", help="Path to folder containing videos", type=str)
    args = parser.parse_args()
    
    if args.photo:
        print(f"Scanning for duplicate photos in: {args.photo}")
        duplicates = find_duplicates(args.photo,"PHOTO")
    elif args.video:
        print(f"Scanning for duplicate videos in: {args.video}")
        duplicates = find_duplicates(args.video,"VIDEO")
    else:
        print("Please specify -photo or -video followed by the folder path.")
        
    delete_duplicates(duplicates)
    

if __name__ == "__main__":
    main()
