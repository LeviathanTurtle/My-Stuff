
# 
# 
# 

import os
import cv2
import shutil
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
# requirements, opencv-python pillow

class MediaViewer:
    def __init__(self, root, media_folder):
        self.root = root
        self.media_folder = media_folder
        self.media_files = self.get_media_files()
        self.current_index = 0
        self.deleted_files = set()
        self.video_cap = None
        self.playing = False

        self.root.title("Media Viewer")

        # Create UI elements
        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.progress = ttk.Scale(root, from_=0, to=100, orient="horizontal", command=self.seek_video)
        self.progress.pack(fill="x")

        self.controls = tk.Frame(root)
        self.controls.pack()

        self.prev_button = tk.Button(self.controls, text="⏮ Prev", command=self.prev_media)
        self.prev_button.pack(side="left")

        self.play_button = tk.Button(self.controls, text="▶ Play", command=self.toggle_playback)
        self.play_button.pack(side="left")

        self.next_button = tk.Button(self.controls, text="Next ⏭", command=self.next_media)
        self.next_button.pack(side="left")

        self.keep_button = tk.Button(self.controls, text="Keep", command=self.keep_file)
        self.keep_button.pack(side="left")

        self.delete_button = tk.Button(self.controls, text="Delete", command=self.delete_file)
        self.delete_button.pack(side="left")

        self.done_button = tk.Button(self.controls, text="Done", command=self.finish)
        self.done_button.pack(side="left")

        self.load_media()

    def get_media_files(self):
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.mp4', '.avi', '.mov', '.m4v')
        return [f for f in os.listdir(self.media_folder) if f.lower().endswith(valid_extensions)]

    def load_media(self):
        if not self.media_files:
            self.canvas.config(text="No media files found.")
            return

        file_path = os.path.join(self.media_folder, self.media_files[self.current_index])
        if file_path.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.show_image(file_path)
        else:
            self.play_video(file_path)

    def show_image(self, file_path):
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None

        image = Image.open(file_path)
        image.thumbnail((600, 400))
        photo = ImageTk.PhotoImage(image)

        self.canvas.config(image=photo)
        self.canvas.image = photo

    def play_video(self, file_path):
        if self.video_cap:
            self.video_cap.release()

        self.video_cap = cv2.VideoCapture(file_path)
        self.progress["to"] = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.playing = True
        self.update_video_frame()

    def update_video_frame(self):
        if self.video_cap and self.playing:
            ret, frame = self.video_cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                image.thumbnail((600, 400))
                photo = ImageTk.PhotoImage(image)

                self.canvas.config(image=photo)
                self.canvas.image = photo

                self.progress.set(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.root.after(30, self.update_video_frame)  # Safe recursive call
            else:
                # Stop recursion properly when video ends
                self.playing = False
                self.play_button.config(text="▶ Play")
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    def seek_video(self, val):
        if self.video_cap:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(val)))
            self.update_video_frame()

    def toggle_playback(self):
        self.playing = not self.playing
        self.play_button.config(text="⏸ Pause" if self.playing else "▶ Play")
        
        if self.playing:
            self.update_video_frame()  # Resume playback

    def prev_media(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_media()

    def next_media(self):
        if self.current_index < len(self.media_files) - 1:
            self.current_index += 1
            self.load_media()

    def delete_file(self):
        file_path = os.path.join(self.media_folder, self.media_files[self.current_index])
        self.deleted_files.add(file_path)
        self.next_media()

    def keep_file(self):
        self.next_media()

    def finish(self):
        for file in self.deleted_files:
            os.remove(file)
        self.root.quit()

if __name__ == "__main__":
    folder = filedialog.askdirectory(title="Select Media Folder")
    if folder:
        root = tk.Tk()
        app = MediaViewer(root, folder)
        root.mainloop()
