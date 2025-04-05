
# 
# 
# 
# requirements: PyQt5, python-vlc

import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QSlider, QFileDialog
from PyQt5.QtCore import Qt

vlc_path = r"C:\Program Files\VideoLAN\VLC"
os.environ["PYTHON_VLC_LIB_PATH"] = vlc_path
sys.path.append(vlc_path)
import vlc 

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        # VLC instance
        self.instance = vlc.Instance()
        self.mediaPlayer = self.instance.media_player_new()

        # Open folder button
        self.openFolderButton = QPushButton("Open Folder")
        self.openFolderButton.clicked.connect(self.openFolder)

        # Previous and Next buttons
        self.prevButton = QPushButton("⏮️ Previous")
        self.prevButton.clicked.connect(self.playPrevious)
        self.nextButton = QPushButton("Next ⏭️")
        self.nextButton.clicked.connect(self.playNext)

        # Play button
        self.playButton = QPushButton("Play")
        self.playButton.clicked.connect(self.togglePlayPause)

        # Stop button
        self.stopButton = QPushButton("Stop")
        self.stopButton.clicked.connect(self.stopVideo)

        # Progress bar
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderMoved.connect(self.setPosition)

        # Layouts
        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.prevButton)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(self.nextButton)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(controlLayout)
        mainLayout.addWidget(self.slider)
        mainLayout.addWidget(self.openFolderButton)

        self.setLayout(mainLayout)

        # Video file tracking
        self.videoFiles = []
        self.currentIndex = -1

    def openFolder(self):
        folderPath = QFileDialog.getExistingDirectory(self, "Open Folder", "")
        if folderPath:
            self.videoFiles = [os.path.join(folderPath, f) for f in sorted(os.listdir(folderPath))
                               if f.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".wmv"))]
            self.currentIndex = 0
            if self.videoFiles:
                self.playVideo(self.videoFiles[self.currentIndex])

    def playVideo(self, filePath):
        media = self.instance.media_new(filePath)
        self.mediaPlayer.set_media(media)
        self.mediaPlayer.play()
        self.playButton.setText("Pause")

    def playPrevious(self):
        if self.videoFiles and self.currentIndex > 0:
            self.currentIndex -= 1
            self.playVideo(self.videoFiles[self.currentIndex])

    def playNext(self):
        if self.videoFiles and self.currentIndex < len(self.videoFiles) - 1:
            self.currentIndex += 1
            self.playVideo(self.videoFiles[self.currentIndex])

    def togglePlayPause(self):
        if self.mediaPlayer.is_playing():
            self.mediaPlayer.pause()
            self.playButton.setText("Play")
        else:
            self.mediaPlayer.play()
            self.playButton.setText("Pause")

    def stopVideo(self):
        self.mediaPlayer.stop()
        self.playButton.setText("Play")

    def setPosition(self, position):
        self.mediaPlayer.set_position(position / 1000.0)

if __name__ == "__main__":
    #app = QApplication(sys.argv)
    #player = VideoPlayer()
    #player.show()
    #sys.exit(app.exec_())
    
    print(vlc.__version__)
