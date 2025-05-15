
# 
# William Wadsworth
# Usage:
#     progress = ProgressBar(n)
#     for i in range(n):
#         progress.update(i+1)
# 

import time
import sys

class ProgressBar:
    def __init__(self, total, length=40):
        self.total = total
        self.length = length
        self.start_time = time.perf_counter()

    def update(self, current):
        now = time.perf_counter()
        # calculate % progress
        percent = current / self.total
        
        # update progress bar
        filled_length = int(self.length * percent)
        bar = '█' * filled_length + '-' * (self.length - filled_length)
        
        # calculate eta
        elapsed = now - self.start_time
        avg_time = elapsed / current if current else 0
        remaining = self.total - current
        eta = avg_time * remaining

        # output msg
        sys.stdout.write(
            f"\r{bar} {percent:.1%} "
            f"Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s"
        )
        # force output to appear immediately
        sys.stdout.flush()

        if current == self.total:
            print() # move to next line on completion

    