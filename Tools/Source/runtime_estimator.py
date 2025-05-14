
# 
# William Wadsworth
# 
# Usage:
#     estimator = RuntimeEstimator(n)
#     estimator.start()
#     for i in range(n):
#         estimator.update(i+1)
# 

import time

class RuntimeEstimator:
    def __init__(self, total_iterations, update_every=10):
        self.total = total_iterations
        self.update_every = update_every
        self.start_time = None
        self.last_update = 0
        self.iteration_times = []

    def start(self):
        self.start_time = time.perf_counter()

    def update(self, current_iteration):
        now = time.perf_counter()
        if current_iteration == 0:
            self.start_time = now
            return

        # Only update every N iterations
        if current_iteration % self.update_every == 0 or current_iteration == self.total:
            elapsed = now - self.start_time
            avg_time = elapsed / current_iteration
            remaining = self.total - current_iteration
            eta = avg_time * remaining

            print(f"[{current_iteration}/{self.total}] "
                  f"Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s")

