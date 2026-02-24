import time


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time
