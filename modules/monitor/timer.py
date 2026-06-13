"""
Timer - 时间管理
"""
import time
import numpy as np


class Timer:
    """
    时间管理: 追踪总训练时间、每个epoch时间、预估剩余时间
    """

    def __init__(self):
        self.start_time = None
        self.epoch_start_time = None
        self.epoch_times = []

    def start_total(self):
        self.start_time = time.time()

    def end_total(self):
        return time.time() - self.start_time if self.start_time else 0

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self):
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            return epoch_time
        return 0

    def estimate_remaining(self, current_epoch, total_epochs):
        if not self.epoch_times:
            return None
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = total_epochs - current_epoch
        return avg_epoch_time * remaining_epochs

    def get_statistics(self):
        if not self.epoch_times:
            return {
                'total_time': self.end_total(),
                'avg_epoch_time': 0,
                'min_epoch_time': 0,
                'max_epoch_time': 0,
                'total_epochs': 0
            }
        return {
            'total_time': self.end_total(),
            'avg_epoch_time': np.mean(self.epoch_times),
            'min_epoch_time': np.min(self.epoch_times),
            'max_epoch_time': np.max(self.epoch_times),
            'total_epochs': len(self.epoch_times)
        }

    @staticmethod
    def format_time(seconds):
        if seconds is None:
            return "N/A"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
