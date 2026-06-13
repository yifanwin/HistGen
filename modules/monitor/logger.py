"""
Training Logger - 结构化日志记录
输出: info.log (文本日志), metrics.jsonl (JSON指标记录)
"""
import json
import os.path as osp
from datetime import datetime
from collections import OrderedDict


class TrainingLogger:
    """
    结构化日志记录

    输出:
    - info.log: 人类可读的文本日志
    - metrics.jsonl: JSON格式的指标记录（每行一个JSON对象）
    """

    def __init__(self, record_dir, resume=False):
        self.record_dir = record_dir
        self.log_file = osp.join(record_dir, 'info.log')
        self.metrics_file = osp.join(record_dir, 'metrics.jsonl')
        self.resume = resume

        if not osp.exists(record_dir):
            import os
            os.makedirs(record_dir)

        if not resume:
            self._init_log_file()
        else:
            self._append_resume_marker()

    def _append_resume_marker(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"# Training Resumed: {timestamp}\n")
            f.write(f"{'='*60}\n\n")

    def _init_log_file(self):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_file, 'w') as f:
            f.write(f"{'#'*60}\n")
            f.write(f"# HistGen Training Log\n")
            f.write(f"# Started: {timestamp}\n")
            f.write(f"{'#'*60}\n\n")

    def log_config(self, config_dict):
        """记录配置信息（接收 vars(args) 字典）"""
        with open(self.log_file, 'a') as f:
            f.write("## Configuration\n")
            for key, value in sorted(config_dict.items()):
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    def log_message(self, msg, level='info'):
        """记录消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        line = f"[{timestamp}] [{level.upper()}]\n{msg}\n"
        print(line, end='')
        with open(self.log_file, 'a') as f:
            f.write(line)

    def log_epoch(self, epoch, train_metrics, val_metrics, epoch_time):
        """记录epoch摘要"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        train_str = ' | '.join([f"{k}: {v:.6f}" for k, v in train_metrics.items()])
        val_str = ' | '.join([f"{k}: {v:.6f}" for k, v in val_metrics.items()])

        line = f"[{timestamp}] Epoch {epoch} ({epoch_time:.2f}s)\n"
        line += f"  Train: {train_str}\n"
        line += f"  Val:   {val_str}\n\n"

        print(line, end='')
        with open(self.log_file, 'a') as f:
            f.write(line)

        # JSON格式记录
        record = OrderedDict([
            ('epoch', epoch),
            ('time', epoch_time),
            ('train', train_metrics),
            ('val', val_metrics),
            ('timestamp', datetime.now().isoformat())
        ])
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

    def log_test(self, test_metrics):
        """记录测试结果"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        test_str = ' | '.join([f"{k}: {v:.6f}" for k, v in test_metrics.items()])

        line = f"[{timestamp}] Test Results:\n"
        line += f"  {test_str}\n\n"

        print(line, end='')
        with open(self.log_file, 'a') as f:
            f.write(line)
