"""
Metrics Tracker - 指标追踪与统计
"""
from collections import defaultdict


class MetricsTracker:
    """
    指标追踪与统计: 追踪train/val/test各阶段的指标，自动记录最佳指标
    """

    def __init__(self, monitor_metric='BLEU_4', mode='max'):
        self.monitor_metric = monitor_metric
        self.mode = mode  # 'max' or 'min'
        self.history = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': {}
        }
        self.best = {
            'metric': monitor_metric,
            'value': -float('inf') if mode == 'max' else float('inf'),
            'epoch': 0
        }

    def update(self, mode, epoch, metrics):
        """更新指标历史"""
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                self.history[mode][k].append(v)

        # 检查是否是最佳
        if mode == 'val' and self.monitor_metric in metrics:
            current = metrics[self.monitor_metric]
            if self.mode == 'max':
                if current > self.best['value']:
                    self.best['value'] = current
                    self.best['epoch'] = epoch
            else:
                if current < self.best['value']:
                    self.best['value'] = current
                    self.best['epoch'] = epoch

    def get_history(self, mode='val'):
        """获取指标历史"""
        return dict(self.history[mode])

    def get_best(self):
        """获取最佳指标"""
        return self.best

    def get_summary(self):
        """获取训练摘要"""
        summary = {
            'best': self.best,
            'final_train': {k: v[-1] if v else None for k, v in self.history['train'].items()},
            'final_val': {k: v[-1] if v else None for k, v in self.history['val'].items()},
            'test': self.history['test'],
            'total_epochs': len(self.history['val'].get(self.monitor_metric, []))
        }
        return summary
