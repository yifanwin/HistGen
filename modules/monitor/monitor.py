"""
Monitor - 训练监控协调器
输出: info.log, metrics.jsonl, final_report.txt, summary.json, plots/
"""
import os
import os.path as osp

from .logger import TrainingLogger
from .metrics_tracker import MetricsTracker
from .timer import Timer
from .visualizer import Visualizer
from .reporter import PerformanceReporter


class Monitor:
    """
    训练监控协调器: 协调日志、指标追踪、可视化、报告生成

    使用示例:
        monitor = Monitor(record_dir=args.save_dir, monitor_metric='BLEU_4', monitor_mode='max')
        monitor.start_training()
        for epoch in range(epochs):
            monitor.start_epoch(epoch)
            # ... 训练代码 ...
            monitor.end_epoch(epoch, train_metrics, val_metrics)
        monitor.end_training()
    """

    def __init__(self, record_dir, monitor_metric='BLEU_4', monitor_mode='max',
                 smooth_sigma=0, plot_metrics=None, plot_loss=True, resume=False):
        self.record_dir = record_dir
        self._is_resume = resume

        if not os.path.exists(self.record_dir):
            os.makedirs(self.record_dir)

        # 初始化组件
        self.logger = TrainingLogger(self.record_dir, resume=resume)
        self.metrics = MetricsTracker(monitor_metric=monitor_metric, mode=monitor_mode)
        self.timer = Timer()

        self.visualizer = Visualizer(
            self.record_dir,
            smooth_sigma=smooth_sigma,
            exp_name=osp.basename(record_dir),
            plot_metrics=plot_metrics,
            plot_loss=plot_loss
        )

        self.reporter = PerformanceReporter(self.record_dir)

        # 兼容性属性
        self.log_dir = osp.join(self.record_dir, 'info.log')
        self.log_name = self.log_dir

    def start_training(self):
        """训练开始时调用"""
        self.timer.start_total()
        if not self._is_resume:
            self.logger.log_message("Training started")

    def start_epoch(self, epoch):
        """每个epoch开始时调用"""
        self.timer.start_epoch()

    def end_epoch(self, epoch, train_metrics, val_metrics, test_metrics=None):
        """每个epoch结束时调用"""
        epoch_time = self.timer.end_epoch()

        # 记录指标
        self.metrics.update('train', epoch, train_metrics)
        self.metrics.update('val', epoch, val_metrics)

        # 日志输出
        self.logger.log_epoch(epoch, train_metrics, val_metrics, epoch_time)

        # 可视化更新
        self.visualizer.update(epoch, train_metrics=train_metrics,
                               val_metrics=val_metrics, test_metrics=test_metrics)

        return epoch_time

    def end_training(self):
        """训练结束时调用"""
        total_time = self.timer.end_total()

        # 生成最终报告
        summary = self.metrics.get_summary()
        timer_stats = self.timer.get_statistics()
        self.reporter.generate_report(summary, total_time, timer_stats)

        # 保存训练曲线
        self.visualizer.save_plots()

        self.logger.log_message(f"Training completed in {Timer.format_time(total_time)}")
        best = summary['best']
        self.logger.log_message(f"Best {best['metric']}: {best['value']:.6f} (Epoch {best['epoch']})")

        return summary

    def log_test_metrics(self, test_metrics):
        """记录测试指标"""
        self.metrics.update('test', 0, test_metrics)
        self.metrics.history['test'] = test_metrics
        self.logger.log_test(test_metrics)

    def log_message(self, msg, level='info'):
        """记录消息"""
        self.logger.log_message(msg, level)

    def log_config(self, config_dict):
        """记录配置"""
        self.logger.log_config(config_dict)

    def get_best(self):
        """获取最佳指标"""
        return self.metrics.get_best()

    def get_history(self):
        """获取训练历史（用于checkpoint保存）"""
        history = {
            'train': dict(self.metrics.history.get('train', {})),
            'val': dict(self.metrics.history.get('val', {})),
            'test': dict(self.metrics.history.get('test', {})),
            'epochs': {
                'train': list(self.visualizer.epochs.get('train', [])),
                'val': list(self.visualizer.epochs.get('val', [])),
                'test': list(self.visualizer.epochs.get('test', []))
            }
        }
        return history

    def load_history(self, history_data):
        """从checkpoint加载历史数据（用于resume）"""
        if not history_data:
            return

        for phase in ['train', 'val', 'test']:
            if phase in history_data:
                for key, values in history_data[phase].items():
                    if key not in self.metrics.history[phase]:
                        self.metrics.history[phase][key] = []
                    self.metrics.history[phase][key] = list(values)

        if 'epochs' in history_data:
            for phase in ['train', 'val', 'test']:
                if phase in history_data['epochs']:
                    self.visualizer.epochs[phase] = list(history_data['epochs'][phase])

            for phase in ['train', 'val', 'test']:
                if phase in history_data:
                    for key, values in history_data[phase].items():
                        self.visualizer.history[phase][key] = list(values)

            self.visualizer.save_plots()

        self._is_resume = True
