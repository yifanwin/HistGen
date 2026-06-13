"""
Visualizer - 训练可视化（仅保存图像，无端口输出）
"""
import os
import os.path as osp
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 默认绘制的指标列表
DEFAULT_PLOT_METRICS = ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'FACT']


class Visualizer:
    """
    训练可视化: 绘制loss曲线、metric曲线，保存为图片文件
    """

    def __init__(self, record_dir, smooth_sigma=0, exp_name='Experiment',
                 plot_metrics=None, plot_loss=True):
        """
        Args:
            record_dir: 记录目录
            smooth_sigma: 高斯平滑参数，0表示不平滑
            exp_name: 实验名称
            plot_metrics: 要绘制的指标列表，None表示绘制所有非loss指标
            plot_loss: 是否绘制loss曲线
        """
        self.record_dir = record_dir
        self.plot_dir = osp.join(record_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

        self.exp_name = exp_name
        self.smooth_sigma = smooth_sigma
        self.plot_metrics = plot_metrics if plot_metrics is not None else DEFAULT_PLOT_METRICS
        self.plot_loss = plot_loss

        # 历史数据
        self.history = {
            'train': defaultdict(list),
            'val': defaultdict(list),
            'test': defaultdict(list)
        }
        self.epochs = {'train': [], 'val': [], 'test': []}

    def update(self, epoch, train_metrics=None, val_metrics=None, test_metrics=None):
        """更新可视化数据并保存图像"""
        if train_metrics:
            if epoch not in self.epochs['train']:
                self.epochs['train'].append(epoch)
                for k, v in train_metrics.items():
                    self.history['train'][k].append(v)
            else:
                idx = self.epochs['train'].index(epoch)
                for k, v in train_metrics.items():
                    if idx < len(self.history['train'][k]):
                        self.history['train'][k][idx] = v
                    else:
                        self.history['train'][k].append(v)

        if val_metrics:
            if epoch not in self.epochs['val']:
                self.epochs['val'].append(epoch)
                for k, v in val_metrics.items():
                    self.history['val'][k].append(v)
            else:
                idx = self.epochs['val'].index(epoch)
                for k, v in val_metrics.items():
                    if idx < len(self.history['val'][k]):
                        self.history['val'][k][idx] = v
                    else:
                        self.history['val'][k].append(v)

        if test_metrics:
            if epoch not in self.epochs['test']:
                self.epochs['test'].append(epoch)
                for k, v in test_metrics.items():
                    self.history['test'][k].append(v)
            else:
                idx = self.epochs['test'].index(epoch)
                for k, v in test_metrics.items():
                    if idx < len(self.history['test'][k]):
                        self.history['test'][k][idx] = v
                    else:
                        self.history['test'][k].append(v)

        self.save_plots()

    def _smooth(self, data):
        """应用高斯平滑"""
        if self.smooth_sigma <= 0 or len(data) < 3:
            return data
        return gaussian_filter1d(data, sigma=self.smooth_sigma)

    def save_plots(self):
        """保存所有曲线图"""
        if not HAS_MATPLOTLIB:
            return

        if self.plot_loss:
            self._plot_loss()

        self._plot_metrics()
        self._plot_combined()

    def _plot_loss(self):
        """绘制loss曲线"""
        train_loss_keys = [k for k in self.history['train'].keys() if k.endswith('_loss') or k == 'loss']

        if not train_loss_keys:
            return

        loss_priority = ['total_loss', 'ce_loss', 'aux_loss', 'loss']
        sorted_loss_keys = []
        for lk in loss_priority:
            if lk in train_loss_keys:
                sorted_loss_keys.append(lk)
        for lk in train_loss_keys:
            if lk not in sorted_loss_keys:
                sorted_loss_keys.append(lk)

        n_losses = len(sorted_loss_keys)
        if n_losses <= 0:
            return

        if n_losses <= 2:
            n_cols = n_losses
            n_rows = 1
        else:
            n_cols = 2
            n_rows = (n_losses + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
        if n_losses == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, loss_name in enumerate(sorted_loss_keys):
            ax = axes[idx]
            values = self.history['train'][loss_name]
            if not values:
                continue

            epochs = self.epochs['train'][:len(values)]
            ax.plot(epochs, values, marker='o', linewidth=2, label=loss_name, alpha=0.7, color='blue')

            if self.smooth_sigma > 0 and len(values) >= 3:
                ax.plot(epochs, self._smooth(values), linewidth=2,
                        label=f'{loss_name} (smoothed)', linestyle='--', color='cyan')

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(f'{loss_name}', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(sorted_loss_keys), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(osp.join(self.plot_dir, 'loss.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_metrics(self):
        """绘制指标曲线（同时显示val和test）"""
        all_keys = set(self.history['val'].keys()) | set(self.history['test'].keys())
        metric_keys = [k for k in self.plot_metrics if k in all_keys]

        if not metric_keys:
            return

        n_metrics = len(metric_keys)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, metric_name in enumerate(metric_keys):
            ax = axes[idx]

            # 绘制val曲线（蓝色）
            if metric_name in self.history['val'] and self.history['val'][metric_name]:
                val_values = self.history['val'][metric_name]
                val_epochs = self.epochs['val'][:len(val_values)]
                ax.plot(val_epochs, val_values, marker='o', linewidth=2,
                        alpha=0.7, label='Val', color='blue')

                if self.smooth_sigma > 0 and len(val_values) >= 3:
                    ax.plot(val_epochs, self._smooth(val_values), linewidth=2,
                            label=f'Val (smoothed)', linestyle='--', color='cyan')

                # 标记最佳点
                best_idx = np.argmax(val_values)
                ax.scatter([val_epochs[best_idx]], [val_values[best_idx]],
                           color='red', s=100, zorder=5, marker='*')
                ax.annotate(f'Best Val: {val_values[best_idx]:.4f}',
                            xy=(val_epochs[best_idx], val_values[best_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=8, ha='left', color='blue',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # 绘制test曲线（橙色）
            if metric_name in self.history['test'] and self.history['test'][metric_name]:
                test_values = self.history['test'][metric_name]
                test_epochs = self.epochs['test'][:len(test_values)]
                ax.plot(test_epochs, test_values, marker='s', linewidth=2,
                        alpha=0.7, label='Test', color='orange')

                if self.smooth_sigma > 0 and len(test_values) >= 3:
                    ax.plot(test_epochs, self._smooth(test_values), linewidth=2,
                            label=f'Test (smoothed)', linestyle='--', color='gold')

                best_idx = np.argmax(test_values)
                ax.scatter([test_epochs[best_idx]], [test_values[best_idx]],
                           color='darkgreen', s=100, zorder=5, marker='D')
                ax.annotate(f'Best Test: {test_values[best_idx]:.4f}',
                            xy=(test_epochs[best_idx], test_values[best_idx]),
                            xytext=(10, -15), textcoords='offset points',
                            fontsize=8, ha='left', color='darkgreen',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(f'{metric_name}', fontsize=12)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # 隐藏多余的子图
        for idx in range(len(metric_keys), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(osp.join(self.plot_dir, 'metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_combined(self):
        """绘制组合图（Val + Test 主要指标对比）"""
        if not self.epochs['val'] and not self.epochs['test']:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        main_metric = None
        all_keys = set(self.history['val'].keys()) | set(self.history['test'].keys())
        for metric in ['BLEU_4', 'CIDEr', 'ROUGE_L', 'METEOR', 'FACT']:
            if metric in all_keys:
                main_metric = metric
                break

        if main_metric:
            # 绘制val曲线（蓝色）
            if main_metric in self.history['val'] and self.history['val'][main_metric]:
                val_values = self.history['val'][main_metric]
                val_epochs = self.epochs['val'][:len(val_values)]
                ax.plot(val_epochs, val_values, marker='o', linewidth=2,
                        color='blue', alpha=0.7, label=f'Val {main_metric}')

                if self.smooth_sigma > 0 and len(val_values) >= 3:
                    ax.plot(val_epochs, self._smooth(val_values), linewidth=2,
                            label=f'Val (smoothed)', linestyle='--', color='cyan')

                best_idx = np.argmax(val_values)
                ax.scatter([val_epochs[best_idx]], [val_values[best_idx]],
                           color='red', s=100, zorder=5, marker='*')
                ax.annotate(f'Best Val: {val_values[best_idx]:.4f}\nEpoch {val_epochs[best_idx]}',
                            xy=(val_epochs[best_idx], val_values[best_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, ha='left', color='blue',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # 绘制test曲线（橙色）
            if main_metric in self.history['test'] and self.history['test'][main_metric]:
                test_values = self.history['test'][main_metric]
                test_epochs = self.epochs['test'][:len(test_values)]
                ax.plot(test_epochs, test_values, marker='s', linewidth=2,
                        color='orange', alpha=0.7, label=f'Test {main_metric}')

                if self.smooth_sigma > 0 and len(test_values) >= 3:
                    ax.plot(test_epochs, self._smooth(test_values), linewidth=2,
                            label=f'Test (smoothed)', linestyle='--', color='gold')

                best_idx = np.argmax(test_values)
                ax.scatter([test_epochs[best_idx]], [test_values[best_idx]],
                           color='darkgreen', s=100, zorder=5, marker='D')
                ax.annotate(f'Best Test: {test_values[best_idx]:.4f}\nEpoch {test_epochs[best_idx]}',
                            xy=(test_epochs[best_idx], test_values[best_idx]),
                            xytext=(10, -25), textcoords='offset points',
                            fontsize=10, ha='left', color='darkgreen',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(main_metric, fontsize=12)
            ax.set_title(f'{self.exp_name} - {main_metric} Progress (Val vs Test)', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No metric data yet', ha='center', va='center', fontsize=14)
            ax.set_title(f'{self.exp_name} - Training Progress', fontsize=14)

        plt.tight_layout()
        plt.savefig(osp.join(self.plot_dir, 'combined.png'), dpi=150, bbox_inches='tight')
        plt.close()
