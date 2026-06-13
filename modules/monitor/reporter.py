"""
Performance Reporter - 性能报告生成
输出: final_report.txt (人类可读), summary.json (机器可读)
"""
import json
import os.path as osp
from datetime import datetime


class PerformanceReporter:
    """性能报告生成器"""

    def __init__(self, record_dir):
        self.record_dir = record_dir

    def generate_report(self, summary, total_time, timer_stats=None):
        """生成最终报告"""
        self._generate_text_report(summary, total_time, timer_stats)
        self._generate_json_summary(summary, total_time, timer_stats)

    def _generate_text_report(self, summary, total_time, timer_stats=None):
        report_path = osp.join(self.record_dir, 'final_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("TRAINING COMPLETED\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Total Training Time: {self._format_time(total_time)}\n")
            f.write(f"Total Epochs: {summary['total_epochs']}\n")

            if timer_stats:
                f.write(f"Average Epoch Time: {timer_stats['avg_epoch_time']:.2f}s\n")
                f.write(f"Min Epoch Time: {timer_stats['min_epoch_time']:.2f}s\n")
                f.write(f"Max Epoch Time: {timer_stats['max_epoch_time']:.2f}s\n")

            f.write("\n")
            f.write("-" * 40 + "\n")
            f.write("BEST RESULTS\n")
            f.write("-" * 40 + "\n")
            best = summary['best']
            f.write(f"Best {best['metric']}: {best['value']:.6f} (Epoch {best['epoch']})\n\n")

            f.write("-" * 40 + "\n")
            f.write("FINAL METRICS\n")
            f.write("-" * 40 + "\n")

            f.write("Train:\n")
            for k, v in summary['final_train'].items():
                if v is not None:
                    f.write(f"  {k}: {v:.6f}\n")

            f.write("\nValidation:\n")
            for k, v in summary['final_val'].items():
                if v is not None:
                    f.write(f"  {k}: {v:.6f}\n")

            if summary['test']:
                f.write("\nTest:\n")
                for k, v in summary['test'].items():
                    f.write(f"  {k}: {v:.6f}\n")

            f.write("\n")
            f.write("=" * 60 + "\n")
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        print(f"Final report saved to: {report_path}")

    def _generate_json_summary(self, summary, total_time, timer_stats=None):
        summary_path = osp.join(self.record_dir, 'summary.json')

        json_summary = {
            'total_time_seconds': total_time,
            'total_epochs': summary['total_epochs'],
            'best_metric': summary['best'],
            'final_train_metrics': summary['final_train'],
            'final_val_metrics': summary['final_val'],
            'test_metrics': summary['test'],
            'timer_stats': timer_stats,
            'generated_at': datetime.now().isoformat()
        }

        with open(summary_path, 'w') as f:
            json.dump(json_summary, f, indent=2)

    def _format_time(self, seconds):
        if seconds is None:
            return "N/A"
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
