import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from scipy.stats import entropy

class QualityMetrics:
    def __init__(self):
        self.metrics = {}

    def calculate_ctr_metrics(self, y_true, y_pred):
        """计算CTR预测相关指标"""
        auc = roc_auc_score(y_true, y_pred)
        precision = precision_score(y_true, (y_pred > 0.5).astype(int))
        recall = recall_score(y_true, (y_pred > 0.5).astype(int))

        self.metrics['ctr'] = {
            'auc': auc,
            'precision': precision,
            'recall': recall
        }
        return self.metrics['ctr']

    def calculate_sentiment_match(self, predicted_sentiment, true_sentiment):
        """计算情感匹配度"""
        accuracy = np.mean(predicted_sentiment == true_sentiment)
        self.metrics['sentiment'] = {
            'accuracy': accuracy
        }
        return self.metrics['sentiment']

    def calculate_title_diversity(self, titles):
        """计算标题多样性指标"""
        # 计算词频分布
        word_freq = {}
        for title in titles:
            for word in title.split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # 计算词频熵
        freq_values = np.array(list(word_freq.values()))
        freq_dist = freq_values / freq_values.sum()
        diversity_score = entropy(freq_dist)

        self.metrics['title_diversity'] = {
            'entropy': diversity_score,
            'unique_words': len(word_freq)
        }
        return self.metrics['title_diversity']

    def calculate_clip_zero_shot_accuracy(self, predictions, ground_truth):
        """计算CLIP零样本分类准确率"""
        accuracy = np.mean(predictions == ground_truth)
        self.metrics['clip_accuracy'] = accuracy
        return accuracy

    def calculate_whisper_alignment(self, predicted_timestamps, true_timestamps, tolerance=0.5):
        """计算Whisper关键帧对齐准确率"""
        correct = 0
        total = len(true_timestamps)

        for pred, true in zip(predicted_timestamps, true_timestamps):
            if abs(pred - true) <= tolerance:
                correct += 1

        alignment_accuracy = correct / total
        self.metrics['whisper_alignment'] = alignment_accuracy
        return alignment_accuracy

    def calculate_latency_metrics(self, response_times):
        """计算延迟相关指标"""
        p50 = np.percentile(response_times, 50)
        p95 = np.percentile(response_times, 95)
        p99 = np.percentile(response_times, 99)

        self.metrics['latency'] = {
            'p50_ms': p50,
            'p95_ms': p95,
            'p99_ms': p99
        }
        return self.metrics['latency']

    def get_all_metrics(self):
        """获取所有计算的指标"""
        return self.metrics

    def clear_metrics(self):
        """清除所有指标"""
        self.metrics = {}