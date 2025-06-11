from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
TRAINING_DIR = BASE_DIR / 'training'
EVALUATION_DIR = BASE_DIR / 'evaluation'

# 数据配置
RAW_DATA_DIR = DATA_DIR / 'raw'
CACHE_DIR = DATA_DIR / 'cache'
VIDEO_URLS_FILE = DATA_DIR / 'video_urls.json'

# 模型配置
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'
CLIP_CHECKPOINT_DIR = CHECKPOINTS_DIR / 'clip'
LLAMA_CHECKPOINT_DIR = CHECKPOINTS_DIR / 'llama'

# 训练配置
CLIP_TRAINING_CONFIG = {
    'num_epochs': 5,
    'batch_size': 32,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    'max_grad_norm': 1.0
}

LLAMA_TRAINING_CONFIG = {
    'num_epochs': 3,
    'batch_size': 4,
    'learning_rate': 1e-5,
    'weight_decay': 0.01,
    'warmup_steps': 50,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 4
}

# 特征提取配置
FEATURE_EXTRACTION_CONFIG = {
    'max_text_length': 512,
    'image_size': 224,
    'audio_max_length': 30,  # 秒
    'audio_sample_rate': 16000
}

# 评估配置
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'cosine_similarity'
]

# 日志配置
LOGGING_CONFIG = {
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(BASE_DIR / 'training.log'),
            'formatter': 'standard',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 