import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.clip_model import CLIPModel

class CLIPTrainer:
    def __init__(self, model_config):
        self.model = CLIPModel(model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.temperature = model_config.get('temperature', 0.07)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def compute_loss(self, image_features, text_features):
        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        labels = torch.arange(len(image_features)).to(self.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def train_step(self, batch):
        images, texts = batch
        images = images.to(self.device)
        texts = texts.to(self.device)

        # 获取特征表示
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(texts)

        # 计算双向对比损失
        loss_i2t = self.compute_loss(image_features, text_features)
        loss_t2i = self.compute_loss(text_features, image_features)
        loss = (loss_i2t + loss_t2i) / 2

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, train_loader, num_epochs):
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))