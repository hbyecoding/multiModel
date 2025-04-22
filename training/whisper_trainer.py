import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.whisper_model import WhisperModel

class WhisperTrainer:
    def __init__(self, model_config):
        self.model = WhisperModel(model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, batch):
        audio_features, transcripts = batch
        audio_features = audio_features.to(self.device)
        transcripts = transcripts.to(self.device)

        # 前向传播
        outputs = self.model(audio_features)
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), transcripts.view(-1))

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, train_loader, num_epochs, validation_loader=None):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}')

            # 验证
            if validation_loader:
                val_loss = self.evaluate(validation_loader)
                print(f'Validation Loss: {val_loss:.4f}')
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model('best_whisper_model.pt')

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                audio_features, transcripts = batch
                audio_features = audio_features.to(self.device)
                transcripts = transcripts.to(self.device)

                outputs = self.model(audio_features)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), transcripts.view(-1))
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def transcribe(self, audio_features):
        self.model.eval()
        with torch.no_grad():
            audio_features = audio_features.to(self.device)
            outputs = self.model(audio_features)
            return self.model.decode(outputs)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])