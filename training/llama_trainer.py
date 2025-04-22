import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.llama_model import LLaMAModel
from transformers import LlamaTokenizer

class LLaMATrainer:
    def __init__(self, model_config):
        self.model = LLaMAModel(model_config)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_config['model_name'])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=model_config.get('learning_rate', 1e-5))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def prepare_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        return input_ids, attention_mask, labels

    def train_step(self, batch):
        self.model.train()
        input_ids, attention_mask, labels = self.prepare_batch(batch)

        # 前向传播
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, train_loader, num_epochs, eval_loader=None):
        best_eval_loss = float('inf')

        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}')

            if eval_loader:
                eval_loss = self.evaluate(eval_loader)
                print(f'Evaluation Loss: {eval_loss:.4f}')
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    self.save_model('best_llama_model.pt')

    def evaluate(self, eval_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                input_ids, attention_mask, labels = self.prepare_batch(batch)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()

        return total_loss / len(eval_loader)

    def generate(self, prompt, max_length=100, temperature=0.7):
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.95
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])