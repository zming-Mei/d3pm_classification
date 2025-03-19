import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import wandb

class LabelDiffusion(nn.Module):
    def __init__(self, num_classes=10, T=1000, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.T = T
        self.device = device
        
        # 噪声调度参数
        self.register_buffer('beta', torch.linspace(0.0001, 0.02, T, device=device))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        
        # 图像特征编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256)
        ).to(device)
        
        # 时间步嵌入
        self.time_emb = nn.Embedding(T, 128).to(device)
        
        # MLP预测器
        self.mlp = nn.Sequential(
            nn.Linear(num_classes + 128 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ).to(device)

    def forward(self, x, y_t, t):
        """前向传播：输入图像x、加噪标签y_t和时间步t，输出类别概率"""
        cond = self.encoder(x)  # (batch_size, 256)
        t_emb = self.time_emb(t)  # (batch_size, 128)
        y_t = y_t.view(y_t.size(0), -1)  # 确保y_t是(batch_size, num_classes)
        input_features = torch.cat([y_t, t_emb, cond], dim=1)  # (batch_size, 10 + 128 + 256)
        return self.mlp(input_features)  # (batch_size, num_classes)

    def forward_process(self, y_0, t):
        """前向扩散过程：将原始标签y_0加噪到时间步t"""
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)  # (batch_size, 1)
        uniform = torch.ones_like(y_0, device=self.device) / self.num_classes
        return alpha_bar_t * y_0 + (1 - alpha_bar_t) * uniform

    def loss(self, x, y):
        """损失函数：计算预测类别与真实类别之间的交叉熵损失"""
        t = torch.randint(0, self.T, (x.size(0),), device=self.device)
        y_0_onehot = F.one_hot(y, self.num_classes).float()
        y_t = self.forward_process(y_0_onehot, t)
        pred_y0 = self(x, y_t, t)
        return F.cross_entropy(pred_y0, y)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project="d3pm_cifar10_classification")
    
    # 初始化模型
    model = LabelDiffusion(num_classes=10, T=1000, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    train_dataset = CIFAR10("./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    
    test_dataset = CIFAR10("./data", train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)

    # 训练循环
    for epoch in range(100):
        model.train()
        pbar = tqdm(train_loader)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            # 计算损失并优化
            loss = model.loss(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 日志记录
            wandb.log({"train_loss": loss.item()})
            pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                # 使用t=T-1和均匀分布作为y_t进行推理
                t = torch.full((x.size(0),), model.T - 1, device=device)
                y_t = torch.ones((x.size(0), model.num_classes), device=device) / model.num_classes
                pred = model(x, y_t, t)
                correct += (pred.argmax(1) == y).sum().item()
                total += x.size(0)
        
        acc = correct / total
        wandb.log({"val_acc": acc})
        print(f"Epoch {epoch} | Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()