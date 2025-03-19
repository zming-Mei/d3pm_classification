import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb
from d3pm_runner import D3PM
from dit import DiT_Llama

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, feature_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 768),  # 第一层全连接
            nn.BatchNorm1d(768),         # 批量归一化
            nn.ReLU(),                   # 激活函数
            nn.Dropout(0.3),             # Dropout 防止过拟合
            nn.Linear(768, 512),         # 第二层全连接
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),         # 第三层全连接
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)  # 输出层，10 个类别
        )
    
    def forward(self, x):
        return self.fc(x)

# ... 导入语句和 Classifier 定义保持不变 ...

if __name__ == "__main__":
    wandb.init(project="d3pm_cifar10_classification")

    N = 8
    feature_dim = 1024  # 与 DiT_Llama 的 dim 匹配
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化 D3PM 和分类器
    d3pm = D3PM(
        DiT_Llama(3, N, dim=feature_dim), 1000, num_classes=N, hybrid_loss_coeff=0.0
    ).to(device)
    classifier = Classifier(feature_dim).to(device)

    # 数据加载
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    train_dataset = CIFAR10("./data", train=True, download=True, transform=transform)
    val_dataset = CIFAR10("./data", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=16)

    # 优化器和调度器
    optim_d3pm = torch.optim.AdamW(d3pm.x0_model.parameters(), lr=2e-5)
    optim_cls = torch.optim.AdamW(classifier.parameters(), lr=5e-5)
    scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optim_cls, T_max=100)
    cls_criterion = nn.CrossEntropyLoss()

    # 训练循环
    n_epoch = 1000
    global_step = 0

    for i in range(n_epoch):
        pbar = tqdm(train_dataloader)
        loss_ema = None
        for x, cond in pbar:
            x = x.to(device)
            cond = cond.to(device)
            x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)

            # 训练 D3PM
            optim_d3pm.zero_grad()
            loss, info = d3pm(x_cat, cond)
            loss.backward()
            norm_d3pm = torch.nn.utils.clip_grad_norm_(d3pm.x0_model.parameters(), 5.0)
            optim_d3pm.step()

            # 提取特征并训练分类器
            with torch.no_grad():
                t = torch.zeros(x.size(0), dtype=torch.long, device=device)
                features = d3pm.x0_model.get_feature(x_cat, t, cond)  # 使用新方法

            optim_cls.zero_grad()
            logits = classifier(features)
            cls_loss = cls_criterion(logits, cond)
            cls_loss.backward()
            norm_cls = torch.nn.utils.clip_grad_norm_(classifier.parameters(), 5.0)
            optim_cls.step()

            # ... 其他日志和验证逻辑保持不变 ...

            global_step += 1
            scheduler_cls.step()

            # 更新损失的指数移动平均
            if loss_ema is None:
                loss_ema = loss.item()
                cls_loss_ema = cls_loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()
                cls_loss_ema = 0.99 * cls_loss_ema + 0.01 * cls_loss.item()

            # 更新进度条描述
            pbar.set_description(
                f"diff_loss: {loss_ema:.4f}, class_loss: {cls_loss_ema:.4f}, "
                f"norm_d3pm: {norm_d3pm:.4f}, norm_cls: {norm_cls:.4f}"
            )

            # 记录到 Weights & Biases
            if global_step % 10 == 0:
                wandb.log({
                    "gen_loss": loss.item(),
                    "cls_loss": cls_loss.item(),
                    "grad_norm_d3pm": norm_d3pm,
                    "grad_norm_cls": norm_cls,
                })

            # 验证和样本生成
            if global_step % 1000 == 1:
                d3pm.eval()
                classifier.eval()

                # 生成样本（保留原始功能）
                with torch.no_grad():
                    cond_val = torch.arange(0, 16).to(device) % 10
                    init_noise = torch.randint(0, N, (16, 3, 32, 32)).to(device)
                    images = d3pm.sample_with_image_sequence(init_noise, cond_val, stride=40)
                    gif = []
                    for image in images:
                        x_from_dataloader = x_cat[:16].cpu() / (N - 1)
                        this_image = image.float().cpu() / (N - 1)
                        all_images = torch.cat([x_from_dataloader, this_image], dim=0)
                        x_as_image = make_grid(all_images, nrow=4)
                        img = (x_as_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))
                    gif[0].save(
                        f"contents/sample_{global_step}.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )
                    wandb.log({"sample": wandb.Image(gif[-1])})

                # 计算验证集准确率
                correct = 0
                total = 0
                for x_val, cond_val in val_dataloader:
                    x_val = x_val.to(device)
                    cond_val = cond_val.to(device)
                    x_val_cat = (x_val * (N - 1)).round().long().clamp(0, N - 1)
                    with torch.no_grad():
                        t_val = torch.zeros(x_val.size(0), dtype=torch.long, device=device)
                        features_val = d3pm.x0_model.get_feature(x_val_cat, t_val, cond_val)
                        logits_val = classifier(features_val)
                        pred = logits_val.argmax(dim=1)
                        correct += (pred == cond_val).sum().item()
                        total += cond_val.size(0)
                accuracy = correct / total
                wandb.log({"val_accuracy": accuracy})
                print(f"test_accuracy: {accuracy*100:.4f}")

                d3pm.train()
                classifier.train()

            global_step += 1
            scheduler_cls.step()  # 更新学习率

            # # 保存检查点
            # if global_step % 1000 == 0:
            #     torch.save(d3pm.state_dict(), f"checkpoints/d3pm_step_{global_step}.pth")
            #     torch.save(classifier.state_dict(), f"checkpoints/cls_step_{global_step}.pth")