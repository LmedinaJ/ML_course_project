import os, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIG & SEED
# ─────────────────────────────────────────────────────────────────────────────
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING & PATHS
# ─────────────────────────────────────────────────────────────────────────────
model_version = "v01"
save_ckpt = f"model/best_model_{model_version}.pth"
metrics_out = f"model_metrics/model_{model_version}_metrics.csv"
epoch_log_out = f"model_metrics/model_{model_version}_epoch_log.csv"

os.makedirs("model", exist_ok=True)
os.makedirs("model_metrics", exist_ok=True)

labels_csv   = "npz_files_stack/labels.csv"
data_folder  = "npz_files_stack"
model_ckpt   = None
batch_size   = 8
learning_rate= 2e-4
pos_weight   = 0.633 / 0.367
max_epochs   = 10
patience     = 3
selected_modalities = ["IR069", "IR107"]

channel_index = {"VIS": 0, "IR069": 1, "IR107": 2}
channels = [channel_index[m] for m in selected_modalities]

# ─────────────────────────────────────────────────────────────────────────────
# 1) DATASET
# ─────────────────────────────────────────────────────────────────────────────
class SequenceDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.data_dir, row['filename'])
        x = np.load(path)['data']
        x = x[:, :, :, channels]  # keep selected channels
        x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)
        y = torch.tensor(row['label_yhat'], dtype=torch.float32)
        return x, y

# ─────────────────────────────────────────────────────────────────────────────
# 2) LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df_all = pd.read_csv(labels_csv)

df_tr, df_tmp = train_test_split(df_all, test_size=0.3,
                                 stratify=df_all["label_yhat"], random_state=42)
df_val, df_te = train_test_split(df_tmp, test_size=0.5,
                                 stratify=df_tmp["label_yhat"], random_state=42)

train_ds = SequenceDataset(df_tr, data_folder)
val_ds   = SequenceDataset(df_val, data_folder)
test_ds  = SequenceDataset(df_te, data_folder)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3) MODEL
# ─────────────────────────────────────────────────────────────────────────────
class CNNLSTMClassifier(nn.Module):
    def __init__(self, in_channels, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.encoder(x).view(B, T, -1)
        _, (h, _) = self.lstm(x)
        logit = self.head(h.squeeze(0)).squeeze(1)
        return logit

# ─────────────────────────────────────────────────────────────────────────────
# 4) TRAINING / EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        'acc':       accuracy_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0)
    }, (y_true, y_pred)

def plot_metrics(train_losses, val_f1s):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_f1s, label="Val F1")
    plt.legend(); plt.grid(); plt.show()

def plot_confmat(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def train(model):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    best_f1, no_improve = 0.0, 0
    train_losses, val_f1s, epoch_log = [], [], []
    overall_start = time.time()

    for epoch in range(1, max_epochs+1):
        model.train()
        epoch_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        train_losses.append(epoch_loss)

        val_metrics, _ = evaluate(model, val_loader, device)
        val_f1 = val_metrics['f1']
        val_f1s.append(val_f1)

        epoch_log.append({
            'epoch': epoch,
            'train_loss': epoch_loss,
            'val_f1': val_f1,
            'val_acc': val_metrics['acc'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
        })

        if val_f1 > best_f1:
            best_f1, no_improve = val_f1, 0
            torch.save(model.state_dict(), save_ckpt)
            status = "🌟 saved"
        else:
            no_improve += 1
            status = f"⚠️ no improv ({no_improve}/{patience})"
            if no_improve >= patience:
                print("⛔ Early stopping triggered.")
                break

        scheduler.step()
        print(f"Epoch {epoch} → val_f1 = {val_f1:.4f} → {status}")

    return train_losses, val_f1s, {
        "best_val_f1": best_f1,
        "final_val_acc": val_metrics['acc'],
        "final_val_precision": val_metrics['precision'],
        "final_val_recall": val_metrics['recall']
    }, epoch_log

# ─────────────────────────────────────────────────────────────────────────────
# 5) MAIN
# ─────────────────────────────────────────────────────────────────────────────
model = CNNLSTMClassifier(in_channels=len(channels)).to(device)

if model_ckpt and os.path.exists(model_ckpt):
    print("🔄 Loading saved model…")
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    val_summary = {}
    epoch_log = []
else:
    print("🚀 Training from scratch…")
    losses, f1s, val_summary, epoch_log = train(model)
    plot_metrics(losses, f1s)
    pd.DataFrame(epoch_log).to_csv(epoch_log_out, index=False)

print("\n🧪 Test evaluation:")
test_metrics, (y_t, y_p) = evaluate(model, test_loader)
print(test_metrics)
plot_confmat(y_t, y_p)

# ─────────────────────────────────────────────────────────────────────────────
# 6) SAVE METRICS
# ─────────────────────────────────────────────────────────────────────────────
all_metrics = {
    "model_version": model_version,
    "pos_weight": pos_weight,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "selected_modalities": ",".join(selected_modalities),
    **val_summary,
    **{f"test_{k}": v for k, v in test_metrics.items()}
}

pd.DataFrame([all_metrics]).to_csv(metrics_out, index=False)
print(f"\n📁 Saved summary metrics to: {metrics_out}")
print(f"📄 Saved epoch log to: {epoch_log_out}")
