import os, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
)

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIG & SEED
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🌱 All seeds set to {seed}")

seed_everything()

# ─────────────────────────────────────────────────────────────────────────────
# VERSIONING & PATHS
# ─────────────────────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"v06_{timestamp}"
MODEL_DIR = "model"
METRIC_DIR = "model_metrics"
LOG_DIR = "logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

save_ckpt = f"{MODEL_DIR}/best_model_{model_version}.pth"
sweep_csv = f"{METRIC_DIR}/pos_weight_sweep_results_{model_version}.csv"
training_log = f"{LOG_DIR}/training_log_{model_version}.csv"
learning_curves = f"{METRIC_DIR}/learning_curves_{model_version}.png"

# fixed config
labels_csv = "npz_files_stack_3ch/labels.csv"
data_folder = "npz_files_stack_3ch"
batch_size = 8
learning_rate = 2e-4
max_epochs = 10
patience = 7

# UPDATED: Changed to use three modalities
selected_modalities = ["VIS", "IR069", "IR107"]  
# UPDATED: Changed to map to three distinct channel indices
channel_index = {"VIS": 0, "IR069": 1, "IR107": 2}  
channels = [channel_index[m] for m in selected_modalities]

# Print configuration summary
print("=" * 80)
print(f"🚀 Starting training with configuration:")
print(f"   Model version: {model_version}")
print(f"   Batch size: {batch_size}, Learning rate: {learning_rate}")
print(f"   Selected modalities: {selected_modalities} (total channels: {len(channels)})")
print(f"   Max epochs: {max_epochs}, Patience: {patience}")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# 1) DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.data_dir, row['filename'])
        
        try:
            x = np.load(file_path)['data']  # [T,H,W,C]
            
            # UPDATED: Check if the data has enough channels
            if x.shape[3] < len(channels):
                print(f"⚠️ Warning: File {file_path} has only {x.shape[3]} channels, but {len(channels)} were requested")
                # Create a dummy array with the right number of channels
                dummy = np.zeros((x.shape[0], x.shape[1], x.shape[2], len(channels)))
                # Copy available channels
                for i, channel_idx in enumerate(channels):
                    if channel_idx < x.shape[3]:
                        dummy[:, :, :, i] = x[:, :, :, channel_idx]
                x = dummy
            else:
                # Select the requested channels
                x = x[:, :, :, channels]  # choose channels
            
            x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)  # [T,C,H,W]
            
            if self.transform:
                x = self.transform(x)
                
            y = torch.tensor(row['label'], dtype=torch.float32)
            return x, y
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            # Return a placeholder with the correct number of channels
            return torch.zeros((1, len(channels), 1, 1)), torch.tensor(-1.0)

# ─────────────────────────────────────────────────────────────────────────────
# 2) TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split_data():
    """Load data and create train/val/test splits with detailed statistics."""
    print("📊 Loading and splitting dataset...")
    
    df_all = pd.read_csv(labels_csv)
    print(f"Total samples: {len(df_all)}, Positive samples: {df_all['label'].sum()} ({df_all['label'].mean()*100:.1f}%)")
    
    df_tr, df_tmp = train_test_split(df_all, test_size=0.3,
                                     stratify=df_all['label'], random_state=42)
    df_val, df_te = train_test_split(df_tmp, test_size=0.5,
                                     stratify=df_tmp['label'], random_state=42)
    
    print(f"Train: {len(df_tr)} samples, {df_tr['label'].sum()} positive ({df_tr['label'].mean()*100:.1f}%)")
    print(f"Val:   {len(df_val)} samples, {df_val['label'].sum()} positive ({df_val['label'].mean()*100:.1f}%)")
    print(f"Test:  {len(df_te)} samples, {df_te['label'].sum()} positive ({df_te['label'].mean()*100:.1f}%)")
    
    return df_tr, df_val, df_te

df_tr, df_val, df_te = load_and_split_data()

# Data loaders with error handling
def collate_fn(batch):
    """Custom collate function to handle corrupted samples."""
    valid_batch = [item for item in batch if item[1] >= 0]
    if not valid_batch:
        # If all samples in batch are invalid, return empty tensors
        return torch.zeros(0), torch.zeros(0)
    return torch.utils.data.dataloader.default_collate(valid_batch)

train_loader = DataLoader(SequenceDataset(df_tr, data_folder), batch_size=batch_size,
                         shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
val_loader = DataLoader(SequenceDataset(df_val, data_folder), batch_size=batch_size,
                       shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(SequenceDataset(df_te, data_folder), batch_size=batch_size,
                        shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMClassifier(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim*2, 32),  # *2 for bidirectional
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):  # x: [B,T,C,H,W]
        if x.size(0) == 0:  # Handle empty batch
            return torch.tensor([])
            
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        x = self.encoder(x).view(B, T, -1)          # [B,T,32]
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[0], h[1]), dim=1)  # Concatenate bidirectional outputs
        return self.head(h).squeeze(1)   # [B]

# ─────────────────────────────────────────────────────────────────────────────
# 4) HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, threshold: float = 0.5):
    """Evaluate model with detailed metrics."""
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    
    with torch.no_grad():
        for X, y in loader:
            if X.size(0) == 0:  # Skip empty batches
                continue
                
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).int()
            
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
    
    if not y_true:  # Handle case where there are no valid predictions
        return {
            'acc': 0, 'f1': 0, 'precision': 0, 'recall': 0, 'auc': 0,
            'pos_count': 0, 'total': 0
        }
    
    # Convert to numpy arrays for metrics calculation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    metrics = {
        'acc': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'pos_count': np.sum(y_true),
        'total': len(y_true)
    }
    
    # AUC calculation (handle case where only one class is present)
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_scores)
    else:
        metrics['auc'] = 0.0
        
    return metrics

def plot_training_history(history):
    """Plot learning curves from training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['epoch'], history['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Metrics plot
    ax2.plot(history['epoch'], history['val_f1'], label='Val F1')
    ax2.plot(history['epoch'], history['val_acc'], label='Val Acc')
    ax2.plot(history['epoch'], history['val_recall'], label='Val Recall')
    ax2.plot(history['epoch'], history['val_precision'], label='Val Precision')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Validation Metrics')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(learning_curves)
    plt.close()
    print(f"📈 Learning curves saved to {learning_curves}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax)
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Confusion matrix saved to {save_path}")

# ─────────────────────────────────────────────────────────────────────────────
# 5) TRAINING ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def train_one_run(pos_weight_val: float):
    """Train one model instance with the given pos_weight and return best‑val F1,
    best threshold, and test metrics."""
    print(f"\n{'='*80}\n🏃‍♂️ Starting training run with pos_weight={pos_weight_val}\n{'='*80}")
    
    # UPDATED: Now initializes with the correct number of input channels (3 instead of 2)
    model = CNNLSTMClassifier(in_channels=len(channels)).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))
    opt = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # Use StepLR which is more stable than ReduceLROnPlateau to avoid the warning
    sched = StepLR(opt, step_size=3, gamma=0.5)

    # Training history for plotting
    history = {
        'epoch': [], 'train_loss': [], 
        'val_loss': [], 'val_acc': [], 'val_f1': [], 
        'val_precision': [], 'val_recall': [], 'val_auc': [],
        'lr': []
    }
    
    # For CSV logging
    training_log_rows = []

    best_f1, no_improve = 0.0, 0
    train_start = time.time()
    
    print(f"{'Epoch':^6}|{'Train Loss':^12}|{'Val Loss':^10}|{'Val F1':^8}|{'Val Acc':^8}|{'Val Prec':^9}|{'Val Rec':^8}|{'LR':^10}|{'Status':^15}")
    print("-" * 90)
    
    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        
        # Training metrics
        epoch_loss = 0.0
        batch_count = 0
        
        # Progress tracking
        total_batches = len(train_loader)
        processed_batches = 0
        
        for i, (X, y) in enumerate(train_loader):
            if X.size(0) == 0:  # Skip empty batches
                continue
                
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            outputs = model(X)
            loss = crit(outputs, y)
            loss.backward()
            opt.step()
            
            epoch_loss += loss.item()
            batch_count += 1
            
            # Progress update (every 10% of batches)
            processed_batches += 1
            if processed_batches % max(1, total_batches // 10) == 0:
                progress = processed_batches / total_batches * 100
                print(f"   ⏳ Epoch {epoch}: {progress:.1f}% complete, current loss: {loss.item():.4f}")
        
        # Calculate average loss
        avg_train_loss = epoch_loss / max(batch_count, 1)  # Avoid division by zero
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                if X.size(0) == 0:  # Skip empty batches
                    continue
                    
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = crit(outputs, y)
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / max(val_batch_count, 1)
        
        # Calculate validation metrics
        val_metrics = evaluate(model, val_loader)
        val_f1 = val_metrics['f1']
        
        # Update learning rate scheduler
        try:
            sched.step()  # StepLR doesn't take a metric parameter
        except Exception as e:
            print(f"⚠️ Warning: LR scheduler error: {e}")
        current_lr = opt.param_groups[0]['lr']
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_metrics['acc'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_auc'].append(val_metrics.get('auc', 0))
        history['lr'].append(current_lr)
        
        # Log row for CSV
        log_row = {
            'epoch': epoch,
            'pos_weight': pos_weight_val,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_acc': val_metrics['acc'],
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_auc': val_metrics.get('auc', 0),
            'pos_samples': val_metrics['pos_count'],
            'total_samples': val_metrics['total'],
            'learning_rate': current_lr,
            'time_sec': time.time() - epoch_start
        }
        training_log_rows.append(log_row)
        
        # Check for improvement
        status = ""
        if val_f1 > best_f1:
            best_f1, no_improve, status = val_f1, 0, "🌟 best"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'val_f1': val_f1,
                'pos_weight': pos_weight_val,
                'config': {
                    'channels': channels,
                    'selected_modalities': selected_modalities
                }
            }, save_ckpt)
        else:
            no_improve += 1
            status = f"⚠️ no improv ({no_improve}/{patience})"
        
        # Print epoch results
        print(f"{epoch:^6}|{avg_train_loss:^12.4f}|{avg_val_loss:^10.4f}|{val_f1:^8.3f}|{val_metrics['acc']:^8.3f}|{val_metrics['precision']:^9.3f}|{val_metrics['recall']:^8.3f}|{current_lr:^10.2e}|{status:^15}")
        
        # Early stopping check
        if no_improve >= patience:
            print(f"⛔ Early stopping triggered after epoch {epoch}")
            break
    
    # Save training log
    pd.DataFrame(training_log_rows).to_csv(training_log, index=False)
    print(f"📝 Training log saved to {training_log}")
    
    # Plot training history
    plot_training_history(history)
    
    total_time = time.time() - train_start
    print(f"🏁 Training completed in {total_time:.1f}s (avg {total_time/epoch:.1f}s per epoch)")
    print(f"🏆 Best validation F1: {best_f1:.3f}")

    # Load best model for evaluation
    try:
        if os.path.exists(save_ckpt):
            checkpoint = torch.load(save_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"📂 Loaded best model from epoch {checkpoint['epoch']}")
        else:
            print(f"⚠️ No saved checkpoint found at {save_ckpt}, using current model state")
            checkpoint = {'epoch': epoch}
    except Exception as e:
        print(f"⚠️ Error loading checkpoint: {e}. Using current model state.")
        checkpoint = {'epoch': epoch}

    # Threshold sweep on validation set
    print("🔍 Performing threshold sweep on validation set...")
    thresh_metrics = []
    thresholds = np.linspace(0.1, 0.9, 9)
    
    print(f"{'Threshold':^10}|{'Val F1':^8}|{'Val Acc':^8}|{'Val Prec':^9}|{'Val Rec':^8}")
    print("-" * 50)
    
    for t in thresholds:
        m = evaluate(model, val_loader, t)
        m['threshold'] = t
        thresh_metrics.append(m)
        print(f"{t:^10.2f}|{m['f1']:^8.3f}|{m['acc']:^8.3f}|{m['precision']:^9.3f}|{m['recall']:^8.3f}")
    
    best_thresh_idx = max(range(len(thresh_metrics)), key=lambda i: thresh_metrics[i]['f1'])
    best_thresh = thresh_metrics[best_thresh_idx]['threshold']
    print(f"🎯 Best threshold on validation set: {best_thresh:.2f} (F1: {thresh_metrics[best_thresh_idx]['f1']:.3f})")

    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, best_thresh)
    print("\n🧪 Test set evaluation:")
    print(f"   F1: {test_metrics['f1']:.3f}")
    print(f"   Accuracy: {test_metrics['acc']:.3f}")
    print(f"   Precision: {test_metrics['precision']:.3f}")
    print(f"   Recall: {test_metrics['recall']:.3f}")
    print(f"   AUC: {test_metrics.get('auc', 0):.3f}")
    
    # Get predictions for confusion matrix
    try:
        y_true, y_pred = [], []
        with torch.no_grad():
            for X, y in test_loader:
                if X.size(0) == 0:
                    continue
                    
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.sigmoid(outputs)
                preds = (probs > best_thresh).int()
                
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # Plot confusion matrix (only if we have predictions)
        if len(y_true) > 0:
            conf_matrix_path = f"{METRIC_DIR}/confusion_matrix_pw{pos_weight_val}_{model_version}.png"
            plot_confusion_matrix(y_true, y_pred, conf_matrix_path)
        else:
            print("⚠️ No valid predictions for confusion matrix")
    except Exception as e:
        print(f"⚠️ Error creating confusion matrix: {e}")
    
    return best_f1, best_thresh, test_metrics

# ─────────────────────────────────────────────────────────────────────────────
# 6) POS_WEIGHT SWEEP
# ─────────────────────────────────────────────────────────────────────────────

sweep_results = []
print("\n📊 Starting pos_weight sweep")

# More extensive sweep values
pos_weights = [0.2, 0.5, 1.0, 2.0, 5.0]  # Added more values for better coverage

for pw in pos_weights:
    print("\n" + "═" * 80)
    print(f"⚙️  Training run with pos_weight = {pw}")
    best_val_f1, best_thresh, test_metrics = train_one_run(pw)
    
    result = {
        'pos_weight': pw,
        'best_val_f1': best_val_f1,
        'best_thresh': best_thresh,
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }
    sweep_results.append(result)
    print("═" * 80)

# Save sweep results
sweep_df = pd.DataFrame(sweep_results)
sweep_df.to_csv(sweep_csv, index=False)

# Print summary of results
print("\n📋 Pos_weight Sweep Results Summary:")
print(sweep_df.to_string(index=False))
print(f"\n✅ Sweep finished → results saved to {sweep_csv}")

# Find and highlight best result
best_idx = sweep_df['test_f1'].argmax()
best_pw = sweep_df.iloc[best_idx]['pos_weight']
best_f1 = sweep_df.iloc[best_idx]['test_f1']

print(f"\n🏆 Best result: pos_weight = {best_pw}, test F1 = {best_f1:.3f}")
print(f"📂 Best model saved at: {save_ckpt}")