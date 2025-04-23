import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

df_stats = pd.read_csv("csv_data/npz_scaled_stats.csv")
sensor_minmax = (
    df_stats.groupby("sensor")[["min", "max"]]
            .agg({"min": "min", "max": "max"})
            .to_dict("index")
)

# Model definition (must match the one used for training)
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

# Dataset for inference
class InferenceDataset(Dataset):
    def __init__(self, file_paths, data_dir=None, channels=None, sensor_stats=None):
        """
        Args:
            file_paths: List of .npz file paths or a DataFrame with 'filename' column
            data_dir: Path to directory containing .npz files
            channels: List of channels to load
            sensor_stats: Dict of sensor -> {"min": float, "max": float}
        """
        if isinstance(file_paths, pd.DataFrame):
            self.file_paths = file_paths['filename'].tolist()
        else:
            self.file_paths = file_paths

        self.data_dir = data_dir
        self.channels = channels or [0, 1, 2]
        self.sensor_stats = sensor_stats or {}

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.data_dir, file_path) if self.data_dir else file_path
        try:
            x = np.load(full_path)['data']  # [T,H,W,C]
            
            if x.shape[3] < len(self.channels):
                print(f"Warning: File {file_path} has {x.shape[3]} channels, expected {len(self.channels)}")
                dummy = np.zeros((x.shape[0], x.shape[1], x.shape[2], len(self.channels)))
                for i, channel_idx in enumerate(self.channels):
                    if channel_idx < x.shape[3]:
                        dummy[:, :, :, i] = x[:, :, :, channel_idx]
                x = dummy
            else:
                x = x[:, :, :, self.channels]

            # Map channels to sensor names (assuming channels[0] = vis, channels[1] = ir069, channels[2] = ir107)
            channel_sensor_map = {0: 'vis', 1: 'ir069', 2: 'ir107'}
            
            # Rescale each channel using its corresponding sensor stats
            for i in range(len(self.channels)):
                sensor = channel_sensor_map.get(i)
                stats = self.sensor_stats.get(sensor)
                if stats:
                    x[:, :, :, i] = rescale_with_minmax(x[:, :, :, i], stats['min'], stats['max'])
                else:
                    print(f"Warning: No stats found for sensor '{sensor}' in channel {i}, skipping rescaling")

            x = torch.tensor(x.transpose(0, 3, 1, 2), dtype=torch.float32)  # [T,C,H,W]
            return x, file_path
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros((1, len(self.channels), 1, 1)), file_path

# Rescaling function
def rescale_with_minmax(x, lo, hi, eps=1e-8):
    rng = max(hi - lo, eps)
    return np.clip((x - lo) / rng, 0.0, 1.0)

def load_model(model_path, device='cuda'):
    """Load a trained model from a checkpoint file."""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract configuration from checkpoint
    config = checkpoint.get('config', {})
    channels = config.get('channels', [0, 1, 2])  # Default to first three channels
    
    # Print model info
    print(f"Loading model from: {model_path}")
    print(f"Trained until epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation F1: {checkpoint.get('val_f1', 'unknown')}")
    print(f"Using channels: {channels}")
    print(f"Selected modalities: {config.get('selected_modalities', 'unknown')}")
    
    # Initialize model with the right number of input channels
    model = CNNLSTMClassifier(in_channels=len(channels))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, channels, checkpoint.get('config', {})

def predict(model, dataloader, threshold=0.5, device='cuda'):
    """Run inference on data and return predictions."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X, paths in dataloader:
            if X.size(0) == 0:  # Skip empty batches
                continue
                
            X = X.to(device)
            outputs = model(X)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            for i, (prob, path) in enumerate(zip(probs, paths)):
                predictions.append({
                    'file_path': path,
                    'filename': os.path.basename(path),
                    'probability': float(prob),
                    'prediction': 1 if prob > threshold else 0
                })
    
    return pd.DataFrame(predictions)

def visualize_predictions(df, output_dir=None):
    """Create visualizations for the predictions."""
    # 1. Distribution of probabilities
    plt.figure(figsize=(10, 6))
    plt.hist(df['probability'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Prediction Probabilities')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
        plt.close()
    else:
        plt.show()
    
    # 2. Count plot for predictions
    plt.figure(figsize=(8, 6))
    counts = df['prediction'].value_counts()
    plt.bar(['Negative (0)', 'Positive (1)'], 
            [counts.get(0, 0), counts.get(1, 0)], 
            color=['skyblue', 'salmon'])
    plt.title('Prediction Counts')
    plt.ylabel('Count')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'prediction_counts.png'))
        plt.close()
    else:
        plt.show()
    
    # If we have ground truth, plot confusion matrix
    if 'label' in df.columns:
        cm = confusion_matrix(df['label'], df['prediction'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 8))
        disp.plot(ax=ax)
        plt.title('Confusion Matrix')
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
            plt.close()
        else:
            plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint file')
    parser.add_argument('--data_dir', type=str, help='Directory containing data files')
    parser.add_argument('--input_list', type=str, help='Text file with one filename per line')
    parser.add_argument('--input_csv', type=str, help='CSV file with a "filename" column')
    parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold for classification')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--output', type=str, default='predictions.csv', help='Output file for predictions')
    parser.add_argument('--vis_dir', type=str, help='Directory for visualization outputs')
    parser.add_argument('--ground_truth', type=str, help='CSV with ground truth (must have filename and label columns)')
    args = parser.parse_args()
    
    # Ensure at least one input source is provided
    if not (args.input_list or args.input_csv or args.data_dir):
        parser.error("At least one of --input_list, --input_csv, or --data_dir must be provided")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, channels, config = load_model(args.model_path, device)
    
    # Prepare file list for inference
    if args.input_csv:
        print(f"Loading file list from CSV: {args.input_csv}")
        df = pd.read_csv(args.input_csv)
        file_paths = df
    elif args.input_list:
        print(f"Loading file list from text file: {args.input_list}")
        with open(args.input_list, 'r') as f:
            file_paths = [line.strip() for line in f if line.strip()]
    else:
        print(f"Scanning directory for NPZ files: {args.data_dir}")
        file_paths = [f for f in os.listdir(args.data_dir) if f.endswith('.npz')]
    
    print(f"Found {len(file_paths)} files for inference")
    
    # Create dataset and dataloader
    dataset = InferenceDataset(file_paths, args.data_dir, channels, sensor_stats=sensor_minmax)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Run inference
    print(f"Running inference with threshold: {args.threshold}")
    predictions_df = predict(model, dataloader, args.threshold, device)
    
    # If ground truth is provided, merge it with predictions
    if args.ground_truth:
        print(f"Loading ground truth from: {args.ground_truth}")
        gt_df = pd.read_csv(args.ground_truth)
        predictions_df = predictions_df.merge(
            gt_df[['filename', 'label']], 
            on='filename', 
            how='left'
        )
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        metrics = {
            'accuracy': accuracy_score(predictions_df['label'], predictions_df['prediction']),
            'f1': f1_score(predictions_df['label'], predictions_df['prediction']),
            'precision': precision_score(predictions_df['label'], predictions_df['prediction']),
            'recall': recall_score(predictions_df['label'], predictions_df['prediction'])
        }
        
        print("\nEvaluation metrics:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
    
    # Save predictions
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    predictions_df.to_csv(args.output, index=False)
    print(f"Predictions saved to: {args.output}")
    
    # Create visualizations
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"Creating visualizations in: {args.vis_dir}")
        visualize_predictions(predictions_df, args.vis_dir)
    
    print("Inference completed successfully")

if __name__ == "__main__":
    main()