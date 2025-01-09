import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from PIL import Image
import json
from datetime import datetime

# Data preprocessing
class RiceDiseaseDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.data = pd.read_csv(csv_file)
        self.base_path = base_path
        self.transform = transform
        self.label_mapping = {label: idx for idx, label in enumerate(self.data['label'].unique())}
        
        # Store valid image paths
        self.valid_data = []
        for _, row in self.data.iterrows():
            img_path = os.path.join(self.base_path, row['label'], row['image_id'])
            if os.path.exists(img_path):
                self.valid_data.append({
                    'image_path': img_path,
                    'label': row['label']
                })
                
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        label = self.label_mapping[item['label']]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Model Trainer class
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, model_name):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(train_loader, desc=f'Training {self.model_name}'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        return total_loss / len(train_loader), 100. * correct / total
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f'Validating {self.model_name}'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        return total_loss / len(val_loader), 100. * correct / total
    
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }, os.path.join(save_dir, f'{self.model_name}_best.pth'))
        
        # Save training history
        with open(os.path.join(save_dir, f'{self.model_name}_history.json'), 'w') as f:
            json.dump(self.history, f)
            
        return self.history

# Model evaluator for test set
class ModelEvaluator:
    def __init__(self, models_dict, device, label_mapping):
        self.models = models_dict
        self.device = device
        self.label_mapping = label_mapping
        self.rev_label_mapping = {v: k for k, v in label_mapping.items()}
        
    def evaluate_image(self, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        results = {}
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                output = model(image)
                prob = torch.nn.functional.softmax(output, dim=1)
                pred_idx = output.argmax(1).item()
                pred_prob = prob[0][pred_idx].item()
                
                results[name] = {
                    'predicted_class': self.rev_label_mapping[pred_idx],
                    'confidence': pred_prob
                }
        
        return results
    
    def evaluate_test_set(self, test_dir, transform):
        results = []
        for image_name in tqdm(os.listdir(test_dir), desc='Evaluating test set'):
            if not image_name.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            image_path = os.path.join(test_dir, image_name)
            predictions = self.evaluate_image(image_path, transform)
            
            results.append({
                'image_name': image_name,
                **{f"{model_name}_prediction": pred['predicted_class']
                   for model_name, pred in predictions.items()},
                **{f"{model_name}_confidence": pred['confidence']
                   for model_name, pred in predictions.items()}
            })
            
        return pd.DataFrame(results)

# Visualization functions
def plot_training_comparison(histories, save_dir):
    metrics = ['loss', 'acc']
    phases = ['train', 'val']
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    for metric, ax in zip(metrics, axes):
        for model_name, history in histories.items():
            for phase in phases:
                key = f'{phase}_{metric}'
                ax.plot(history[key], label=f'{model_name} {phase}')
        
        ax.set_title(f'{metric.capitalize()} vs Epoch')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_comparison.png'))
    plt.close()

# Main training pipeline
def main():
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    config = {
        'csv_file': '/content/drive/MyDrive/Dataset/train.csv',
        'train_dir': '/content/drive/MyDrive/Dataset/train_images',
        'test_dir': '/content/drive/MyDrive/Dataset/test_images',
        'save_dir': '/content/drive/MyDrive/Dataset/model_results',
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 0.01
    }
    
    # Create save directory
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    full_dataset = RiceDiseaseDataset(config['csv_file'], config['train_dir'])
    num_classes = len(full_dataset.label_mapping)
    
    # Create model instances
    models = {
        'CustomRiceNet': CustomRiceNet(num_classes=num_classes),
        'EfficientViTNet': EfficientViTNet(num_classes=num_classes),
        'InceptionResNetModel': InceptionResNetModel(num_classes=num_classes),
        'ConvNextSwinModel': ConvNextSwinModel(num_classes=num_classes)
    }
    
    # Training histories
    histories = {}
    
    # Train each model
    for model_name, model in models.items():
        print(f"\nTraining {model_name}")
        model = model.to(device)
        
        # Create fresh train/val splits for each model
        train_idx, val_idx = train_test_split(
            range(len(full_dataset)),
            test_size=0.2,
            random_state=42
        )
        
        train_dataset = RiceDiseaseDataset(
            config['csv_file'],
            config['train_dir'],
            transform=train_transform
        )
        val_dataset = RiceDiseaseDataset(
            config['csv_file'],
            config['train_dir'],
            transform=val_transform
        )
        
        train_dataset = Subset(train_dataset, train_idx)
        val_dataset = Subset(val_dataset, val_idx)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # Initialize training components
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
        # Create trainer and train model
        trainer = ModelTrainer(
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            model_name
        )
        
        history = trainer.train(
            train_loader,
            val_loader,
            config['num_epochs'],
            config['save_dir']
        )
        
        histories[model_name] = history
    
    # Plot training comparisons
    plot_training_comparison(histories, config['save_dir'])
    
    # Load best models for testing
    best_models = {}
    for model_name, model in models.items():
        checkpoint = torch.load(
            os.path.join(config['save_dir'], f'{model_name}_best.pth')
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        best_models[model_name] = model
    
    # Evaluate on test set
    evaluator = ModelEvaluator(best_models, device, full_dataset.label_mapping)
    test_results = evaluator.evaluate_test_set(config['test_dir'], val_transform)
    
    # Save test results
    test_results.to_csv(
        os.path.join(config['save_dir'], 'test_predictions.csv'),
        index=False
    )
    
    print("\nTraining and evaluation completed!")
    print(f"Results saved in: {config['save_dir']}")

if __name__ == "__main__":
    main()