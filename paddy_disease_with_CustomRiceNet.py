
from google.colab import drive
drive.mount('/content/drive')

# Check if the mount was successful by checking if the directory exists
import os
google_drive_connected = os.path.exists('/content/drive')

# Show message google drive connect or not
print("Google Drive Connected:", google_drive_connected)

# Require library

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from PIL import Image
from tqdm import tqdm

# Data preprocesing custom class

class RiceDiseaseDataset(Dataset):
  def __init__(self, csv_file, base_path, transform=None):
    self.data = pd.read_csv(csv_file)
    self.base_path = base_path
    self.transform = transform
    self.label_mapping = {label: idx for idx, label in enumerate(self.data['label'].unique())}

    # print Dataset information:
    print(f"Number of classes: {len(self.label_mapping)}")
    print("Label Mapping:", self.label_mapping)

    # Verify and store valid image paths
    self.valid_data = []
    for _, row in self.data.iterrows():
      img_name = row['image_id']
      label = row['label']

      # Check in label subdirectory
      potential_path = os.path.join(self.base_path, label, img_name)
      if os.path.exists(potential_path):
        self.valid_data.append({
            'image_path': potential_path,
            'label': label
        })

    print(f"Number of valid images: {len(self.valid_data)}")

  def __len__(self):
    return len(self.valid_data)

  def __getitem__(self, idx):
    item = self.valid_data[idx]
    img = Image.open(item['image_path']).convert('RGB')
    label = self.label_mapping[item['label']]

    if self.transform:
      img = self.transform(img)

    return img, label

# custom RiceNet Model using combination of Resnet18 with Denenet121

class CustomRiceNet(nn.Module):
  def __init__(self, num_classes):
    super(CustomRiceNet, self).__init__()

    # Feature Edxtractors
    # Resnet18
    self.resnet_features = models.resnet18(weights='IMAGENET1K_V1')
    renet_out = self.resnet_features.fc.in_features
    self.resnet_features.fc = nn.Identity()

    # Densenet121
    self.densenet_features = models.densenet121(weights='IMAGENET1K_V1')
    densenet_out = self.densenet_features.classifier.in_features
    self.densenet_features.classifier = nn.Identity()

    # Combine features
    combined_features = renet_out + densenet_out

    # show total parameter of the combine
    print(f"Total parameters of combined features: {combined_features}")

    # Attention Mechanism
    self.attention = nn.Sequential(
        nn.Linear(combined_features, 512),
        nn.ReLU(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    # Feature fusion Layers
    self.fusion = nn.Sequential(
        nn.Linear(combined_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
    )

    # Final Classification Layer
    self.classifier = nn.Linear(64, num_classes)

  # Forward Pass
  def forward(self, x):
    # Extract featiure from both networks
    resnet_out = self.resnet_features(x)
    densenet_out = self.densenet_features(x)

    # Concatenate features
    combined_features = torch.cat((resnet_out, densenet_out), dim=1)

    # Apply attention
    attention_weights = self.attention(combined_features)
    weighted_features = attention_weights * combined_features

    # Fuse features
    fused_features = self.fusion(weighted_features)

    # Final Classification
    output = self.classifier(fused_features)
    return output

class CustomRiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomRiceLoss, self).__init__()
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # Calculate Cross-Entropy Loss
        return self.criterion(outputs, targets)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30):
    best_val_loss = float('inf')
    training_history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Train Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in tqdm(train_loader, desc='Training'):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc='Validation'):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, 'best_model.pth')

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Store results
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)


def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.show()

# Call main Function

def main():
  # set random seed
  torch.manual_seed(42)

  # setup device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  # Define Transforms
  train_trainsforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomRotation(30),
      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])


  # Validation transform
  val_transforms = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.456], std=[0.229, 0.224, 0.225])
  ])


  # Create datasets
  csv_file = '/content/drive/MyDrive/Dataset/train.csv'
  base_path = '/content/drive/MyDrive/Dataset/train_images'


  full_dataset = RiceDiseaseDataset(csv_file=csv_file, base_path=base_path, transform=None)

  # Split indices
  train_indices, val_indices = train_test_split(
      range(len(full_dataset)),
      test_size=0.2,
      random_state=42,
      # stratify=[full_dataset[i]['label'] for i in range(len(full_dataset))]
  )

  # Create train and validation datasets
  train_dataset = RiceDiseaseDataset(csv_file=csv_file, base_path=base_path, transform=train_trainsforms)
  val_dataset = RiceDiseaseDataset(csv_file=csv_file, base_path=base_path, transform=val_transforms)

  # Create subsets
  train_dataset = Subset(train_dataset, train_indices)
  val_dataset = Subset(val_dataset, val_indices)

  # create Dataloaders
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

  # Initialize model
  model = CustomRiceNet(num_classes=len(full_dataset.label_mapping)).to(device)

  # Initilize loss and optimizer
  criterion = CustomRiceLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

  # Train the model
  training_history = train_model(
      model = model,
      train_loader = train_loader,
      val_loader = val_loader,
      criterion = criterion,
      optimizer = optimizer,
      scheduler = scheduler,
      device = device,
      num_epochs = 30
  )

  # save training history
  save_dir = '/content/drive/MyDrive/Dataset/training_history.csv'
  pd.DataFrame(training_history).to_csv(save_dir, index=False)


  # plot training history
  plot_training_history(training_history)


# call Main function
if __name__ == "__main__":
  main()