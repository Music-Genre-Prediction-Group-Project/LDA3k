import datetime
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

num_topics = 512
num_epochs = 30
batch_size = 128
hidden_sizes = [512,256,128]

# Generate text to describe layer in output's filename
layers_text = ""
for size in hidden_sizes:
    layers_text += f"{size}-"
layers_text = layers_text[:-1]

dataset_path = f'dataset/topics/genly3k_topics{num_topics}.csv'
output_path = f'classifiers/genre{num_topics}_{layers_text}_{timestamp}.pth'

class LyricsDataset(Dataset):
    def __init__(self, csv_file):
        self.data_frame = pd.read_csv(csv_file)
        self.genres = self.data_frame['genre'].unique()
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.genres)}
        self.idx_to_genre = {idx: genre for genre, idx in self.genre_to_idx.items()}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        topics = self.data_frame.iloc[idx, 2:].values.astype('float32')
        genre = self.data_frame.iloc[idx, 1]
        label = self.genre_to_idx[genre]
        sample = {'topics': torch.tensor(topics), 'label': torch.tensor(label)}
        return sample

# Load the dataset
dataset = LyricsDataset(csv_file=dataset_path)

# Split the dataset into training and validation sets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class GenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(GenreClassifier, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        
        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return self.softmax(x)

# Set device to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Initialize the model
input_size = len(dataset[0]['topics'])  # Number of topics
print(f"Input Size: {input_size}")
output_size = len(dataset.genres)       # Number of genres
print(f"Output Size: {output_size}")
model = GenreClassifier(
    input_size=input_size, 
    hidden_sizes=hidden_sizes,
    output_size=output_size
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

training_losses = []
validation_losses = []
best_accuracy = 0
has_peaked = False
best_epoch = 0
best_model_state = model.state_dict()
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs = batch['topics'].to(device)
        labels = batch['label'].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    training_losses.append(epoch_loss)

    # Validation phase
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['topics'].to(device)
            labels = batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_model_state = model.state_dict()
        best_epoch = epoch+1 # 0 is reserved for no best
        if has_peaked:
            print(f"New best at Epoch {epoch+1}: ")
    else:
        has_peaked = True

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, '
          f'Val Loss: {val_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}')

print(f"Epoch {best_epoch} is best with {best_accuracy*100:.2f}% accuracy")

os.makedirs(os.path.dirname(output_path), exist_ok=True)
torch.save(best_model_state, output_path)
