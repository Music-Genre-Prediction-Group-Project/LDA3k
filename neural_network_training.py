import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from lyric_dataset import LyricsDataset
from genre_classifier import GenreClassifier
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

num_topics = 128
learning_rate = 0.001
num_epochs = 20
batch_size = 128
hidden_sizes = [128,128]

# Generate text to describe layer in output's filename
layers_text = ""
for size in hidden_sizes:
    layers_text += f"{size}_"
layers_text = layers_text[:-1]

dataset_path = f'dataset/topics_train/genly3k_train{num_topics}.csv'
output_path = f'classifiers/official/genre_partial{num_topics}-{layers_text}-{timestamp}.pth'
# val_output_path = f'classifiers/genre{num_topics}_{layers_text}_{timestamp}_val.csv'
test_set_path = f'dataset/topics_test/genly3k_test{num_topics}.csv'

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load the dataset
dataset = LyricsDataset(csv_file=dataset_path)

# # Test set: This is kind of awkward because it's a retroactive choice
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# remain_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Split the dataset into training and validation sets (e.g., 80% train, 20% val)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Saving validation set
# original_df = pd.read_csv(dataset_path)
# val_indices = val_dataset.indices
# val_df = original_df.iloc[val_indices]
# os.makedirs(os.path.dirname(val_output_path), exist_ok=True)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size, False)

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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

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
model.load_state_dict(best_model_state)
model.eval()

# Plotting the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Training Loss')
plt.plot(validation_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image file
curve_path = output_path + "-loss_curve.png"
plt.savefig(curve_path)
plt.show()

# Close the plot to free up memory
plt.close()

# # Validation validation
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# correct = 0
# all_labels = []
# all_predictions = []
# with torch.no_grad():
#     for batch in val_loader:
#         inputs = batch['topics'].to(device)
#         labels = batch['label'].to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs, 1)
#         correct += (predicted == labels).sum().item()
#         all_labels.extend(labels.cpu().numpy()) 
#         all_predictions.extend(predicted.cpu().numpy()) 
# accuracy = correct / len(val_loader.dataset)
# print(f"Validation Accuracy: {accuracy:.4f}")
# print()
# print("Validation")
# print(classification_report(all_labels, all_predictions, target_names=dataset.genres))
# print(confusion_matrix(all_labels, all_predictions))

# Testing
test_dataset = LyricsDataset(csv_file=test_set_path)
test_loader = DataLoader(test_dataset, batch_size, False)
correct = 0
all_labels = []
all_predictions = []
with torch.no_grad():
    for batch in test_loader:
        inputs = batch['topics'].to(device)
        labels = batch['label'].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        all_labels.extend(labels.cpu().numpy()) 
        all_predictions.extend(predicted.cpu().numpy()) 
accuracy = correct / len(test_loader.dataset)
print(f"Testing Accuracy: {accuracy:.4f}")
# print()
# print("Testing")
# print(classification_report(all_labels, all_predictions, target_names=dataset.genres))
# print(confusion_matrix(all_labels, all_predictions))


# Saving the model
torch.save(best_model_state, output_path)
# val_df.to_csv(val_output_path, index=False)

report = classification_report(
    all_labels, 
    all_predictions, 
    target_names=dataset.genres, 
    output_dict=True
    )
df = pd.DataFrame(report).transpose()

report_path = output_path + "-report.csv"
df.to_csv(report_path, index=True)

# Assuming y_true and y_pred are your true labels and predictions
cm = confusion_matrix(all_labels, all_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.genres)
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(rotation=45)
plt.tight_layout()

cm_path = output_path + "-cm.png"
plt.savefig(cm_path)
plt.show()