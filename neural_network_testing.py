import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from lyric_dataset import LyricsDataset
from genre_classifier import GenreClassifier

def run_tests(dataset, loader, device):
    input_size = len(dataset[0]['topics'])  # Number of input features
    output_size = len(dataset.genres)  # Number of output classes
    model = GenreClassifier(input_size, hidden_sizes, output_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_labels = []
    all_predictions = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in loader:
            inputs = batch['topics'].to(device)  # Move inputs to GPU/CPU
            labels = batch['label'].to(device)  # Move labels to GPU/CPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())  # Move labels to CPU for evaluation
            all_predictions.extend(predicted.cpu().numpy())  # Move predictions to CPU for evaluation

    # Generate classification report
    report = classification_report(all_labels, all_predictions, target_names=dataset.genres)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return report, conf_matrix

# Set device to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 128

model_path = "classifiers/genre128_512-256-128_20241204180834.pth"
val_set_path = "classifiers/genre128_512-256-128_20241204180834_val.csv"
num_topics = 128
hidden_sizes = [512,256,128]  # Sizes of hidden layers

val_dataset = LyricsDataset(csv_file=val_set_path)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
val_report, val_cm = run_tests(val_dataset, val_loader, device)

test_set_path = f"dataset/topics_test/genly3k_test{num_topics}.csv"
test_dataset = LyricsDataset(csv_file=test_set_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
test_report, test_cm = run_tests(test_dataset, test_loader, device)

print("Validation Report")
print(val_report)
print()
print(val_cm)
print()
print("---")
print()
print("Test Report")
print(test_report)
print()
print(test_cm)
