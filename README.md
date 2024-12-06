# LDA3k

Attempt to solve the problem using LDA with our 3,000 sample dataset

# Experiment Design

Factors: Levels
LDA Size: Partial, Full
Topic Count: 128, 2048
hidden_layers: 2x128, 4x256

Constants:
learning_rate: 0.001
epochs: best of 20
batch_size: 128

# Test Results

| Run | LDA     | Topics | Network |     | accuracy |
| --- | ------- | ------ | ------- | --- | -------- |
| 1   | Partial | 128    | 2x128   |     | 12.62%   |
| 2   | Partial | 2048   | 4x256   |     | 18.19%   |
| 3   | Full    | 128    | 4x256   |     | 40.71%   |
| 4   | Full    | 2048   | 2x128   |     | 47.83%   |

# LDA Training Size

| LDA     | Accuracy |
| ------- | -------- |
| Partial | 15.4%    |
| Full    | 44.3%    |

# Number of Topics

| Topics | Accuracy |
| ------ | -------- |
| 128    | 26.7%    |
| 2048   | 33.0%    |

# Size of Neural Network

| Network | Accuracy |
| ------- | -------- |
| 2x128   | 30.2%    |
| 4x256   | 29.5%    |
