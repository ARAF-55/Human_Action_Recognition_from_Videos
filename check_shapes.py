import torch
import numpy as np

train_features = torch.from_numpy(np.load('train_features.npy')).float()
train_labels = torch.from_numpy(np.load('train_labels.npy'))#.long()
idx = np.random.permutation(len(train_features))
train_features, train_labels = train_features[idx], train_labels[idx]

test_features = torch.from_numpy(np.load('test_features.npy')).float()
test_labels = torch.from_numpy(np.load('test_labels.npy'))#.long()

num_classes = len(np.unique(train_labels))


print(f"Train_features: {train_features.shape}")
print(f"Train_labels: {train_labels.shape}")

print(f"No. of classes: {num_classes}")