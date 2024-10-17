import torch
import torch.nn as nn  
import torch.optim as optim 
import matplotlib.pyplot as plt 
import numpy as np  
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_classes):
        
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1,:])
        out = nn.functional.softmax(out, dim=1)
        
        return out
    
    

class MultiLayerBiLSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
       
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = nn.functional.softmax(out, dim=1)
        
        return out
    

train_features = torch.from_numpy(np.load('train_features.npy')).to(torch.float)
train_labels = torch.from_numpy(np.load('train_labels.npy'))

idx = np.random.permutation(len(train_features))
train_features, train_labels = train_features[idx], train_labels[idx]

test_features = torch.from_numpy(np.load('test_features.npy')).to(torch.float)
test_labels = torch.from_numpy(np.load('test_labels.npy'))


train_dataset = TensorDataset(train_features, train_labels)
test_dataset = TensorDataset(test_features, test_labels)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=True)


input_size = train_features.shape[-1]
hidden_size = 128
num_classes = len(np.unique(train_labels))
num_frames = 15


model = MultiLayerBiLSTMClassifier(input_size, hidden_size, 2, num_classes)
model = model.to(device)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)


num_epochs = 50
batch_size = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []


def train(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()  
        
        outputs = model(features)  
        loss = loss_function(outputs, labels)
        loss.backward()  
        optimizer.step() 
        
        running_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_size
        correct += (predicted == labels).sum().item() 
    
    epoch_loss = running_loss / total  
    epoch_accuracy = correct*100 / total 
    
    return epoch_loss, epoch_accuracy



def evaluate(model, test_loader, loss_function, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)  
            loss = loss_function(outputs, labels)  
            
            running_loss += loss.item() * features.size(0)  
            _, predicted = torch.max(outputs.data, 1)  
            total += batch_size
            correct += (predicted == labels).sum().item()  
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct*100 / total  
    
    return epoch_loss, epoch_accuracy


total_epochs = 0
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, train_loader, loss_function, optimizer, device)
    test_loss, test_accuracy = evaluate(model, test_loader, loss_function, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    total_epochs += 1
    print(f"Epoch No - {total_epochs}")
    print('Train Loss: {:.4f}, Train Accuracy: {:.2f}%  Test Loss: {:.4f}, Test Accuracy: {:.2f}%'.format( train_loss, train_accuracy, test_loss, test_accuracy))
    print("\n")


torch.save(model.state_dict(), 'model.pth')

    
    

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()