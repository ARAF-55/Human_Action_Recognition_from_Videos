import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import cv2
import torch
import torch.nn as nn
import pickle  
import numpy as np  


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Feature Extraction Phase
video_path = './action_youtube_naudio/horse_riding/v_riding_02/v_riding_02_02.avi'
num_frames = 15
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model = model.to(device)
model_extractor = torch.nn.Sequential(*list(model.children())[:-1])
model_extractor = model_extractor.to(device)

model_extractor.eval()

cap = cv2.VideoCapture(video_path)
frame_count = 0
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frame = frame.to(device)
    frames.append(frame)
    if frame_count == num_frames:
        break

cap.release()

frames_tensor = torch.stack(frames, dim=0)
with torch.no_grad():
    feature_tensor = model_extractor(frames_tensor)
        
feature_tensor = torch.flatten(feature_tensor, start_dim=1)
feature_tensor = feature_tensor.unsqueeze(dim=0)

# GET THE LABEL MAPPING
file_path = 'label_mapping.pkl'
with open(file_path, 'rb') as file:
    label_mapping = pickle.load(file)   
label_mapping = {key: str(value) for key, value in label_mapping.items()}


# Prediction phase
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

input_size = 2048
hidden_size = 128
num_classes = 11
prediction_model = MultiLayerBiLSTMClassifier(input_size, hidden_size, 2, num_classes)
prediction_model.load_state_dict(torch.load('model.pth', weights_only=True))
prediction_model.eval()
output = prediction_model(feature_tensor)
_, pred = torch.max(output, dim=1)


prediction = pred.item()
print(label_mapping[prediction])




