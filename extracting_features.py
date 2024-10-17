import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import os
import numpy as np  
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle

data_directory = './action_youtube_naudio/'

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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

model_extractor = torch.nn.Sequential(*list(model.children())[:-1])
model_extractor = model_extractor.to(device)
samples = []

model_extractor.eval()


for label in os.listdir(data_directory):
    label_dir = os.path.join(data_directory, label)
    print(label_dir)
    
    for sub_dir in os.listdir(label_dir):
        if sub_dir == 'Annotation':
            continue
        video_dir = os.path.join(label_dir, sub_dir)
        
        for video_file in os.listdir(video_dir):
            video_path = os.path.join(video_dir, video_file)
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
            
            if len(frames) == num_frames:
                
                frames_tensor = torch.stack(frames, dim=0)
                
                with torch.no_grad():
                    feature_tensor = model_extractor(frames_tensor)
                    
                features_tensor = torch.flatten(feature_tensor, start_dim=1)
                
                features = features_tensor.to('cpu')
                features = features.numpy()
                
                samples.append((features, label))
                
                
np.random.shuffle(samples)


split_index = int(0.8*(len(samples)))
train_samples, test_samples = samples[:split_index], samples[split_index:]

print(train_samples)
print(test_samples)

train_features, train_labels = zip(*train_samples)
test_features, test_labels = zip(*test_samples)

le = LabelEncoder()
train_numerical_labels = le.fit_transform(train_labels)
test_numerical_labels = le.transform(test_labels)
label_mapping = {index: label for index, label in enumerate(le.classes_)}
with open('label_mapping.pkl', 'wb') as file:
    pickle.dump(label_mapping, file)

train_features = np.array(train_features)
train_labels = train_numerical_labels

test_features = np.array(test_features)
test_labels = test_numerical_labels


print(f"train_features shape: {train_features.shape}")
print(f"train_labels shape: {train_labels.shape}")
print(f"test_features shape: {test_features.shape}")
print(f"test_lebels shape: {test_labels.shape}")


np.save('train_features.npy', train_features)
np.save('train_labels.npy', train_labels)
np.save('test_features.npy', test_features)
np.save('test_labels.npy', test_labels)


                
