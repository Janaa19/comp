#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()


# Train your dx classification model.
def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()
    dxs = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image, but only if the image has one or more dx classes.
        if check_dx(record)==0:
            print(record,' No header')
            continue
        dx = load_dx(record)
        if dx:
            current_features = load_image(record)
            features.append(current_features)
            dxs.append(dx)

    if not dxs:
        raise Exception('There are no labels for the data.')

    features = np.vstack(features)
    classes = sorted(set.union(*map(set, dxs)))
    dxs = compute_one_hot_encoding(dxs, classes)


    training_data = preprocess_images(features)

    # Convert the image data to the format (batch_size, channels, height, width)
    X_train = np.transpose(training_data, (0, 3, 1, 2)).astype(np.float32)
    y_train = dxs.astype(np.int64)

    # Convert the NumPy arrays to PyTorch tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train).to(torch.float32)
    

    # Create a TensorDataset
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Create a DataLoader
    batch_size = 32  # You can adjust the batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print('Device: ',device)
    # Initialize the model, loss function, and optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Assuming you have a DataLoader for your training data: train_loader

    # Training loop
    num_epochs = 10 # Set the number of epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_dx_model(model_folder, model, classes)

    if verbose:
        print('Done.')
        print()


# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.
def load_dx_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'base_cnn.pth')
    return joblib.load(filename)

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.
def run_dx_model(dx_model, record, signal, verbose):
    model = dx_model['model']
    classes = dx_model['classes']

    # Extract features.
    features = load_image(record)
    features = np.asarray(features)
    test_data = np.transpose(features, (0, 3, 1, 2)).astype(np.float32)
    images_tensor = torch.tensor(test_data, dtype=torch.float32)
    # Get model probabilities.
    with torch.no_grad():
        # If your model and data are on different devices (e.g., model on GPU), move the data to the same device
        if torch.cuda.is_available():
            images_tensor = images_tensor.to('cuda')
            model.to('cuda')
        
        # Perform prediction
        predictions = model(images_tensor)
        
        # Convert predictions to probabilities using softmax if your model does not include a softmax layer
        probabilities = torch.softmax(predictions, dim=1)
        
        # If you need to move the predictions back to CPU and convert to numpy
        probabilities_np = probabilities.cpu().numpy()
        max_probability = np.argmax(probabilities_np,axis=1)
        labels = [list(classes)[i] for i in max_probability]
    return labels

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)

# Save your trained dx classification model.
def save_dx_model(model_folder, model, classes):
    d = {'model': model, 'classes': classes}
    filename = os.path.join(model_folder, 'base_cnn.pth')
    torch.save(model, filename)



def preprocess_images(images):
    processed_images = np.zeros((images.shape[0], 224, 224, 3))
    for i, img in enumerate(images):
        # Resize image (using skimage or similar library)
        resized_img = np.resize(img, (224, 224))
        # Convert to 3 channels if necessary by dropping or averaging the fourth channel
        processed_images[i] = resized_img[..., :3]
    return processed_images

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def check_dx(record):

    header_file = get_header_file(record)
    header = load_text(header_file)
    dxs, has_dx = get_variables(header, '#Dx:')
    if not has_dx:
        return 0
    