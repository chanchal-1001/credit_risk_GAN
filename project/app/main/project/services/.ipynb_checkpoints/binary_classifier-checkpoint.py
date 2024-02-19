# Define the neural network
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

folder_path = '../../resource/model/'
model_path  = folder_path + "credit_risk_classifier.pth"

# Define the neural network
class BinaryClassifier(nn.Module):
    def __init__(self, in_columns):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(in_columns, 256, bias=True)
        self.layer2 = nn.Sequential(nn.Linear(256, 1024, bias=True), nn.Dropout(0.1))
        self.layer3 = nn.Sequential(nn.Linear(1024, 512, bias=True), nn.Dropout(0.1))
        self.layer4 = nn.Sequential(nn.Linear(512, 256, bias=True), nn.Dropout(0.1))
        self.layer5 = nn.Linear(256, 1, bias=True)
        

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = torch.relu(self.layer4(x))
        x = torch.sigmoid(self.layer5(x))        
        return x

def trainingTheClassifier(model, criterion, optimizer, X_train, y_train): 
    print("Invoking the Binary Classifier Training")
    # Training loop
    epochs = 1000    
    for epoch in range(epochs):
        # Forward pass

        y_pred = model(torch.tensor(X_train.values, dtype=torch.float32))        
        # Modify the target tensor
        y_train_tn = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

        # Calculate the loss
        loss = criterion(y_pred, y_train_tn)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}') 
            
    print("Binary Classifier Training Completed")
    return model
                
                
def saveBinaryClassifierModel(model):
    #Save the state of the classification model    
    torch.save(model.state_dict(), model_path)    
    print("Binary Classifer Model Saved")
    
def loadBinaryClassifierModel():
    # Create an instance of the model
    credit_risk_model = BinaryClassifier(80)
    # Load the model's parameters
    credit_risk_model.load_state_dict(torch.load(model_path))
    print("Loaded Binary Classifer Model")
    print("Credit Risk Model Parameters:")
    credit_risk_model.eval()  # Set the model to evaluation model
    return credit_risk_model  
        

def createBinaryClassifierModel(columns):        
    # Create an instance of the model
    model = BinaryClassifier(len(columns))
    # Define the loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)   
    print("Generated Binary Classifier Model")
    return model, criterion, optimizer

def testBinaryClassifierModel(model, X_test, y_test):
    print("Testing Binary Classifer Model and printing classification report")
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test.values, dtype=torch.float32))        
        y_pred_binary = np.where(y_pred >= y_pred.mean(), 1, 0)

    print(classification_report(y_test, y_pred_binary))
    
    


