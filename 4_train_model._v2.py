import json
import numpy as np
import torch
import app.mil as mil
import itertools
from torch.optim import SGD, Adam, RMSprop
from app.mil import BagModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# load Data
with open('filtered_snps.json', 'r') as f:
    snps = json.load(f)

# Extract labels for stratification
labels = [snp['functionalClass'] if snp['functionalClass'] is not None else 'None' for snp in snps]

# Check the distribution of labels
from collections import Counter
counts = Counter(labels)

# Change the name of labels with count of 1
rare_functions = {None, 'coding_sequence_variant', 'TFBS_ablation', 'stop_retained_variant'}

for snp in snps:
    if snp['functionalClass'] in rare_functions:
        snp['functionalClass'] = 'rare_function'

# Re-assign labels after change
labels = [snp['functionalClass'] for snp in snps]

labels2 = [gene['location']['chromosomeName'] for snp in snps for gene in snp['genomicContexts']]
counts = Counter(labels2)

# Split the data into training and testing sets
train_snps, test_snps = train_test_split(snps, test_size=0.25, stratify = labels)

# Prepare chromosomeNames and functionalClass labels from snps
def extract_data(snps):
    chromosomeNames = []
    functionalClass = []

    for snp in snps:
        for gene in snp["genomicContexts"]:
            chromosomeNames.append(gene['location']["chromosomeName"])
        functionalClass.append(snp['functionalClass'] if snp['functionalClass'] is not None else 'None')

    return chromosomeNames, functionalClass

# Prepare training data
chromosomeNames, functionalClass = extract_data(snps)

# Fit LabelEncoder
le = LabelEncoder()
le.fit(chromosomeNames)

le2 = LabelEncoder()
le2.fit(functionalClass)

def prepare_data(snps, le, le2):
    instances = []
    ids = []
    labels = []

    for i, snp in enumerate(snps):
        # Convert features to 2D tensor
        associated_features = torch.tensor([
            [
                int(gene["isIntergenic"]), int(gene["isUpstream"]),
                int(gene["isDownstream"]), int(gene["distance"]),
                le.transform([gene['location']["chromosomeName"]])[0],
                int(gene['location']["chromosomePosition"]),
            ]
            for gene in snp["genomicContexts"]
        ])
        instances.append(associated_features)
        # Each instance in this SNP has the same ID
        ids.append(torch.full((associated_features.shape[0],), i))
        # Convert FunctionalClass to binary label
        labels.append(le2.transform([snp["functionalClass"]])[0])

    # Concatenate all instances and ids
    instances = torch.cat(instances, dim=0)
    ids = torch.cat(ids, dim=0)
    labels = torch.tensor(labels)

    return mil.MilDataset(instances, ids, labels), len(np.unique(functionalClass))

# Prepare training and testing data
train_dataset, n_classes = prepare_data(train_snps, le, le2)
test_dataset, _ = prepare_data(test_snps, le, le2)

# Save LabelEncoders
torch.save(le, 'label_encoder_chromosomeNames.pth')
torch.save(le2, 'label_encoder_functionalClass.pth')

# Define custom model

# Define custom prepNN
prepNN = torch.nn.Sequential(
    torch.nn.Linear(6, 64),  # Input layer
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32), # 1 hidden layer
    torch.nn.ReLU(),
)

# Define custom afterNN
afterNN = torch.nn.Sequential(
    torch.nn.Linear(32, n_classes),  # Output layer
    torch.nn.LogSoftmax(dim=1) 
)

# Define model with prepNN, afterNN and torch.mean as aggregation function
model = BagModel(prepNN, afterNN, torch.mean)

# Define DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, collate_fn=mil.collate)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=mil.collate)


# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
criterion = torch.nn.CrossEntropyLoss()
n_epochs = 5


for epoch in range(n_epochs):
    model.train()  # Set the model to training mode
    train_losses = []
    for batch in train_loader:
        # Get data
        instances, ids, labels = batch

        # Forward pass
        outputs = model((instances, ids))

        # Compute loss
        loss = criterion(outputs, labels.long())
        train_losses.append(loss.item())

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Calculate average training loss for this epoch
    avg_train_loss = sum(train_losses) / len(train_losses)

    # Now we evaluate the model on the test set
    model.eval()  # Set the model to evaluation mode
    test_losses = []
    all_outputs = []
    all_labels = []
    with torch.no_grad():  # Disable gradient computation
        for batch in test_loader:
            # Get data
            instances, ids, labels = batch

            # Forward pass
            outputs = model((instances, ids))
            all_outputs.extend(outputs.argmax(dim=1).tolist())
            all_labels.extend(labels.tolist())

            # Compute loss
            loss = criterion(outputs, labels.long())
            test_losses.append(loss.item())

    # Calculate average testing loss for this epoch
    avg_test_loss = sum(test_losses) / len(test_losses)

    # Compute accuracy, precision, recall and F1-score
    accuracy = accuracy_score(all_labels, all_outputs)

    # Print losses and metrics for this epoch
    print(f'Epoch {epoch+1}/{n_epochs}, Training Loss: {avg_train_loss}, Testing Loss: {avg_test_loss}, Accuracy: {accuracy}')

torch.save(model.state_dict(), 'model.pth')

# Grid search #2
import itertools
from torch.optim import SGD, Adam, RMSprop

# Define hyperparameters for grid
param_grid = {
    'lr': [0.00001, 0.000001],
    'n_epochs': [3, 5, 7, 10],
    'optimizer': [Adam]
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(*param_grid.values()))

# Store the best model
best_model = None
best_loss = float('inf')

# Iterate over all combinations
for params in param_combinations:
    lr, n_epochs, Optimizer = params
    print(f"Training with lr={lr}, n_epochs={n_epochs}, optimizer={Optimizer.__name__}")

    # Define model and optimizer with the current hyperparameters
    model = BagModel(prepNN, afterNN, torch.mean)
    optimizer = Optimizer(model.parameters(), lr=lr)

    # Train and evaluate the model
    for epoch in range(n_epochs):
        # The training part remains the same...
        model.train()  # Set the model to training mode
        train_losses = []
        for batch in train_loader:
            # Get data
            instances, ids, labels = batch

            # Forward pass
            outputs = model((instances, ids))

            # Compute loss
            loss = criterion(outputs, labels.long())
            train_losses.append(loss.item())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Now we evaluate the model on the test set
        model.eval()
        test_losses = []
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                instances, ids, labels = batch
                outputs = model((instances, ids))
                all_outputs.extend(outputs.argmax(dim=1).tolist())
                all_labels.extend(labels.tolist())
                loss = criterion(outputs, labels.long())
                test_losses.append(loss.item())

        # Calculate average testing loss for this epoch
        avg_test_loss = sum(test_losses) / len(test_losses)

        # Compute accuracy, precision, recall and F1-score
        accuracy = accuracy_score(all_labels, all_outputs)

        print(f'Epoch {epoch+1}/{n_epochs}, Testing Loss: {avg_test_loss}, Accuracy: {accuracy}')

        # Check if this model has the lowest loss so far
        if avg_test_loss < best_loss:
            best_model = model
            best_loss = avg_test_loss

print(f"Best loss was {best_loss}")

# Save the best model
torch.save(best_model.state_dict(), 'best_model.pth')

# Load the best model
best_model = BagModel(prepNN, afterNN, torch.mean)  # Initialize the model architecture
best_model.load_state_dict(torch.load('best_model.pth'))  # Load the saved parameters

