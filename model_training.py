import torch
from torch import nn
import numpy as np

def train_model(X_train, y_train, num_classes, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2, learning_rate=0.001):
    # Check if X_train is empty
    if len(X_train) == 0:
        raise ValueError("X_train is empty")
    
    # Define the model architecture
    input_size = X_train.shape[1]
    model = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, num_classes),
        nn.Softmax(dim=1),
    )
    if bidirectional:
        model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size*2, num_classes),
            nn.Softmax(dim=1),
        )

        # Move the model to the GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        num_epochs = 10
        batch_size = 128
        num_batches = int(np.ceil(len(X_train) / batch_size))
        if num_batches == 0:
            raise ValueError("Batch size is larger than the number of training samples")
     
        if num_batches > 0:
            for epoch in range(num_epochs):
                total_loss = 0.0
                for batch in range(num_batches):
                    start = batch * batch_size
                    end = (batch + 1) * batch_size
                    X_batch = X_train[start:end].to(device)
                    y_batch = y_train[start:end].to(device)

                    optimizer.zero_grad()
                    output = model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                print("Epoch %d, Loss=%.4f" % (epoch+1, total_loss/num_batches))

        return model
