
from sklearn.metrics import classification_report
import torch
import numpy as np

def evaluate_model(model, X_test, y_test):
    # Move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    model.eval()
    y_pred = []
    batch_size = 128
    num_batches = int(np.ceil(len(X_test) / batch_size))
    with torch.no_grad():
        for batch in range(num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            X_batch = X_test[start:end].to(device)
            output = model(X_batch)
            y_pred.extend(torch.argmax(output, dim=1).cpu().numpy().tolist())

    # Print classification report
    print(classification_report(y_test, y_pred))
