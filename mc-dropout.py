# Skeleton code for how to do MC dropout

# Generate predictions with MC dropout
num_samples = 100
with torch.no_grad():
    # Replace `model` to actual trained model with dropout layers
    model.train()  # Set the model to training mode, so that dropout layers are enabled
    mc_predictions = torch.zeros(num_samples, len(X_tensor)) # Replace `X_tensor` to actual input data
    for i in range(num_samples):
        mc_predictions[i] = model(X_tensor).squeeze()

# Calculate standard deviation of predictions
std_predictions = mc_predictions.std(dim=0) # shape: [len(X_tensor)], i.e. the number of data points/labels
print(std_predictions) # this is the estimation of uncertainty for each data point
