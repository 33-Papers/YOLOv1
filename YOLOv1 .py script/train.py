import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

from model import Yolov1
from loss import YoloLoss
from dataset_and_dataloader import train_loader, test_loader
from bboxes_and_plotting import get_bboxes
from algorithms import mean_average_precision

# Hyperparameters
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32


"""# Training setup and training"""
# Instantiate the model
model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)

# Compile the model
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define the loss function
loss_fn = YoloLoss()

# Save train history
history_tl = []  # For epoch vs. loss
history_ta = [] # For epoch vs. mAP

num_epochs = 2

for epoch in tqdm(range(num_epochs), desc='Epochs'):
    model.train()  # Set the model to training mode

    start_time = time.time() # Start time of the epoch

    mean_loss = []

    # Iterate over the training data in batches
    for inputs, labels in train_loader:
        # Move the inputs and labels to the selected device
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(inputs)

        torch.cuda.empty_cache() # Limit GPU memory growth

        # Calculate the loss
        loss = loss_fn(outputs, labels)
        mean_loss.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache() # Limit GPU memory growth

    end_time = time.time()  # End time of the epoch
    epoch_duration = end_time - start_time  # Duration of the epoch

    # Use the trained model to predict label
    pred_boxes, target_boxes = get_bboxes(
        train_loader, model, iou_threshold=0.5, threshold=0.4
    )

    # Calculate the mean average precision after every epoch/training
    mean_avg_prec = mean_average_precision(
        pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
    )

    history_tl.append(sum(mean_loss)/len(mean_loss))
    history_ta.append(mean_avg_prec)

    if (epoch+1) % 10 == 0:
        # Print the epoch duration
        tqdm.write(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds")

        # Print the loss and accuracy for training and validation data
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {sum(mean_loss)/len(mean_loss):.4f}, mAP: {mean_avg_prec:.4f}")

# Save the model
torch.save(model.state_dict(), "YOLOV1_model.pth")


"""# Plotting"""
epochs = range(1, len(history_tl)+1)
# Plot losses
plt.plot(epochs, history_tl)
plt.xlabel("Epochs")
plt.ylabel("Losses")
plt.title("Training Loss")
plt.show()

epochs = range(1, len(history_ta)+1)
# Plot losses
plt.plot(epochs, history_ta)
plt.xlabel("Epochs")
plt.ylabel("mAP")
plt.title("Mean Absolute Precision")
plt.show()


"""# Testing and Evaluation"""
# Use the trained model to predict label
pred_boxes, target_boxes = get_bboxes(
    test_loader, model, iou_threshold=0.5, threshold=0.4
)

# Calculate the mean average precision after every epoch/training
mean_avg_prec = mean_average_precision(
    pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
)

# Print the mean average precision on test dataset
print(f"Mean Average Precision: {mean_avg_prec:.4f}")