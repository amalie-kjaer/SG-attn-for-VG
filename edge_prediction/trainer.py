import sys
import torch
import wandb
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from focal_loss.focal_loss import FocalLoss
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from loss import FocalLoss
from dataset import SG
from model import MLP, MLP2
sys.path.insert(0, '..')
from VQA_models.utils import load_config, map_int_to_label, map_label_to_int

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(config_path):
    config = load_config(config_path)
    #-----------------
    # Initialize WandB logging
    #-----------------
    if eval(config['wandb']['log']) == True:
        wandb.init(project=config['wandb']['project_name'], name=config['wandb']['run_name'])
        wandb.config.update(config)

    #-----------------
    # Load data
    #-----------------
    train_dataset = SG(split='train')
    test_dataset = SG(split='test')
    
    # TODO: calculate class weights here instead of hardcoding
    class_weights = [14.757734858428652, 13.22517223995967, 1.4133693296778287, 14.768812159879902, 13.298918553565393, 322.1132332878581, 323.880658436214, 0, 0]
    sample_weights = [0.0] * len(train_dataset)
    for idx, (X, y) in enumerate(train_dataset):
        class_weight = class_weights[y]
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(config['training']['batch_size']),
        shuffle=False,
        # sampler=sampler, # comment to remove weighted sampling (oversampling)
        drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=int(config['testing']['batch_size']))

    #-----------------
    # Initialize model, loss function, and optimizer
    #-----------------
    model_name = config['model']['model_name']
    model = globals()[model_name]().to(device)
    
    optimizer_name = config['training']['optimizer']
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=float(config['training']['lr']))
    
    loss_fn_name = config['training']['loss_fn']
    # Weighted CE Loss:
    # loss_fn = getattr(nn, loss_fn_name)(weight=torch.tensor([
    #     14.757734858428652, 13.22517223995967, 1.4133693296778287, 14.768812159879902, 13.298918553565393, 322.1132332878581, 323.880658436214, 0, 0
    #     ]).to(device))

    # Regular CE Loss:
    loss_fn = getattr(nn, loss_fn_name)()
   
    num_epochs = int(config['training']['epochs'])

    #-----------------
    # Train model
    #-----------------
    for e in range(num_epochs):
        print(f"Epoch {e+1}\n-------------------------------")
        loss = train_epoch(train_dataloader, model, loss_fn, optimizer)
        # Training accuracy
        print("Train error:")
        train_acc, train_loss = test_epoch(train_dataloader, model, loss_fn)
        # Test accuracy
        print("Test error:")
        test_acc, test_loss = test_epoch(test_dataloader, model, loss_fn)
        
        if e % 10 == 0 and eval(config['wandb']['log']):
            print('Logging results to WandB')
            wandb.log({"Criterion": loss})
            wandb.log({"Train accuracy": train_acc, "Train loss": train_loss, "Test accuracy": test_acc, "Test loss": test_loss})
    
    #-----------------
    # Save model to "./checkpoints/"
    #-----------------
        if e % 500 == 0 and eval(config['paths']['save_model']):
            torch.save(model.state_dict(), f'./checkpoints/ckpt_{e}.pth')
            print(f"Model saved to ./checkpoints/ckpt_{e}.pth")
    
    #-----------------
    # Plot confusion matrix and save predictions for test set
    #-----------------
    if eval(config['visualization']['plot_confusion_matrix']):
        test_acc, test_loss = test_epoch(test_dataloader, model, loss_fn, conf=True)

def train_epoch(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(torch.float32)
        y = y.to(torch.long)
        X, y = X.to(device), y.to(device)
       
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss

def test_epoch(dataloader, model, loss_fn, conf=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    results, true_labels = [], []
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(torch.float32)
            y = y.to(torch.long)
            X, y = X.to(device), y.to(device)

            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            correct += (logits.argmax(1) == y).type(torch.float).sum().item()

            if conf:
                predictions = torch.argmax(logits, dim=1)
                results.extend(predictions.cpu().numpy())
                true_labels.extend(y.cpu().numpy())
    
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    if conf:
        get_confusion_matrix('confusion_matrix_TEST.png', results, true_labels)
        write_predictions_to_file('predictions_TEST.txt', results)
    
    return accuracy, test_loss

def test_model(config_path):
    config = load_config(config_path)
    #-----------------
    # Load inference data
    #-----------------
    test_dataset = SG(split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=256)

    #-----------------
    # Load model
    #-----------------
    model_name = config['model']['model_name']
    model = globals()[model_name]().to(device)
    model.load_state_dict(torch.load('./checkpoints/ckpt_320.pth'))
    model.eval()  # Set model to evaluation mode
    
    #-----------------
    # Perform inference
    #-----------------
    results = []
    true_labels = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X = X.to(torch.float32)
            X = X.to(device)

            # Compute prediction
            logits = model(X)
            predictions = torch.argmax(logits, dim=1)

            # Store the results
            results.extend(predictions.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
    
    
    # Save predictions to a text file
    output_predictions = 'TEST_predictions.txt'
    write_predictions_to_file(output_predictions, results)
    
    # Generate and save the confusion matrix
    output_conf_matrix_path = 'confusion_matrix.png'
    get_confusion_matrix(output_conf_matrix_path, results, true_labels)
    

def get_confusion_matrix(output_path, pred, true_labels):
    cm = confusion_matrix(pred, true_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=map_int_to_label.values(), yticklabels=map_int_to_label.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to {output_path}")

def write_predictions_to_file(output_path, pred):
    with open(output_path, 'w') as f:
        for p in pred:
            f.write(f"{p}\n")
    print(f"Predictions saved to {output_path}")

def print_config_summary(config):
    print('Config summary:')
    print('...')
    print('Press enter to confirm')
