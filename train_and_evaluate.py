import torch
import datetime
import time
import wandb
from tqdm import tqdm
from data import augment_data


def get_loss_and_accuracy(output, y):
    loss = torch.nn.BCEWithLogitsLoss()(output, y)

    probs = torch.sigmoid(output)
    predictions = (probs >= 0.5).float()
    accuracy = (predictions==y).float().mean()
    return loss, accuracy


def evaluate_loop(loader,model):
    loss_vals = []
    accuracy_vals = []
    for x, y in loader:
        output = model(x).squeeze(dim=-1)
        loss, accuracy = get_loss_and_accuracy(output, y)
        loss_vals.append(loss)
        accuracy_vals.append(accuracy)

    avg_loss = torch.stack(loss_vals, dim=0).mean(dim=0)
    avg_accuracy = torch.stack(accuracy_vals, dim=0).mean(dim=0)
    return avg_loss, avg_accuracy


def train_loop(x, y, model, model_type, optimizer):
        if model_type == 'NN_Augmentation':
            x, y = augment_data(x, y)
        output = model(x).squeeze(dim=-1)
        loss, accuracy = get_loss_and_accuracy(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, accuracy


def train_and_evaluate(train_loader, test_loader, model, device, lr, epochs, model_type, log_wandb):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    print('Model:', model)
    model.to(device)
    model.train()
    timeout = 60 * 30
    start_time = time.time()

    for epoch in tqdm(range(epochs + 1)):
        train_loss_vals = []
        train_accuracy_vals = []
        for x, y in train_loader:
            loss, accuracy = train_loop(x,y, model,model_type, optimizer)
            train_loss_vals.append(loss)
            train_accuracy_vals.append(accuracy)

        with torch.no_grad():
            train_loss_avg = torch.stack(train_loss_vals, dim=0).mean(dim=0)
            train_accuracy_avg = torch.stack(train_accuracy_vals, dim=0).mean(dim=0)
            test_loss_avg, test_accuracy_avg = evaluate_loop(test_loader, model)
        current_time = time.time() - start_time
        if log_wandb:
            wandb.log({"epoch": epoch, "train loss": train_loss_avg, 'train accuracy': train_accuracy_avg,
                       "test loss": test_loss_avg, 'test accuracy': test_accuracy_avg, "time": current_time})
        print(f"Epoch: {epoch}, Train Loss: {train_loss_avg:.4f}, Train accuracy: {train_accuracy_avg:.4f}, "
              f"Test Loss: {test_loss_avg:.4f}, Test accuracy: {test_accuracy_avg:.4f}, time: {current_time:4f}")

        if torch.isnan(train_loss_avg):
            raise ValueError('Optimizer diverged')

        if current_time > timeout:
            print("Training timed out after 30 minutes")
            break

    print(datetime.datetime.now(), 'ENDED TRAINING')
    return model

