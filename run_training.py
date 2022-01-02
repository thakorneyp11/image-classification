import torch
import pendulum
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim

from models.vanilla import VanillaClassification
from dataset.dataset_dataloader import create_dataset_dataloader


def set_device(display=False):
    """transfer data to `gpu` for faster processing time"""
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    if display:
        print("device : ", device)
    return torch.device(device)


def acc_score(y_preds, y_trues):
    length = len(y_preds)
    count_correct = 0
    for y_pred, y_true in zip(y_preds, y_trues):
        if y_pred == y_true:
            count_correct += 1
    return count_correct / length


def get_maximum_class(outputs, labels):
    _, pred = torch.max(outputs, 1)
    pred = pred.detach().cpu().clone().numpy().astype(np.int32).tolist()
    target = labels.detach().cpu().clone().numpy().astype(np.int32).tolist()
    return pred, target


def train_model(model, train_loader, valid_loader, criterion, optimizer, n_epochs, device_type="cpu"):
    if (device_type == "gpu"):
        device = set_device()

    train_iter = len(train_loader)
    test_iter = len(valid_loader)
    print("train_iter: ", train_iter)
    print("valid_iter: ", test_iter)

    for epoch in range(n_epochs):
        print(f"\nEpoch number : {str(epoch + 1)}/{n_epochs}")
        model.train()
        loss_total = 0.0
        count_iter = 0
        y_pred = []
        y_true = []

        for i, (images, labels, _) in enumerate(train_loader):
            if (device_type == "gpu"):
                images = images.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            with torch.no_grad():
                # get class with maximum probability
                pred, target = get_maximum_class(outputs, labels)
                y_pred.extend(pred)
                y_true.extend(target)

            loss_total += loss.item()
            count_iter += 1

            del images, labels, outputs, loss, pred, target
            torch.cuda.empty_cache()

        epoch_acc = acc_score(y_pred, y_true)
        average_loss = loss_total / count_iter

        print("- Training dataset : Got {} accuracy, Epoch loss {}".format(epoch_acc, average_loss))

        # evaluate valid_dataset every X epoch
        if (epoch + 1) % 5 == 0:
            _, _ = evaluate_model(model, valid_loader, device_type=device_type)

        del y_pred, y_true, epoch_acc, loss_total, average_loss
        torch.cuda.empty_cache()

    print("\nFinish training process")
    return model


def evaluate_model(model, valid_loader, device_type="cpu"):
    if (device_type == "gpu"):
        device = set_device()

    model.eval()
    loss_total = 0.0
    count_iter = 0
    y_pred = []
    y_true = []

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(valid_loader):
            if (device_type == "gpu"):
                images = images.to(device)
                labels = labels.to(device)

            outputs = model(images)
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)

            # get class with maximum probability
            pred, target = get_maximum_class(outputs, labels)
            y_pred.extend(pred)
            y_true.extend(target)

            loss_total += float(loss)
            count_iter += 1

            del images, labels, outputs, loss, pred, target
            torch.cuda.empty_cache()

        epoch_acc = acc_score(y_pred, y_true)
        average_loss = loss_total / count_iter

        print("- Testing dataset : Got {} accuracy, Epoch loss {}".format(epoch_acc, average_loss))

    return y_pred, y_true


if __name__ == '__main__':
    csv_path = 'pokemon_dataset.csv'
    metadata_df = pd.read_csv(csv_path)

    n_epochs = 200
    batch_size = 32
    image_size = 400
    device_type = "gpu"
    learning_rate = 0.001
    momentum = 0.9
    weight_decay = 0.003
    class_count = metadata_df['label'].nunique()

    # Create Dataset and DataLoader
    train_dataset, valid_dataset, train_loader, valid_loader = create_dataset_dataloader(metadata_df, image_size=image_size, test_size=0.2, batch_size=batch_size)

    # Declare Custom Model
    model = VanillaClassification(num_class=class_count)

    print("Model : ", model.model_name)
    if device_type == "gpu":
        device = set_device(display=True)
        model.to(device)

    # Declare Optimizer and Loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Model Training
    new_model = train_model(model, train_loader, valid_loader, criterion, optimizer, n_epochs=n_epochs, device_type="gpu")

    # Save Model
    new_model.to("cpu")
    dt_sting = pendulum.now(tz='Asia/Bangkok').to_atom_string()
    filename = f"models/CustomModel_{dt_sting}.pt"
    torch.save(new_model.state_dict(), filename)
