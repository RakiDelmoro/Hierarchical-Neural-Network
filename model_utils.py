import torch
import statistics
from torch import optim
from time import perf_counter
from torch.utils.data import DataLoader

def train(model: torch.nn.Module, optimizer: optim.AdamW, loss_function: torch.nn.CrossEntropyLoss, dataloader: DataLoader):
    model.train()
    print('Training...')
    
    def train_one_batch(model, optimizer, input_batch, expected_batch, loss_function):
        output_batch = model(input_batch)
        loss = loss_function(output_batch, expected_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    losses = []
    for input_batch, expected_batch in dataloader:
        loss = train_one_batch(model, optimizer, input_batch, expected_batch, loss_function)
        losses.append(loss.item())

    average_loss = statistics.fmean(losses)
    return average_loss

def runner(dataloader_for_training, dataloader_for_validate, model, optimizer, loss_function, number_of_epochs):
    for epoch in range(1, number_of_epochs+1):
        start = perf_counter()
        model.print_trainable_parameters()
        train_loss = train(model, optimizer, loss_function, dataloader_for_training)
        end = perf_counter()
        print(f"EPOCH-{epoch}: {train_loss}, time:{end-start}")
        