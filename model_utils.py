import os
import torch
import statistics
import numpy as np
from torch import optim
from time import perf_counter
from torch.utils.data import DataLoader
from capsule_nn import CapsuleNeuralNetwork
from features import GREEN, RED, RESET, WHITE_UNDERLINE

def train(model: torch.nn.Module, optimizer: optim.AdamW, loss_function: torch.nn.CrossEntropyLoss, dataloader: DataLoader):
    model.train()    
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

def validate(model: torch.nn.Module, loss_func, dataloader: DataLoader):
    list_of_digit_probabilities = []
    list_of_correct_prediction = []
    list_of_wrong_prediction = []
    losses = 0.0
    for input_batch, expected_batch in dataloader:
        output = torch.nn.functional.softmax(model(input_batch), dim=-1)
        probability_of_correct_digit = output[:, expected_batch].item()
        loss = loss_func(output, expected_batch)
        list_of_digit_probabilities.append(probability_of_correct_digit)
        losses += loss.item()
        predicted = output.argmax().item()
        if expected_batch.item() == predicted:
            predicted_and_expected = {'predicted': predicted, 'expected': expected_batch.item()}
            list_of_correct_prediction.append(predicted_and_expected)
        else:
            predicted_and_expected = {'predicted': predicted, 'expected': expected_batch.item()}
            list_of_wrong_prediction.append(predicted_and_expected)
        
    total_loss = losses / len(dataloader)
    return total_loss, list_of_correct_prediction, list_of_wrong_prediction, list_of_digit_probabilities

def print_correct_prediction(correct_prediction_list, number_to_print):
    print(f"{GREEN}Correct prediction!{RESET}")
    for i in range(number_to_print):
        each_item = correct_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_wrong_prediction(wrong_prediction_list, number_to_print):
    print(f"{RED}Wrong prediction!{RESET}")
    for i in range(number_to_print):
        each_item = wrong_prediction_list[i]
        predicted, expected = each_item['predicted'], each_item['expected']
        print(f"Predicted: {predicted} Expected: {expected}")

def print_percentile_of_correct_probabilities(probabilities_list):
    tenth_percentile = np.percentile(probabilities_list, 10)
    ninetieth_percentile = np.percentile(probabilities_list, 90)
    average = statistics.fmean(probabilities_list)

    print(f"Average: {average} Tenth percentile: {tenth_percentile} Ninetieth percentile: {ninetieth_percentile}")

def save_model_checkpoint(epoch, optimizer, training_loss, model, file_name):
    checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "training_loss": training_loss}

    return torch.save(checkpoint, f"{file_name}")

def load_model_checkpoint(model_checkpoint_file, model, optimizer):
    model_checkpoint_loaded = torch.load(model_checkpoint_file)
    model.load_state_dict(model_checkpoint_loaded['model_state'])
    optimizer.load_state_dict(model_checkpoint_loaded['optimizer_state'])
    loaded_epoch = model_checkpoint_loaded["epoch"]
    training_loss = model_checkpoint_loaded["training_loss"]

    return loaded_epoch, training_loss

def runner(dataloader_for_training, dataloader_for_validate, model, optimizer, loss_function, number_of_epochs, use_checkpoint):
    start_epoch = 1
    # Check if model checkpoint is available
    if use_checkpoint:
        if os.path.exists("checkpoint.tar"):
            loaded_epoch, loss = load_model_checkpoint("checkpoint.tar", model, optimizer)
            start_epoch = loaded_epoch + 1
            print(f"Loaded checkpoint! Epoch: {loaded_epoch} loss: {loss}")
        else:
            print('No checkpoint file available.')

    for epoch in range(start_epoch, number_of_epochs+1):
        start = perf_counter()
        train_loss = train(model, optimizer, loss_function, dataloader_for_training)
        validation_loss, correct_prediction_list, wrong_prediction_list, digit_probabilities = validate(model, loss_function, dataloader_for_validate)
        end = perf_counter()
        print_correct_prediction(correct_prediction_list, 5)
        print_wrong_prediction(wrong_prediction_list, 5)
        print(f"{WHITE_UNDERLINE}EPOCH: {epoch} Training loss: {train_loss} Validation loss: {validation_loss} time: {end-start}{RESET}")
        print_percentile_of_correct_probabilities(digit_probabilities)
        # save_model_checkpoint(epoch, optimizer, train_loss, model, "checkpoint.tar")
        # print("Saved model checkpoint!")
