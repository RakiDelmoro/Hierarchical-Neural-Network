import gzip
import torch
import pickle
from functools import partial
from model_utils import runner
from mlp import MultiLayerPerceptron
from functional_capsule_nn import capsule_neural_network
from torch.utils.data import TensorDataset, DataLoader
from features import tensor

def main():
    EPOCHS = 10000000
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.0001
    WIDTH = 28
    HEIGHT = 28

    # load training data into memory
    with (gzip.open('./training-data/mnist.pkl.gz', 'rb')) as file:
        ((training_input, training_expected), (validation_input, validation_expected), _) = pickle.load(file, encoding='latin-1')
	# convert numpy arrays to torch tensors
    training_input, training_expected, validation_input, validation_expected = map(tensor, (training_input, training_expected, validation_input, validation_expected))
    input_feature = HEIGHT * WIDTH

    training_dataset = TensorDataset(training_input, training_expected)
    training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset = TensorDataset(validation_input, validation_expected)
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    
    model, parameters = capsule_neural_network(feature_sizes=[2000, 10], input_feature=input_feature, capsule_tall=1, capsule_wide=1, rotation_amount=1)
    # model = MultiLayerPerceptron()
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE)
    loss_func = torch.nn.CrossEntropyLoss()

    runner(training_dataloader, validation_dataloader, model, optimizer, loss_func, EPOCHS, False)

main()
