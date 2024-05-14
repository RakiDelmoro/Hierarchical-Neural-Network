import gzip
import torch
import pickle
from functools import partial
from model_utils import runner
from capsule_nn import CapsuleNeuralNetwork
from capsule_nn_v2 import CapsuleNeuralNetworkV2
from torch.utils.data import TensorDataset, DataLoader
from features import tensor

def main():
    EPOCHS = 10000000
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.01
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

    model = CapsuleNeuralNetwork(num_capsule_wide=1, num_capsule_tall=1, feature_sizes=[32, 32, 32, 32, 32, 32], input_data_size=input_feature, roll_each_layer=1)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
    loss_func = torch.nn.CrossEntropyLoss()

    runner(training_dataloader, validation_dataloader, model, optimizer, loss_func, EPOCHS, True)

main()
