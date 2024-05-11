import torch
from functools import partial

tensor = partial(torch.tensor, device="cuda")
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'
