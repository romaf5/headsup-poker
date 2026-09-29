import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

torch.set_num_threads(min(4, torch.get_num_threads()))  # tiny networks: many intra-op threads only contend
