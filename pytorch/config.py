import torch

"""
device initialization
"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_num = 2

"""
data handling parameters
"""
complete_threshold = 0.05
complete_rate = 0.56
core_threshold = 0.05
core_rate = 0.66
enhancing_threshold = 0.02
enhancing_rate = 0.7
