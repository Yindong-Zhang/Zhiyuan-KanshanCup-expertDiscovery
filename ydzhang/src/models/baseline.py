from torch import nn
import torch
import torch.functional as F
from src.featureExtractLayer import EmbeddingMLPLayer
from src.layers import BiLSTMRCNN
from src.DIN import DIN
import numpy as np
dataDir = './test'
word_embedding = np.random.randn(1024, 128)

# baseline using pooling layer to extract semantic info rather than RCNN
wv_size = 128
