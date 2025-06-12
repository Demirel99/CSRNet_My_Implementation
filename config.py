# config.py

import torch

# --- Model & Training Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-6  # As mentioned in section 3.2.3 of the paper
BATCH_SIZE = 1        # The paper doesn't specify batch size, 1 is common for this task
NUM_EPOCHS = 200      # A reasonable number of epochs to train for
LOAD_MODEL = False    # Set to True to load a pre-trained model
SAVE_MODEL = True     # Set to True to save the model during training
MODEL_CHECKPOINT = "csrnet_shanghaitech_a.pth.tar" # Filename for saving/loading

# --- Dataset Paths ---
# Modify this to the path where your ShanghaiTech dataset is located
DATASET_BASE_PATH = r"C:\Users\Mehmet_Postdoc\Desktop\ShanghaiTech_Crowd_Counting_Dataset"

# --- Ground Truth Generation Parameters ---
# These values are taken directly from the paper's description for
# generating the geometry-adaptive density maps.

# For ShanghaiTech Part_A (Geometry-adaptive kernels)
# Section 3.2.1: "we follow the configuration in [18] where β = 0.3 and k = 3"
# [18] is the MCNN paper.
SIGMA_METHOD = 'adaptive' # 'adaptive' or 'fixed'
BETA = 0.3                # Parameter for geometry-adaptive kernels
K_NEAREST = 3             # k for k-nearest neighbors

# For other datasets, you would use a fixed sigma (as per Table 2 in the paper)
# e.g., for ShanghaiTech Part_B:
# SIGMA_METHOD = 'fixed'
# FIXED_SIGMA = 15

# --- Network Architecture ---
# The paper states the output of the network is 1/8 of the original input size.
# This downsampling factor is crucial for creating the target density map.
DOWNSAMPLE_RATIO = 8