# model.py

import torch
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        
        # --- Front-End ---
        # As per the paper, the front-end is the first 10 layers of VGG-16
        # with 3 max-pooling layers. This corresponds to all layers before
        # the 4th max-pooling layer in the standard VGG-16 architecture.
        # In PyTorch's torchvision.models.vgg16, this is the first 23 layers
        # of the `features` module.
        vgg = models.vgg16(pretrained=True)
        self.frontend_feat = vgg.features[:23]

        # --- Back-End ---
        # The back-end consists of dilated convolutional layers.
        # We implement the 'CSRNet B' configuration from Table 3 in the paper,
        # as it gave the best performance on ShanghaiTech Part_A.
        # The dilation rates are all 2. Padding is set to maintain resolution.
        # Padding = (dilation * (kernel_size - 1)) / 2 = (2 * (3-1)) / 2 = 2.
        self.backend_feat = [
            512, 512, 512, 256, 128, 64
        ]
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        
        # --- Output Layer ---
        # A 1x1 convolution to generate the final single-channel density map.
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Initialize weights for the back-end and output layer.
        # The paper specifies Gaussian initialization with std=0.01.
        if not load_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        # Iterate over all modules in the network
        for m in self.modules():
            # Initialize Conv2d layers
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # Initialize biases to 0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # Initialize BatchNorm2d layers
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        if dilation:
            d_rate = 2 # Dilation rate for CSRNet-B
        else:
            d_rate = 1
        
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
    # Define frontend and backend as properties to make layer access clean
    @property
    def frontend(self):
        return self.frontend_feat


if __name__ == '__main__':
    # --- Example Usage & Sanity Check ---
    # Create a dummy input tensor with a shape similar to a cropped image
    # from the dataset (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 384, 512)

    # Instantiate the model
    model = CSRNet()
    
    # Pass the dummy input through the model
    output = model(dummy_input)

    print("--- Model Sanity Check ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # The output shape should be (Batch, 1, H/8, W/8)
    expected_h = dummy_input.shape[2] // 8
    expected_w = dummy_input.shape[3] // 8
    print(f"Expected output shape: (1, 1, {expected_h}, {expected_w})")
    assert output.shape == (1, 1, expected_h, expected_w), "Output shape is incorrect!"
    print("\nModel test passed successfully!")