import torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable



import torch.nn.functional as F

def preprocess_data(input_data):
    # Normalize input data to 0-1 range
    normalized_data = (input_data - input_data.min()) / (input_data.max() - input_data.min())
    
    # Apply Gaussian filter for noise reduction
    denoised_data = gaussian_filter(normalized_data, sigma=1)
    
    return denoised_data

#load data from matlab
import scipy.io as sio
import numpy as np
# Load the .mat file
d = sio.loadmat('data.mat')

# Assuming `truncatedBsVol` and `truncatedAnchor` are cell arrays,
# and each contains N elements of 3D arrays (11x11x95)

# Convert cell arrays to regular numpy arrays
X_cell_array = d['truncatedBsVol']
Y_cell_array = d['truncatedAnchor']

# The cell arrays come in as an object type array in numpy.
# You will need to iterate through it and stack the individual numpy arrays inside.
# Extract all items from the cell array and stack them along a new 'samples' dimension.
X = np.stack([X_cell_array[0, i] for i in range(X_cell_array.shape[1])], axis=0)
Y = np.stack([Y_cell_array[0, i] for i in range(Y_cell_array.shape[1])], axis=0)

# Add a channel dimension and make sure the shape is like (n_samples, n_channels, depth, height, width)
X = X[:, np.newaxis, ...]
Y = Y[:, np.newaxis, ...]


# Assume X and Y are your dataset numpy arrays with shape [n_samples, depth, height, width]
X_preprocessed = (X)
Y_preprocessed = (Y)

# Convert NumPy arrays to PyTorch tensors
X_tensor = torch.tensor(X_preprocessed).float()
Y_tensor = torch.tensor(Y_preprocessed).long()  # Use long for categorical labels

# Create a dataset and a dataloader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Batch size depending on your GPU memory






class DoubleConv(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling + double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Use ConvTranspose3d for upsampling
        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Pad on depth, height, and width if necessary
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)
        
        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self):
        super(UNet3D, self).__init__()
        
        # Depending on GPU memory you may reduce the number of channels 
        self.inc = DoubleConv(1, 8)
        
        # Downsampling
        self.down1 = DoubleConv(8, 16)
        
        # Upsampling 
        self.up1 = Up(16 + 8, 8)

        # The final layer would give two channels output because your num_classes is 2.
        # These two channels correspond to background and tumor respectively.
        self.outc = OutConv(8, 2)
        
    def forward(self, x):
        
        # Contracting Path (downsampling)
        x1 = self.inc(x)
        
        # If max pooling is used then this line replaces "self.down1" layer call. 
        #x = F.max_pool3d(x1,kernel_size=2,stride=2) 
        
        # Downsampling layer call should not do max pooling itself if previous line used.
        x_downsampled = self.down1(x1) 

         # Expansive Path (upsampling)
         # Original feature maps from downsampling paths (skip connections) are passed here along with their twin upsampled feature map. 
         # For our given data shape we will not have level symmetry so mix n match the levels as seems fit for skip connections.
         # Changing below line to accept any asymmetry between contractive-expansive path.
         # Add necessary padding if needed to match spatial dimensions when concatenating.
        x_upsampled = self.up1(x_downsampled,x1)

        logits = self.outc(x_upsampled)
        return logits
    
model = UNet3D()

# Move model to device if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)





criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

num_epochs = 10  # Replace with the actual number of epochs you want to train

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        # Convert tensors to Variables
        data, target = Variable(data.to(device)), Variable(target.to(device))

        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)

        # Compute loss
        loss = criterion(output.squeeze(), target.squeeze())

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}")

torch.save(model.state_dict(), 'unet3d_model.pth')

    