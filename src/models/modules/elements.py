
from torch import nn
import torch
import math

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size, padding=padding, groups=in_channels, stride=stride
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Dilated Convolution Block
class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate):
        super(DilatedConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=dilation_rate, dilation=dilation_rate
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Dilated_Blocks(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        # Parallel dilated convolutions
        self.dilated_conv1 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=1, padding='same')
        self.dilated_conv2 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=2, padding='same')
        self.dilated_conv4 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=4, padding='same')
        self.dilated_conv8 = nn.Conv1d(output_channels, output_channels // 4, kernel_size=3, dilation=8, padding='same')

        # Remove combine_conv since it's not used
        self.relu = nn.ReLU()

    def forward(self, x):
        # Parallel dilated convolutions
        out1 = self.relu(self.dilated_conv1(x))
        out2 = self.relu(self.dilated_conv2(x))
        out4 = self.relu(self.dilated_conv4(x))
        out8 = self.relu(self.dilated_conv8(x))

        # Concatenate along channel dimension
        out_concat = torch.cat([out1, out2, out4, out8], dim=1)
        return out_concat



class MemoryPositionalEncoding(nn.Module):
    """
    Simple 1D positional encoding for the memory (encoder output)
    to be used in the decoder's cross-attention mechanism.
    """

    def __init__(self, d_model, max_seq_len=100, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings once and for all
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices, cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Make pe not a model parameter (we don't need to train it)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        Returns:
            Positional encoding to be added to x, same shape as x
        """
        # Get positional encoding for the length of the input sequence
        seq_len = x.size(1)
        pos_encoding = self.pe[:, :seq_len, :]

        # Broadcast to batch dimension
        return pos_encoding

