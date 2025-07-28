"""Trainable layers
"""
import torch
import torch.nn.functional as F
import json
import tinycudann as tcnn
import torch
import os
from zunis.tensor_check import check_tensor

class OverallAffineLayer(torch.nn.Module):
    """Learnable overall affine transformation
    f(x) = alpha x + delta
    """

    def __init__(self, alpha=10., delta=0.):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.delta = torch.nn.Parameter(torch.tensor(delta), requires_grad=True)

    def forward(self, input):
        """Output of the OverallAffineLayer"""
        return input * self.alpha + self.delta

class LambdaLayer(torch.nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class ToHalf(torch.nn.Module):
    """Layer to convert tensors to half precision."""
    def __init__(self):
        super(ToHalf, self).__init__()

    def forward(self, x):
        return x.half()

class OneBlobEncodingLayer(torch.nn.Module):
    """One-Blob Encoding Layer using a Gaussian kernel for batched input vector s ∈ [0, 1], without normalization."""
    def __init__(self, num_bins=32, sigma_scale=1.0):
        super(OneBlobEncodingLayer, self).__init__()
        self.num_bins = num_bins
        # Sigma is scaled inversely with the number of bins
        self.sigma = sigma_scale / num_bins
        # Precompute the centers of each bin
        self.bin_centers = torch.linspace(0.5 / num_bins, 1 - 0.5 / num_bins, num_bins)

    def forward(self, s):
        """Evaluate the Gaussian kernel centered at each element of s on each bin center"""
        # Ensure s is of shape [batch_size, num_features]
        batch_size, num_features = s.shape
        s = s.unsqueeze(2)  # shape [batch_size, num_features, 1] for broadcasting

        # Ensure bin_centers is properly broadcasted across batch and features
        bin_centers = self.bin_centers.to(s.device).to(s.dtype)  # shape [num_bins]
        bin_centers = bin_centers.view(1, 1, self.num_bins)  # shape [1, 1, num_bins] for broadcasting

        # Calculate the Gaussian function
        gaussians = torch.exp(-0.5 * ((s - bin_centers) / self.sigma) ** 2)  # shape [batch_size, num_features, num_bins]

        # Flatten along features and bins
        flattened_output = gaussians.view(batch_size, -1)  # shape [batch_size, num_features * num_bins]

        return flattened_output

class OneBlobEncodingOnY_N(torch.nn.Module):
    def __init__(self, y_n_dim, opt_feats_dim, num_bins=32, sigma_scale=1.0):
        super(OneBlobEncodingOnY_N, self).__init__()
        self.y_n_dim = y_n_dim
        self.opt_feats_dim = opt_feats_dim
        self.encoding_layer = OneBlobEncodingLayer(num_bins=num_bins, sigma_scale=sigma_scale)

    def forward(self, x):
        # Split the input into y_n and opt_feats
        y_n = x[:, :self.y_n_dim]
        opt_feats = x[:, self.y_n_dim:]
        # Apply OneBlobEncoding only to y_n
        y_n_encoded = self.encoding_layer(y_n)
        # Concatenate the encoded y_n with opt_feats
        return torch.cat((y_n_encoded, opt_feats), dim=1)

def create_rectangular_dnn(
        *,
        d_in,
        d_out,
        d_hidden,
        n_hidden,
        input_block = None,
        input_activation=None,
        hidden_activation=torch.nn.ReLU,
        output_activation=None,
        use_batch_norm=False):

        layers = []

        if input_block is not None:
            layers.append(input_block)
            #layers.append(LambdaLayer(lambda x: x.float()))


        if input_activation is not None:
            layers.append(input_activation())
        layers.append(torch.nn.Linear(d_in,d_hidden))
        layers.append(hidden_activation())
        if use_batch_norm:
            layers.append(torch.nn.BatchNorm1d(d_hidden))

        for i in range(n_hidden):
            layers.append(torch.nn.Linear(d_hidden, d_hidden))
            layers.append(hidden_activation())
            if use_batch_norm:
                layers.append(torch.nn.BatchNorm1d(d_hidden))

        layers.append(torch.nn.Linear(d_hidden, d_out))

        if output_activation is not None:
            layers.append(output_activation())

        return torch.nn.Sequential(*layers)


class ArbitraryShapeRectangularDNN(torch.nn.Module):
    """Rectangular DNN with the output layer reshaped to a given shape"""
    def __init__(self, *,
                 d_in,
                 out_shape,
                 d_hidden,
                 n_hidden,
                 input_activation=None,
                 hidden_activation=torch.nn.ReLU,
                 output_activation=None,
                 use_batch_norm=False):

        super(ArbitraryShapeRectangularDNN, self).__init__()
        self.out_shape = out_shape

        d_out = 1
        for d in out_shape:
            d_out *= d

        self.nn = create_rectangular_dnn(d_in=d_in,
                                         d_out=d_out,
                                         d_hidden=d_hidden,
                                         n_hidden=n_hidden,
                                         input_activation=input_activation,
                                         hidden_activation=hidden_activation,
                                         output_activation=output_activation,
                                         use_batch_norm=use_batch_norm)

    def forward(self, x):
        return self.nn(x).view(*(x.shape[:-1]), *self.out_shape)



class ArbitraryShapeRectangularTinyCudaNN(torch.nn.Module):
    """Rectangular DNN with the output layer reshaped to a given shape"""
    def __init__(self, *,
                 d_in,
                 out_shape,
                 d_hidden,
                 n_hidden,
                 input_activation=None,
                 hidden_activation=torch.nn.ReLU,
                 output_activation=None,
                 use_batch_norm=False):

        super(ArbitraryShapeRectangularTinyCudaNN, self).__init__()

        execution_path = os.getcwd()
        with open(execution_path+"/config.json") as f:
            config = json.load(f)
        print(config)
        self.out_shape = out_shape

        d_out = 1
        for d in out_shape:
            d_out *= d

        layers = []
        if input_activation is not None:
            layers.append(input_activation())
        network = tcnn.Network(d_in, d_out, config["network"])
        layers.append(network)
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        r = self.nn(x)
        return r.view(*(x.shape[:-1]), *self.out_shape)

class ArbitraryShapeRectangularOneBlob(torch.nn.Module):
    """Rectangular DNN with the output layer reshaped to a given shape"""
    def __init__(self, *,
                 d_in_y_n,
                 d_in_opt_feats,
                 out_shape,
                 d_hidden,
                 n_hidden,
                 input_activation=None,
                 hidden_activation=torch.nn.ReLU,
                 output_activation=None,
                 use_batch_norm=False):

        super(ArbitraryShapeRectangularOneBlob, self).__init__()
        self.out_shape = out_shape

        d_out = 1
        for d in out_shape:
             d_out *= d

        d_in_real = d_in_y_n * 32 + d_in_opt_feats

        print("ONE BLOB ENCODING!!")

        input_block = OneBlobEncodingOnY_N(y_n_dim=d_in_y_n,
                                           opt_feats_dim=d_in_opt_feats,
                                           num_bins=32,
                                           sigma_scale=1.0)

        self.nn = create_rectangular_dnn(d_in=d_in_real,
                                         d_out=d_out,
                                         d_hidden=d_hidden,
                                         n_hidden=n_hidden,
                                         input_activation=None,
                                         hidden_activation=hidden_activation,
                                         output_activation=output_activation,
                                         use_batch_norm=use_batch_norm,
                                         input_block= input_block)

    def forward(self, x):
        check_tensor(x, "x", force_analyze=False)
        return self.nn(x).view(*(x.shape[:-1]), *self.out_shape)

'''
Every per-layer network has a U-net (see Figure 3) with 8 fully connected
layers, where the outermost layers contain 256 neurons and the
number of neurons is halved at every nesting level. We use 2 ad-
ditional layers to adapt the input and output dimensionalities to
and from 256, respectively. The networks only differ in their output
layer to produce the desired parameters of their respective coupling
transform (s and t, bQ, or bW and bV ).
'''

class FullyConnectedUNetDNN(torch.nn.Module):
    def __init__(self,
                 d_in,
                 out_shape,
                 hidden_dims):

        super(FullyConnectedUNetDNN, self).__init__()
        self.out_shape = out_shape

        d_out = 1
        for d in out_shape:
            d_out *= d

        self.encoder = torch.nn.ModuleList()
        current_dim = d_in
        for h_dim in hidden_dims:
            self.encoder.append(torch.nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        self.decoder = torch.nn.ModuleList()
        for h_dim in reversed(hidden_dims[:-1]):
            self.decoder.append(torch.nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        self.final_layer = torch.nn.Linear(current_dim, d_out)

    def forward(self, x):

        skip_connections = []
        for layer in self.encoder:
            x = layer(x)
            x = F.relu(x)
            skip_connections.append(x)

        for layer in self.decoder:
            skip_x = skip_connections.pop()
            x = layer(x + skip_x)
            x = F.relu(x)

        x = self.final_layer(x)
        return x

