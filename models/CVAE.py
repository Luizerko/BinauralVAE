import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexPyTorch.complexLayers import ComplexConv2d, ComplexLinear, ComplexConvTranspose2d
from complexPyTorch.complexFunctions import complex_relu


# Computing output dimension of conv layer for later computing proper output padding
def calc_out_shape(in_dim, ks, s, pad):
    return int((in_dim + 2 * pad - (ks - 1) - 1) / s + 1)


# Computing padding for proper transpose convolution
def calc_out_pad(target_dim, in_dim, ks, s, pad):
    return target_dim - ((in_dim - 1) * s - 2 * pad + (ks - 1) + 1)


# Wrapper module to use complex functions inside nn.Sequential
class ComplexReLU(nn.Module):
    def forward(self, x):
        return complex_relu(x)


# Defining Complex VAE model
class CVAE(nn.Module):
    def __init__(self, image_dimensions, image_channels=1, latent_dim_pow=5, n_filters=3, ks_v=7, ks_h=5, s_v=2, s_h=1, pad=0):
        # Initializing network
        super(CVAE, self).__init__()
        self.image_dimensions = image_dimensions
        self.latent_dim = 2**latent_dim_pow

        # Defining encoder
        encoder = []
        for i in range(n_filters):
            if i == 0:
                encoder.append(ComplexConv2d(image_channels, 2**(5+i), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad))
            else:
                encoder.append(ComplexConv2d(2**(5+i-1), 2**(5+i), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad))
            encoder.append(ComplexReLU())
        encoder.append(nn.Flatten())
        self.encoder = nn.Sequential(*encoder)

        # Computing all shapes for later decoder output_padding
        self.shapes = [image_dimensions]
        for _ in range(n_filters):
            prev_v, prev_h = self.shapes[-1]
            next_v = calc_out_shape(prev_v, ks_v, s_v, pad)
            next_h = calc_out_shape(prev_h, ks_h, s_h, pad)
            self.shapes.append((next_v, next_h))

        # Computing number of input features for fully-connected layer
        v_in_features, h_in_features = self.shapes[-1]
        in_features = 2**(5+n_filters-1) * v_in_features * h_in_features

        # Defining mean, (real) variance, and complex pseudo-covariance branches. It is important to introduce the pseudo-covariance so that we can have the improper clomplex gaussians, which mean not only circular distributions on the latent variables. This is a key ingredient to make the posterior for CVAE as flexible as the VAE one
        self.mu = ComplexLinear(in_features, self.latent_dim)
        self.sigma = ComplexLinear(in_features, self.latent_dim)
        self.delta = ComplexLinear(in_features, self.latent_dim)

        # Output padding computation for transpose convolutions
        out_pad_v = []
        out_pad_h = []
        for i in range(n_filters):
            out_pad_v.append(calc_out_pad(self.shapes[-i-2][0], self.shapes[-i-1][0], ks_v, s_v, pad))
            out_pad_h.append(calc_out_pad(self.shapes[-i-2][1], self.shapes[-i-1][1], ks_h, s_h, pad))

        # Defining decoder
        self.decoder_input = ComplexLinear(self.latent_dim, in_features)
        decoder = [nn.Unflatten(1, (2**(5+(n_filters-1)), v_in_features, h_in_features))]
        for i in range(n_filters):
            if i == n_filters-1:
                decoder.append(ComplexConvTranspose2d(2**(5+(n_filters-1)-i), image_channels, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(out_pad_v[i], out_pad_h[i])))
                
                # Sigmoid separately on real and imaginary parts to bound the complex reconstruction
                decoder.append(nn.Sigmoid()) 
            else:
                decoder.append(ComplexConvTranspose2d(2**(5+(n_filters-1)-i), 2**(5+(n_filters-1)-i-1), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(out_pad_v[i], out_pad_h[i])))
                decoder.append(ComplexReLU())
        self.decoder = nn.Sequential(*decoder)

    # Initialization function
    def init_weights(self, m, method='torch_default'):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            if method == 'he':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif method == 'torch_default':
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            
            if m.bias is not None:
                if method == 'torch_default':
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
                else:
                    nn.init.constant_(m.bias, 0)

    # Complex Reparametrization trick
    def reparameterize(self, mu, sigma, delta):
        # I need to better explain this
        numerator = sigma ** 2 - torch.abs(delta) ** 2
        denominator = 2 * sigma + 2 * torch.real(delta)

        # Signed square root guard
        numerator_sign = torch.sign(numerator.detach())
        denominator_sign = torch.sign(denominator.detach())
        numerator_sign = (numerator_sign - 1) / 2 * -1j + ((numerator_sign + 1) / 2)
        denominator_sign = (denominator_sign - 1) / 2 * -1j + ((denominator_sign + 1) / 2)
        numerator = numerator_sign * torch.sqrt(torch.abs(numerator))
        denominator = denominator_sign * torch.sqrt(torch.abs(denominator)) + 1e-8

        kx = (sigma + delta) / denominator
        ky = 1j * numerator / denominator

        # sampling two independent real standard normals
        epsilon_r = torch.randn(mu.shape, device=mu.device)
        epsilon_i = torch.randn(mu.shape, device=mu.device)

        return mu + kx * epsilon_r + ky * epsilon_i
    
    # Forward function
    def forward(self, x):
        h = self.encoder(x)
        
        mu = self.mu(h)
        
        # Sigma must be strictly real and positive (represents total variance)
        sigma = torch.exp(torch.real(self.sigma(h)))
        
        # Delta bounded by |delta| < sigma to maintain a valid covariance matrix
        delta_raw = self.delta(h)
        delta_mag = torch.abs(delta_raw)
        rho = delta_raw / (delta_mag + 1e-8) * torch.tanh(delta_mag)
        delta = sigma * rho

        z = self.reparameterize(mu, sigma, delta)
        
        h_dec = self.decoder_input(z)
        return self.decoder(h_dec), mu, sigma, delta
    
    # Loss function rec_loss + beta*kl_loss
    def loss(self, rec, x, mu, sigma, delta, beta=2.0):
        # Complex L2 norm as reconstruction loss
        rec_loss = torch.sum(torch.abs(x - rec)**2) / x.size(0)
        
        # Complex KL-Divergence against standard circular standard complex gaussian CSCG prior N_c(0, I, 0)
        log_det = torch.log(torch.clamp(sigma ** 2 - torch.abs(delta) ** 2, min=1e-8))
        mean_term = torch.real((torch.conj(mu) * mu).sum(dim=1))
        variance_term = torch.abs(sigma - 1 - log_det / 2).sum(dim=1)
        kl_loss = torch.mean(mean_term + variance_term)
        
        total_loss = rec_loss + beta * kl_loss
        return rec_loss, kl_loss, total_loss