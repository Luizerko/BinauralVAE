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
    def __init__(self, image_dimensions, image_channels=2, latent_dim_pow=8, n_filters=5, ks_v=5, ks_h=3, s_v=2, s_h=1, pad=0):
        # Initializing network
        super(CVAE, self).__init__()
        self.image_dimensions = image_dimensions
        self.latent_dim = 2**latent_dim_pow

        # Defining encoder
        encoder = []
        for i in range(n_filters):
            if i == 0:
                encoder.append(ComplexConv2d(image_channels, 2**(9), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad))
            else:
                encoder.append(ComplexConv2d(2**(9-i+1), 2**(9-i), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad))
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
        in_features = 2**(9-n_filters+1) * v_in_features * h_in_features

        # Defining mean, (real) variance, and complex pseudo-covariance branches. It is important to introduce the pseudo-covariance so that we can have the improper complex gaussians, which mean not only circular distributions on the latent variables. This is a key ingredient to make the posterior for CVAE as flexible as the VAE one
        self.mu = ComplexLinear(in_features, self.latent_dim)
        self.logsigma = ComplexLinear(in_features, self.latent_dim)
        self.delta = ComplexLinear(in_features, self.latent_dim)

        # Output padding computation for transpose convolutions
        out_pad_v = []
        out_pad_h = []
        for i in range(n_filters):
            out_pad_v.append(calc_out_pad(self.shapes[-i-2][0], self.shapes[-i-1][0], ks_v, s_v, pad))
            out_pad_h.append(calc_out_pad(self.shapes[-i-2][1], self.shapes[-i-1][1], ks_h, s_h, pad))

        # Defining decoder
        self.decoder_input_1 = ComplexLinear(self.latent_dim, self.latent_dim//2)
        self.decoder_input_2 = ComplexLinear(self.latent_dim//2, in_features)
        decoder = [nn.Unflatten(1, (2**(9-(n_filters-1)), v_in_features, h_in_features))]
        for i in range(n_filters):
            if i == n_filters-1:
                decoder.append(ComplexConvTranspose2d(2**(9-(n_filters-1)+i), image_channels, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(out_pad_v[i], out_pad_h[i])))
                
                # Tanh separately on real and imaginary parts to bound the complex reconstruction
                decoder.append(nn.Tanh()) 
            else:
                decoder.append(ComplexConvTranspose2d(2**(9-(n_filters-1)+i), 2**(9-(n_filters-1)+i+1), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(out_pad_v[i], out_pad_h[i])))
                decoder.append(ComplexReLU())
        self.decoder = nn.Sequential(*decoder)

    # Initialization function
    # def init_weights(self, m, method='torch_default'):
    #     if isinstance(m, (ComplexConv2d, ComplexConvTranspose2d, ComplexLinear)):
    #         if method == 'he':
    #             nn.init.kaiming_uniform_(m.weight.real, nonlinearity='relu')
    #             nn.init.kaiming_uniform_(m.weight.imag, nonlinearity='relu')
    #         elif method == 'xavier':
    #             nn.init.xavier_uniform_(m.weight.real)
    #             nn.init.xavier_uniform_(m.weight.imag)
    #         elif method == 'torch_default':
    #             nn.init.kaiming_uniform_(m.weight.real, a=math.sqrt(5))
    #             nn.init.kaiming_uniform_(m.weight.imag, a=math.sqrt(5))
            
    #         if m.bias is not None:
    #             if method == 'torch_default':
    #                 fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    #                 bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    #                 nn.init.uniform_(m.bias.real, -bound, bound)
    #                 nn.init.uniform_(m.bias.imag, -bound, bound)
    #             else:
    #                 nn.init.constant_(m.bias.real, 0)
    #                 nn.init.constant_(m.bias.imag, 0)


    # Complex Reparametrization trick
    def reparameterize(self, mu, sigma, delta):
        # Computing reparametrization trick with h = mu + k_x * epsilon_x + k_y + epsilon_y. On a high-level, the idea is to create two multivariate standard gaussian distributions, one for the real parts of each latent varibale and one for the imaginary parts. Then we do something analogous to the VAE, where we sample the new standard gaussian so that we can stretch it with the variance and shift it by the mean, but it's a bit more tricky because we need both of these new distributions to have the same covariance as our initial complex variables (remember that latent variables are independent from one another, but there's dependency between their real and imaginary parts). For traditional VAEs, Var(epsilon * sigma) = sigma^2 * Var(epsilon) = sigma^2. For CVAEs, we want to stretch the two distributions with a L so that their new covariance matrix is S, so z = A*epsilon => Cov(A*epsilon_xy) = A*Cov(epsilon_xy)*A^T = AA^T = S. Getting to the exact formulae of both k_x and k_y is a bit longer though, so make sure to check out our technical report
        denominator = torch.clamp(2*sigma + 2*torch.real(delta), min=1e-8)
        kx = (sigma + delta) / denominator
        
        numerator = torch.clamp(sigma**2 - torch.abs(delta)**2, min=1e-8)
        ky = 1j * numerator / denominator

        epsilon_r = torch.randn(mu.shape, device=mu.device)
        epsilon_i = torch.randn(mu.shape, device=mu.device)

        return mu + kx*epsilon_r + ky*epsilon_i
    
    # Forward function
    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        
        # Sigma must be strictly real and positive (represents total variance, the "size" of the distribution for each variable)
        sigma = torch.exp(torch.real(self.logsigma(h)))
        
        # delta expresses the correlation between real and imaginary parts of each variable. To maintain a valid covariance matrix, it's bounded by |delta| < sigma. If |delta| = sigma, the elipse of real/imaginary relationship collapses into one line, so we lose our complex value varibale. If |delta| > sigma, the determinant of the covariance matrix becomes negative, which at this point is no longer a covariance matrix since it must be a positive definite matrix. By computing rho the way we do, we first compute the unit complex by dividing by the magnitude (finding the "direction" of the number) and then we scale it by [0, 1) depending on the magnitude. With these two combined, we guarantee |delta| < sigma, but still give delta variables two degrees of freedom 
        delta_raw = self.delta(h)
        delta_mag = torch.abs(delta_raw)
        rho = delta_raw / (delta_mag + 1e-8) * torch.tanh(delta_mag)
        delta = sigma * rho

        z = self.reparameterize(mu, sigma, delta)
        
        h_dec = self.decoder_input_1(z)
        h_dec = self.decoder_input_2(h_dec)
        return self.decoder(h_dec), mu, sigma, delta
    
    # Loss function rec_loss + beta*kl_loss
    def loss(self, rec, x, mu, sigma, delta, beta=2.0):
        # Complex L2 norm as reconstruction loss. Analogously to the conventional VAE, we assume p_theta(z|h) = N_c(z; a, I, O), with z the complex input and h the complex latent variables, so the output of the CVAE is just the complex mean a. For clarity, this distribution means identity covariance (unit circle for every complex variable) and zero pseudo-covariance (circular distribution for every complex variable). This also means that the maximum likelihood estimator for p_theta(z|h) can be approximated with just the squared norm of the difference between the data input an the mean output of the CVAE
        rec_loss = torch.sum(torch.abs(x - rec)**2) / x.size(0)
        
        # Complex KL-Divergence against standard circular standard complex gaussian CSCG prior N_c(0, I, O). Formally, D_KL(q_phi(h|z) || p_theta(h)) = D_KL(N_c(mu, sigma, delta) || N_c(0, I, O)) = mu^H * mu + ||sigma - 1 - 1/2 * log(sigma^2 - |delta|^2)||_1
        mean_term = torch.real((torch.conj(mu) * mu).sum(dim=1))
        log_det = torch.log(torch.clamp(sigma ** 2 - torch.abs(delta) ** 2, min=1e-8))
        variance_term = torch.abs(sigma - 1 - log_det / 2).sum(dim=1)
        kl_loss = torch.mean(mean_term + variance_term)
        
        total_loss = rec_loss + beta * kl_loss
        return rec_loss, kl_loss, total_loss