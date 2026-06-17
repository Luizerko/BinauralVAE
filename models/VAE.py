import torch
import torch.nn as nn
import torch.nn.functional as F

# Computing output dimension of conv layer for later computing proper output padding
def calc_out_shape(in_dim, ks, s, pad):
    return int((in_dim + 2 * pad - (ks - 1) - 1) / s + 1)

# Computing padding for proper transpose convolution
def calc_out_pad(target_dim, in_dim, ks, s, pad):
    return target_dim - ((in_dim - 1) * s - 2 * pad + (ks - 1) + 1)

# Defining VAE model
class VAE(nn.Module):
    def __init__(self, image_dimensions, image_channels=2, latent_dim=32, n_filters=3, ks_v=7, ks_h=5, s_v=2, s_h=1, pad=0):
        # Initializing network
        super(VAE, self).__init__()
        self.image_dimensions = image_dimensions
        self.latent_dim = latent_dim

        # Defining encoder
        encoder = []
        for i in range(n_filters):
            if i == 0:
                encoder.append(nn.Conv2d(image_channels, 2**(5+i), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad))
            else:
                encoder.append(nn.Conv2d(2**(5+i-1), 2**(5+i), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad))
            encoder.append(nn.ReLU())
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

        # Defining mean and log-variance branches
        self.mu = nn.Linear(in_features, self.latent_dim)
        self.logvar = nn.Linear(in_features, self.latent_dim)

        # Output padding computation for transpose convolutions
        out_pad_v = []
        out_pad_h = []
        for i in range(n_filters):
            out_pad_v.append(calc_out_pad(self.shapes[-i-2][0], self.shapes[-i-1][0], ks_v, s_v, pad))
            out_pad_h.append(calc_out_pad(self.shapes[-i-2][1], self.shapes[-i-1][1], ks_h, s_h, pad))

        # Defining decoder
        self.decoder_input = nn.Linear(latent_dim, in_features)
        decoder = [nn.Unflatten(1, (2**(5+(n_filters-1)), v_in_features, h_in_features))]
        for i in range(n_filters):
            if i == n_filters-1:
                decoder.append(nn.ConvTranspose2d(2**(5+(n_filters-1)-i), image_channels, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(out_pad_v[i], out_pad_h[i])))
                decoder.append(nn.Sigmoid())
            else:
                decoder.append(nn.ConvTranspose2d(2**(5+(n_filters-1)-i), 2**(5+(n_filters-1)-i-1), kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(out_pad_v[i], out_pad_h[i])))
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    # Reparametrization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    # Forward function: 
    #            -> linear mu     -|
    # encoder ->|                  |-> repar. trick -> linear -> decoder
    #            -> linear logvar -|
    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        h = self.decoder_input(z)
        return self.decoder(h), mu, logvar
    
    # Loss function rec_loss + beta*kl_loss
    def loss(self, rec, x, mu, logvar, beta=2.0):
        rec_loss = nn.functional.mse_loss(rec, x, reduction='sum') / x.size(0)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)
        total_loss = rec_loss + beta*kl_loss

        return rec_loss, kl_loss, total_loss