import torch
import torch.nn as nn
import torch.nn.functional as F

# Computing output dimension of conv layer for later computing proper output padding
def calc_out_shape(in_dim, ks, s, pad):
    return int((in_dim - ks + 2 * pad) / s) + 1

# Computing output padding for proper transpose convolution
def calc_out_pad(target_dim, in_dim, ks, s, pad):
    return target_dim - (in_dim - 1) * s - 2 * pad + ks

# Defining VAE model
class VAE(nn.Module):
    def __init__(self, image_dimensions, image_channels=2, latent_dim=32, filters1=32, filters2=64, filters3=128, ks_v=7, ks_h=5, s_v=2, s_h=1, pad=0):
        # Initializing network
        super(VAE, self).__init__()
        self.image_dimensions = image_dimensions
        self.latent_dim = latent_dim

        # Defining encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, filters1, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad),
            nn.ReLU(),

            nn.Conv2d(filters1, filters2, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad),
            nn.ReLU(),

            nn.Conv2d(filters2, filters3, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad),
            nn.ReLU(),

            nn.Flatten()
        )

        # Computing all shapes for later decoder output_padding
        self.shapes = [image_dimensions]
        for _ in range(3):
            prev_v, prev_h = self.shapes[-1]
            next_v = calc_out_shape(prev_v, ks_v, s_v, pad)
            next_h = calc_out_shape(prev_h, ks_h, s_h, pad)
            self.shapes.append((next_v, next_h))

        # Computing number of input features for fully-connected layer
        v_in_features, h_in_features = self.shapes[-1]
        in_features = filters3 * v_in_features * h_in_features

        # Defining mean and log-variance branches
        self.mu = nn.Linear(in_features, self.latent_dim)
        self.logvar = nn.Linear(in_features, self.latent_dim)

        # Output padding computation for transpose convolutions
        op1_v = calc_out_pad(self.shapes[-2][0], self.shapes[-1][0], ks_v, s_v, pad)
        op1_h = calc_out_pad(self.shapes[-2][1], self.shapes[-1][1], ks_h, s_h, pad)
        
        op2_v = calc_out_pad(self.shapes[-3][0], self.shapes[-2][0], ks_v, s_v, pad)
        op2_h = calc_out_pad(self.shapes[-3][1], self.shapes[-2][1], ks_h, s_h, pad)
        
        op3_v = calc_out_pad(self.shapes[-4][0], self.shapes[-3][0], ks_v, s_v, pad)
        op3_h = calc_out_pad(self.shapes[-4][1], self.shapes[-3][1], ks_h, s_h, pad)

        # Defining decoder
        self.decoder_input = nn.Linear(latent_dim, in_features)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (filters3, v_in_features, h_in_features)),

            nn.ConvTranspose2d(filters3, filters2, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(op1_v, op1_h)),
            nn.ReLU(),

            nn.ConvTranspose2d(filters2, filters1, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(op2_v, op2_h)),
            nn.ReLU(),

            nn.ConvTranspose2d(filters1, image_channels, kernel_size=(ks_v, ks_h), stride=(s_v, s_h), padding=pad, output_padding=(op3_v, op3_h)),
            nn.Sigmoid(),
        )

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
        return self.decoder(h)
    
    # Loss function rec_loss + beta*kl_loss
    def loss(self, rec, x, mu, logvar, beta=2.0):
        rec_loss = nn.functional.mse_loss(rec, x, reduction='sum') / x.size(0)

        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / x.size(0)

        total_loss = rec_loss + beta*kl_loss

        return rec_loss, kl_loss, total_loss