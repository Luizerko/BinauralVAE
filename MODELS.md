# Models

In this section, we explain our models' architectures, the ideas behind them, and eventually the mathematical foundations that make them work. This should provide enough substance for you to understand the approaches, try them yourself, and potentially enhance them with your creativity. The primary objective of this project is to properly reconstruct binaural audio for a 0.2-second sound segment (representing one robot action in a simulation). By doing this, we train our latent space to encode binaural audio efficiently enough to serve as a latent state for a future audio-based world model. 

To achieve this, we started simple, utilizing one of the most classic reconstruction pipelines, the Variational Autoencoder (VAE), notably inspired by the architecture used by David Ha et al. in their [World Models paper](https://arxiv.org/pdf/1803.10122). While simple, our CNN-based architecture is fast to implement, "quick" to train, and crucially, makes it easier to trace successes and failures back to specific architectural changes. Furthermore, this architecture allows us the flexibility to adapt to the complex realm, making better use of our acoustic signals. However, we must accept a fundamental trade-off: our output is the maximum likelihood estimator of our data given our latent variables, which is equivalent to the mean of a normal distribution with an identity covariance. The price we pay for this is a lack of proper variance, reflecting as a loss of fine detail in the reconstructions, a characteristic you might notice in our outputs regardless of the approach taken.

---

## Mel Spectrogram VAE

For our first approach, we took the classic route: analyzing Mel spectrograms. The idea is to treat Mel spectrograms as image inputs and use a CNN-based VAE to process them. The fundamental flaw with this approach is that spectrograms are not images. They possess a time axis that CNNs inherently treat with translation equivariance. If our goal was speech recognition, this could cause us to lose critical timing, event, or phoneme information. However, for spatial understanding, the time issue manifests differently: **Internal Time Difference (ITD).**

ITD is the microsecond-level difference in time it takes for a sound to reach one ear versus the other. On our simulator ([AudioWorldSim](https://github.com/Luizerko/AudioWorldSim)), we capture data at 44.1KHz to satisfy the Nyquist theorem for human hearing (up to ~22.05KHz). We compute every frame of the Mel spectrogram using 2048 samples, which equates to roughly $0.046$ seconds, orders of magnitude larger than the microsecond ITD humans use to localize sound. Therefore, Mel spectrograms physically cannot capture this crucial temporal spatial cue.

Fortunately, humans rely on two other techniques alongside ITD: the **Internal Level Difference (ILD)**, which is the difference in volume between the ears, and the **Spectral Cues**, which are torso and ear-shape frequency filtering (modeled by HRTFs) that create frequency peaks to help us differentiate sounds coming from above, below, front, or back. Because ILD and spectral cues are captured by Mel spectrograms, all hope is not lost. The architecture provides good general reconstructions and captures these level differences well, though lacking in fine texture. The overall audio reconstruction is satisfactory, though it carries an inherent metallic artifact because we must rely on estimation algorithms (like Griffin-Lim) to approximate the missing phase when inverting the Short-Time Fourier Transform (STFT).

[]

---

## STFT 4-Channel Stacked VAE

This architecture mirrors the Mel VAE, but instead of Mel spectrograms, we directly use the magnitude and phase from the STFT of our signal. By including the phase, we gain access to the instantaneous frequency shifts that differentiate the left and right ears, effectively solving our missing ITD problem. Because we now have phase data, we altered the time-frequency trade-off of the STFT, reducing the window from 2048 to 1024 samples. With Mel, we decided to favor frequency resolution because phase was absent, but here, we can afford to be more time-biased.

We stacked the information into four channels: Left Magnitude, Left Phase, Right Magnitude, and Right Phase. You might think our problem is solved, but it's actually not. Phase information is notoriously difficult for neural networks to reconstruct for two reasons. First, it looks like absolute noise, meaning it requires highly variant, fine-detail reconstruction (which VAEs struggle with). Second, phase is circular, wrapping from $-\pi$ to $\pi$. A network treating this as linear image data fails to understand that $-\pi$ and $\pi$ are the exact same value.

The magnitude reconstruction performs decently, but as expected, the phase reconstruction fails drastically. Because the network cannot parse the circular variance of phase, it takes the safest path to minimize Mean Squared Error: it mostly predicts an average, smoothed-out value for all inputs.

[]

---

## Complex-Valued VAE (CVAE)

Our third approach attempts to capture the best of both worlds: excellent magnitude reconstruction without sacrificing phase. By upgrading from real-valued networks to complex-valued networks, we respect the inherently complex nature of STFT outputs. This structure naturally enforces the cyclic nature of phase, yielding promising spatial audio reconstructions. This approach is heavily detailed in [Nakashika's paper](https://www.isca-archive.org/interspeech_2020/nakashika20_interspeech.pdf), but we'll provide an introduction here too.

Transitioning a VAE to a CVAE requires adapting the network parameters to complex values (storing a real branch and an imaginary branch for every layer) and expanding our probability framework to complex distributions. Our input, prior, posterior, and likelihood functions all become Complex Gaussian distributions.

### Understanding Complex Gaussians

A standard real Gaussian uses a covariance matrix. A Complex Gaussian requires two matrices: a **Covariance Matrix** $\mathbf{\Gamma} \in \mathbb{C}^{D \times D}$, and a **Pseudo-Covariance Matrix** $\mathbf{C} \in \mathbb{C}^{D \times D}$. This second matrix might seem a little weird at first, but the intuition is straightforward: in a real Gaussian, every variable just has one value, so you only need one matrix to describe its relationship with itself (the spread) and its relationship with every other variable (the skew). However, in a Complex Gaussian, every variable lives in 2D space. Therefore, every pair of variables has four distinct relationships: $real_a$ to $real_b$, $real_a$ to $imag_b$, $imag_a$ to $real_b$, and $imag_a$ to $imag_b$ ($2 \times 2 = 4$ relationships). That’s exactly why a single matrix isn't enough, and we need two.

Mathematically, the covariance matrix is defined using the conjugate transpose: $ \mathbf{\Gamma} = \mathbb{E}[(\mathbf{z} - \boldsymbol{\mu})(\mathbf{z} - \boldsymbol{\mu})^H] $. The diagonal elements here are real, positive numbers representing the total variance ($\sigma_{real} + \sigma_{imag}$), essentially, how big the distribution's "circle" is. The off-diagonal elements are complex numbers that correlate variables in magnitude and phase. Conversely, the pseudo-covariance matrix is defined using the standard transpose: $ \mathbf{C} = \mathbb{E}[(\mathbf{z} - \boldsymbol{\mu})(\mathbf{z} - \boldsymbol{\mu})^T] $. The diagonal elements here are complex numbers capturing the intrinsic relationship between the real and imaginary parts of the same complex variable (allowing a single complex variable to form a skewed ellipse rather than a perfect circle). The off-diagonal elements capture the relationships between the real and imaginary parts of one complex variable with another.

To get a better intuition of what is happening under the hood, suppose we have two zero-mean complex variables $z_1 = x_1 + iy_1$ and $z_2 = x_2 + iy_2$. Let's look at how their covariance and pseudo-covariance capture those 4 unknown relationships. First, the covariance (off-diagonal elements): $ \mathbb{E}[z_1 z_2^*] = \mathbb{E}[(x_1 + iy_1)(x_2 - iy_2)] = \mathbb{E}[x_1x_2 + y_1y_2 + i(y_1x_2 - x_1y_2)] $. If we define our 4 core real relationships as expected values: $A = \mathbb{E}[x_1x_2]$, $B = \mathbb{E}[y_1y_2]$, $D = \mathbb{E}[y_1x_2]$, and $E = \mathbb{E}[x_1y_2]$, we get: $ \mathbb{E}[z_1 z_2^*] = (A + B) + i(D - E) $.

Next, the pseudo-covariance (also off-diagonal elements): $ \mathbb{E}[z_1 z_2] = \mathbb{E}[(x_1 + iy_1)(x_2 + iy_2)] = \mathbb{E}[x_1x_2 - y_1y_2 + i(y_1x_2 + x_1y_2)] $. Using the exact same definitions, we get: $ \mathbb{E}[z_1 z_2] = (A - B) + i(D + E) $. Because we have 4 unknowns ($A, B, D, E$), we strictly need both the covariance and pseudo-covariance to tell the full story. We need $(A + B)$ and $(A - B)$ to solve for the exact values of $A$ and $B$, and analogously, we need $(D - E)$ and $(D + E)$ to isolate $D$ and $E$. 

Now, if we scale this intuition down to a single zero-mean complex variable $z = x + iy$, we can clearly see how the diagonal elements of these matrices are formed. **Variance**: $ \sigma = \mathbb{E}[zz^*] = \mathbb{E}[(x+iy)(x-iy)] = \mathbb{E}[x^2 + y^2] = \sigma_{xx} + \sigma_{yy} $. This yields a real, positive number representing the total spread. **Pseudo-Variance**: $ \delta = \mathbb{E}[zz] = \mathbb{E}[(x+iy)(x+iy)] = \mathbb{E}[x^2 - y^2 + 2ixy] = (\sigma_{xx} - \sigma_{yy}) + 2i\sigma_{xy} $. This yields a complex number ($\delta_r + i\delta_i$) that captures the imbalance between the real and imaginary variances ($\sigma_{xx} - \sigma_{yy}$) and their correlation ($\sigma_{xy}$), defining how elliptical (and skewed) the distribution is.

And now, by combining $\sigma$ and $\delta$, we can perfectly reconstruct the underlying $2 \times 2$ real covariance matrix for the $x$ and $y$ components of a single complex variable:
$$ \mathbf{S} = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} \\ \sigma_{yx} & \sigma_{yy} \end{bmatrix} = \begin{bmatrix} \frac{\sigma + \delta_r}{2} & \frac{\delta_i}{2} \\ \frac{\delta_i}{2} & \frac{\sigma - \delta_r}{2} \end{bmatrix} $$

### CVAE Formulation and KL-Divergence

So now that we understand complex Gaussian distributions a bit better, let’s get to the actual formulation of the CVAE. Let’s first talk about the prior. For the standard VAE, the prior $p(\mathbf{h})$ is a standard normal $\mathcal{N}(\mathbf{0}, \mathbf{I})$. For the CVAE, we’ll also have a standard complex normal $\mathcal{N}_c(\mathbf{0}, \mathbf{I}, \mathbf{O})$, with the last parameter being a matrix full of zeros. This means there’s no correlation at all between complex variables, and every single one of them is a perfect unit circle distribution.

Now for the posterior. In the standard VAE, we have $q_{\phi}(\mathbf{h}|\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}, \Delta(\boldsymbol{\sigma}))$, with $\Delta$ indicating a diagonal matrix. This basically tells us that we have the freedom to expand variables individually, but not skew the distribution. This is the case for standard VAEs for two reasons: first, it makes the math tractable, and second, it actually gives every latent dimension a unique meaning - we don’t allow latent dimensions to correlate because we want them to express their individual information. For the exact same reasons, we’ll follow an analogous approach for the complex posterior, defining it as $q_{\phi}(\mathbf{h}|\mathbf{z}) = \mathcal{N}_c(\boldsymbol{\mu}, \Delta(\boldsymbol{\sigma}), \Delta(\boldsymbol{\delta}))$. This means that we allow the complex variables of the posterior to have no correlation between one another, but we allow each individual variable to be an ellipse (and even skewed) within its own real/imaginary parts, and with arbitrary (positive) size.

With this formulation, we can derive the KL-Divergence. Because all the complex variables are independent of one another (thanks to the diagonal matrices), the total KL-Divergence is just the sum of the divergences of each individual latent variable $h_j$. We can compute the divergence for each $h_j$ by treating its prior and posterior as bivariate (2D) real Gaussians. The general KL-Divergence formula between a posterior $q = \mathcal{N}(\boldsymbol{\mu}_q, \mathbf{\Sigma}_q)$ and a prior $p = \mathcal{N}(\boldsymbol{\mu}_p, \mathbf{\Sigma}_p)$ for a $k$-dimensional distribution is:

$$D_{KL}(q || p) = \frac{1}{2} \left[ \text{tr}(\mathbf{\Sigma}_{p}^{-1} \mathbf{\Sigma}_{q}) + (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)^T \mathbf{\Sigma}_{p}^{-1} (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p) - k + \ln\left(\frac{|\mathbf{\Sigma}_{p}|}{|\mathbf{\Sigma}_{q}|}\right) \right]$$

Now we plug in the values for a single complex variable $h_j$ (where $k=2$ dimensions, real and imaginary):

- The prior ($p$) is a standard complex normal, meaning the total variance is 1, split equally between the real and imaginary parts. So, $\boldsymbol{\mu}_p = \mathbf{0}$ and $\mathbf{\Sigma}_p = \frac{1}{2}\mathbf{I}$. This makes its inverse $\mathbf{\Sigma}_p^{-1} = 2\mathbf{I}$ and its determinant is $|\mathbf{\Sigma}_p| = \frac{1}{4}$.

- The posterior ($q$) has mean $\boldsymbol{\mu}_q = [\text{Re}(\mu_j), \text{Im}(\mu_j)]^T$ and, from our previous section, a $2 \times 2$ covariance matrix given by $\mathbf{\Sigma}_q = \mathbf{S}_j = \begin{bmatrix} \frac{\sigma_j + \text{Re}(\delta_j)}{2} & \frac{\text{Im}(\delta_j)}{2} \\ \frac{\text{Im}(\delta_j)}{2} & \frac{\sigma_j - \text{Re}(\delta_j)}{2} \end{bmatrix}$. Its determinant is $|\mathbf{\Sigma}_q| = \frac{\sigma_j^2 - |\delta_j|^2}{4}$.

Now, we evaluate each term inside the KL-Divergence brackets:

1.  $\text{tr}(2\mathbf{I} \cdot \mathbf{\Sigma}_q) = 2 \left( \frac{\sigma_j + \text{Re}(\delta_j)}{2} + \frac{\sigma_j - \text{Re}(\delta_j)}{2} \right) = 2\sigma_j$

2.  $\boldsymbol{\mu}_q^T (2\mathbf{I}) \boldsymbol{\mu}_q = 2 (\text{Re}(\mu_j)^2 + \text{Im}(\mu_j)^2) = 2|\mu_j|^2$

3.  $- k = -2$

4.  $\ln\left(\frac{1/4}{(\sigma_j^2 - |\delta_j|^2)/4}\right) = \ln\left(\frac{1}{\sigma_j^2 - |\delta_j|^2}\right) = -\ln(\sigma_j^2 - |\delta_j|^2)$

Putting it all together and multiplying by the $\frac{1}{2}$ at the front gives the divergence for a single variable $h_j$:

$$D_{KL}(q_j || p_j) = \frac{1}{2} \left[ 2\sigma_j + 2|\mu_j|^2 - 2 - \ln(\sigma_j^2 - |\delta_j|^2) \right] = $$
$$= |\mu_j|^2 + \sigma_j - 1 - \frac{1}{2}\log(\sigma_j^2 - |\delta_j|^2)$$

To get the total KL-Divergence for the entire latent space, we simply sum this over all dimensions, which, in vector notation, simplifies to:

$$D_{KL}(q_{\phi}(\mathbf{h}|\mathbf{z}) || p(\mathbf{h})) = \boldsymbol{\mu}^H\boldsymbol{\mu} + \left|\left| \boldsymbol{\sigma} - \mathbf{1} - \frac{1}{2}\log(\boldsymbol{\sigma}^2 - |\boldsymbol{\delta}|^2) \right|\right|_1$$

<!-- ### The Reparameterization Trick & Cholesky Decomposition

To backpropagate through this complex distribution, we must use the reparameterization trick. We sample a standard real noise vector $\boldsymbol{\epsilon} = [\epsilon_r, \epsilon_i]^T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and stretch it using a lower-triangular transformation matrix $\mathbf{L}$ such that $\mathbf{L}\mathbf{L}^T = \mathbf{S}$.

**Finding L (Cholesky Decomposition):**
Given our covariance matrix $\mathbf{S}$:
$$ \mathbf{L} = \begin{bmatrix} l_{11} & 0 \\ l_{21} & l_{22} \end{bmatrix} \implies \mathbf{L}\mathbf{L}^T = \begin{bmatrix} l_{11}^2 & l_{11}l_{21} \\ l_{11}l_{21} & l_{21}^2 + l_{22}^2 \end{bmatrix} = \begin{bmatrix} \frac{\sigma + \delta_r}{2} & \frac{\delta_i}{2} \\ \frac{\delta_i}{2} & \frac{\sigma - \delta_r}{2} \end{bmatrix} $$

Solving system of equations:
1.  $l_{11} = \sqrt{\frac{\sigma + \delta_r}{2}}$
2.  $l_{21} = \frac{\delta_i / 2}{l_{11}} = \frac{\delta_i}{\sqrt{2(\sigma + \delta_r)}}$
3.  $l_{22} = \sqrt{\frac{\sigma - \delta_r}{2} - l_{21}^2} = \sqrt{\frac{\sigma^2 - \delta_r^2 - \delta_i^2}{2(\sigma + \delta_r)}} = \frac{\sqrt{\sigma^2 - |\delta|^2}}{\sqrt{2(\sigma + \delta_r)}}$

Applying this matrix to our noise $\boldsymbol{\epsilon}$ maps perfectly to Nakashika's complex multipliers, $k_r$ and $k_i$, combining the real and imaginary shifts into a single complex multiplication:
$$ k_r = l_{11} + i l_{21} = \frac{\sigma + \delta_r + i\delta_i}{\sqrt{2(\sigma + \delta_r)}} = \frac{\sigma + \delta}{\sqrt{2(\sigma + \text{Re}(\delta))}} $$
$$ k_i = i l_{22} = \frac{i \sqrt{\sigma^2 - |\delta|^2}}{\sqrt{2(\sigma + \text{Re}(\delta))}} $$

As noted in the code, for the matrix to be positive definite (and the math to exist), the determinant must be strictly greater than zero:
$$ \text{Det}(\mathbf{S}) = \frac{\sigma^2 - |\delta|^2}{4} > 0 \implies \sigma^2 > |\delta|^2 \implies \sigma > |\delta| $$

### Reconstruction Loss

After passing through the complex latent space, we require a maximum likelihood estimation of our data $\mathbf{z}$ given the latent variables $\mathbf{h}$. Assuming our output is a complex normal $\mathcal{N}_c(\boldsymbol{\mu}, \mathbf{I}, \mathbf{0})$, this elegantly simplifies in exact parallel to a standard VAE:
$$ \mathbb{E}_{q_\phi(\mathbf{h}|\mathbf{z})}[\log p_\theta(\mathbf{z}|\mathbf{h})] \approx -||\mathbf{z} - \boldsymbol{\mu}||_2^2 $$
Which is, beautifully, just the Mean Squared Error (MSE).

With the mathematical intuition laid out, the implementation becomes a strict translation of these complex constraints. Below, we provide the architecture (generalizable, though complex networks are notoriously sensitive to scale changes) and the vastly improved phase-aware reconstruction results. -->