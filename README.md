<p align="center">
  <img src="assets/logo.png" alt="BinauralVAE Logo" width="800" height="300"/>
</p>

# BinauralVAE: Spatial Audio Reconstruction

Welcome to **BinauralVAE**. This project explores several Variational Autoencoder (VAE) approaches - including complex-valued variants - to reconstruct spatialized audio. 

Developed alongside [AudioWorldSim](https://github.com/Luizerko/AudioWorldSim), this project serves as a foundation for generating latent state representations for a binaural, audio-based world model. We highly recommend checking out AudioWorldSim for a better understanding of the acoustic data used here. The core objective is to reconstruct binaural audio data corresponding to every action a robot takes in a simulation, effectively translating acoustic information into navigational states.

This project was developed by [Luis Zerkowski](https://luizerko.github.io/) under the supervision of [Luiz Velho](https://lvelho.impa.br/) at [VISGRAF](https://www.visgraf.impa.br/home/index.php), the Vision and Graphics Laboratory at [IMPA](https://impa.br/?lang=en).

<br>

<figure align="center">
  <!-- Replace image and audio paths with your actual repository files -->
  <img src="docs/assets/mel_reconstruction.png" alt="Left and Right Ears Mel Spectrogram Reconstructions" width="800"/>
  <figcaption>
    <b>Figure 1:</b> Mel spectrogram reconstructions for the left and right ears.<br> 
    Listen to the samples: 
    <a href="docs/assets/audio/original_spatial.wav">🔊 Original Spatial Sound</a> | 
    <a href="docs/assets/audio/original_mel_rec.wav">🔊 Original Mel Reconstruction</a> | 
    <a href="docs/assets/audio/reconstructed_mel.wav">🔊 Reconstructed Spectrogram Audio</a>
  </figcaption>
</figure>

---

## Introduction

The primary goal of this repository is to develop a pipeline for binaural audio reconstruction using Variational Autoencoders (including complex-valued variants). This is the first step toward an **audio-based world model**, allowing us to encode binaural audio captured during robotic navigation.

Using [AudioWorldSim](https://github.com/Luizerko/AudioWorldSim), a simulated robot navigates a room equipped with a binaural sensor that captures realistic audio data based on room acoustics and a Head-Related Transfer Function (HRTF). Because the robot captures audio with every action, we can map the direct connection between actions and resulting left/right ear audio. This forms the perfect basis for an audio-based world model that connects states, actions, and the acoustic consequences of those actions.

While visual world models are prevalent, research into realistic spatial audio reconstruction and audio-only world models remains sparse—and open-source implementations are even rarer. This project bridges that gap. We provide a highly flexible, mathematically grounded architecture to explore multiple VAE options for spatial audio reconstruction. 

Unlike images, sound does not suffer from visual occlusion. It offers a complementary, albeit different, sense of perception. Imagine navigating an indoor blackout using only the sound of an emergency siren, or rescuers locating someone in a pitch-black cave by following their voice. We are not advocating against multimodality; rather, we emphasize that sound alone holds essential information for understanding the world and the consequences of actions—an area often overshadowed by vision-first approaches.

We have compiled a comprehensive list of [References](https://github.com/Luizerko/audio-nav/blob/main/REFERENCES.md) covering the literature that guided this project. We highly recommend exploring these works.

---