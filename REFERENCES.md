# References

This is a section dedicated to the dozens of references that guided this work. We don't necessarily recommend reading them all, but they were all material we consumed - either by reading entirely or parts of it - to get us to where we are right now. We split them in sections for better guidance, although various references could be put in multiple categories. And we don't even go far enough to say that we are putting them necessarily in their most important category, just in the category that they were most useful to us.

## World Models Understanding and Approaches

1. [World Models](https://arxiv.org/pdf/1803.10122) by David Ha et al. -> Introducing the idea of world models.
2. [From Words to Worlds: Spatial Intelligence is AI's Next Frontier](https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence) by Li Fei-Fei -> Discussing the idea of world models and spatial intelligence.
3. [PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation](https://arxiv.org/pdf/2601.03782) by Wenlong Huang et al. -> World models pointcloud approach for robotics.
4. [Mastering Diverse Domains through World Models](https://arxiv.org/pdf/2301.04104) by Danijar Hafner et al. -> Generalization and modernization of the typical world model pipeline for multiple scenarios.
5. [V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning](https://arxiv.org/pdf/2506.09985) by Mahmoud Assran et al. -> Self supervised JEPA approach for world models.
6. [Genie: Generative Interactive Environments](https://arxiv.org/html/2402.15391v1) by Jake Bruce et al. -> Using world models to create interactive worlds from "static" input.
7. [WoVR: World Models as Reliable Simulators for Post-Training VLA Policies with RL](https://arxiv.org/html/2602.13977v1) by Zhennan Jiang et al. -> World models approach for VLA models stabilization.
8. [Video2Game: Real-time, Interactive, Realistic and Browser-Compatible Environment from a Single Video](https://arxiv.org/pdf/2404.09833) by Hongchi Xia et al. -> 3D generation and introduction of physics and interaction from video.
9. [Navigation World Models](https://arxiv.org/pdf/2412.03572) by Amir Bar et al. -> World models for robot navigation.
10. [Audio-Visual World Models: Towards Multisensory Imagination in Sight and Sound](https://arxiv.org/pdf/2512.00883) by Jiahua Wang et al. -> Audio-visual world models.
11. [AV-RIR: Audio-Visual Room Impulse Response Estimation](https://arxiv.org/pdf/2312.00834) by Anton Ratnarajah et al. -> Audio-visual room impulse response estimation (basically the name indeed hehe).

## Tools for World Understanding and Simulation

12. [SAM 3D: 3Dfy Anything in Images](https://arxiv.org/pdf/2511.16624) by SAM 3D Team at Meta -> From images to segmentation and 3D generation.
13. [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) by OpenAI -> Video generation as a first step into the realm of world models.
14. [ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation](https://arxiv.org/pdf/2007.04954) by Chuang Gan et al. -> Interactive and multimodal physical simulation (including audio).
15. [Attend Before Attention: Efficient and Scalable Video Understanding via Autoregressive Gazing](https://arxiv.org/pdf/2603.12254) by Baifeng Shi et al. -> Processing video for specific attention in regions of frames for better image understanding.

## Audio Processing and Simulation

16. [Blog post on audio deep learning](https://medium.com/data-science/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5) by Ketan Doshi -> Introduction to audio deep learning.
17. [SoundSpaces](https://soundspaces.org/) and [SoundSpaces 2.0](https://arxiv.org/pdf/2206.08312) by Changan Chen et al. -> Panoramic images + binaural audio dataset, and realistic binaural audio simulation on top of [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) for reinforcement learning training using audio.
18. [Efficient Encoding and Decoding of Binaural Sound with Resonance Audio](https://www.tara.tcd.ie/tara8/server/api/core/bitstreams/6f6e697e-aded-4aa3-89b4-8934fcb5ef85/content) by Marcin Gorzel et al. -> SDK for ambisonic sound processing and adaptation for binaural sound encoding and decoding.
19. [Meta XR Audio SDK](https://developers.meta.com/horizon/documentation/unity/meta-xr-audio-sdk-unity/) by Meta -> SDK for spatial audio processing in expanded reality scenarios and great introduction to audio spatialization.
20. [Unity Documentation on Audio](https://docs.unity3d.com/6000.5/Documentation/Manual/AudioOverview.html) by Unity Technologies -> Surprisingly insightful information and introduction to audio spatialization as well.
21. [Spectrogram Features for Audio and Speech Analysis](https://www.mdpi.com/2076-3417/16/2/572) by Ian McLoughlin et al. -> Very good survey on audio processing, particularly using spectrogram features.
22. [Complex-Valued Variational Autoencoder: A Novel Deep Generative Model for Direct Representation of Complex Spectra](https://www.isca-archive.org/interspeech_2020/nakashika20_interspeech.pdf) by Toru Nakashika -> Amazing work introducing a complex variant to the traditional VAE.
23. [A Survey of Deep Learning for Complex Speech Spectrograms](https://arxiv.org/html/2505.08694v1) by Yuying Xie et al. -> Survey on neural methods for complex audio data processing.
24. [Binaural Sound Event Localization and Detection based on HRTF Cues for Humanoid Robots](https://arxiv.org/pdf/2507.20530) by Gyeong-Tae Lee et al. -> Where we first saw convolutional recurrent networks for audio processing.
25. [BINAURAL SPEECH ENHANCEMENT USING COMPLEX CONVOLUTIONAL RECURRENT NETWORKS](https://uol.de/f/6/dept/mediphysik/ag/sigproc/download/papers/SP2023_11.pdf) by Vikas Tokala et al. -> Introduction to complex convolutional recurrent networks for audio processing.
26. [BINAURAL SPEECH ENHANCEMENT USING DEEP COMPLEX CONVOLUTIONAL TRANSFORMER NETWORKS](https://arxiv.org/html/2403.05393v1) by Vikas Tokala et al. -> An adaptation of the previous paper approach, from recurrent networks to transformers.
27. [PHASE-AWARE SPEECH ENHANCEMENT WITH DEEP COMPLEX U-NET](https://openreview.net/pdf?id=SkeRTsAcYm) by Hyeong-Seok Choi et al. -> Complex encoder-decoder approach for audio processing.
28. [Neural Discrete Representation Learning](https://avdnoord.github.io/homepage/vqvae/) and [Tutorial](https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb) by Aaron van den Oord et al. -> Usage of VQ-VAE and discrete latents for audio reconstruction.
29. [AST: Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) by Yuan Gong et al. -> Transformer-based audio encoder, but trained for classification.
30. [Wav2vec 2.0: Learning the structure of speech from raw audio](https://ai.meta.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/) by Meta -> Another audio encoder, but trained for speech structure.
31. [Semantic Audio-Visual Navigation in Continuous Environments](https://arxiv.org/pdf/2603.19660) by Yichen Zeng et al. -> We actually only used this one to check out their method for audio encoding and perhaps have some future ideas for audio-based navigation for a world model.

## Other Interesting Work Involving Both Audio and World Models

32. [SONOWORLD: From One Image to a 3D Audio-Visual Scene](https://arxiv.org/pdf/2603.28757) by Derong Jin et al.
33. [Few-shot Acoustic Synthesis with Multimodal Flow Matching](https://amandinebtto.github.io/FLAC/) by Amandine Brunetto.
34. [Audio-Omni: Extending Multi-modal Understanding to Versatile Audio Generation and Editing](https://zeyuet.github.io/Audio-Omni/) by Zeyue Tian et al.