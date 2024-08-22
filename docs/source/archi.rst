===================
Model Architectures
===================
Flemme supports various architectures for segmentation, reconstruction and generation tasks, but they are all built from a base encoder-decoder architecture.

Base Architectures
==================

Base Architectures contains a encoder and decoder, which are defined in a same code block. The number of input channels of a base model is determined by the encoder, and the number of output channels is determined by the decoder. But, it contains several optional components for context embedding.

Segmentation