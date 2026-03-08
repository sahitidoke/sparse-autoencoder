note: data, config and model.py were all directly used from https://github.com/shehper/sparse-dictionary-learning. i only modified hooked_model.py to collect residual activations instead of mlp activations and coded my own SAE (sae.py).

this is a replication of anthropic's paper on monosemanticity https://transformer-circuits.pub/2023/monosemantic-features#related-work i played around with residual stream activations instead of mlp activations to see if anything cool shows up

the project has three main components. one is training the nanoGPT model, second is the hooked model which uses PyTorch's forward hook system to store internal activations during a forward pass and finally the SAE which is trained on these activations

