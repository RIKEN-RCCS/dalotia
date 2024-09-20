#!/usr/bin/env python3

import torch
from safetensors.torch import save_file
# cf. https://huggingface.co/docs/safetensors/en/index

embedding = torch.arange(60., dtype=torch.float64)
embedding = embedding.reshape(3,4,5)
embedding_firstchanged = torch.permute(embedding, (1, 0, 2))
print(embedding)
print(embedding_firstchanged)

tensors = {
    "embedding": embedding,
    "embedding_firstchanged": embedding_firstchanged.contiguous(),
    "attention": torch.zeros((2, 3))
}
save_file(tensors, "model.safetensors")