import numpy as np
import torch
from models import canonize_batch

T = torch.tensor([[[2, 3], [1, 5], [1, 3]],  # Batch 1
                  [[4, 2], [3, 6], [3, 1]]],  # Batch 2
                 dtype=torch.float32)

sorted_T = canonize_batch(T)
print(sorted_T)

# import torch
#
# Example tensor T of shape (b, n, d)
# (b = 2, n = 3, d = 2)
# T = torch.tensor([[[2, 3], [1, 5], [1, 3]],  # Batch 1
#                   [[4, 2], [3, 6], [3, 1]]],  # Batch 2
#                  dtype=torch.float32)
#
# # Start sorting from the least significant dimension (d-1) and work backwards to the most significant (0)
# sorted_T = T.clone()
# for dim in reversed(range(T.shape[2])):  # d=2, so we sort on dim=1 first, then dim=0
#     _, indices = torch.sort(sorted_T[..., dim], dim=1)
#     sorted_T = torch.gather(sorted_T, 1, indices.unsqueeze(-1).expand_as(sorted_T))
# #
# # print(sorted_T)
