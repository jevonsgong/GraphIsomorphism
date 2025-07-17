import numpy as np
import torch
from torch import nn
import xxhash, struct
from collections import defaultdict


buckets = defaultdict(list)
hashes = [0,0,1,1,2,2]
for v, h in enumerate(hashes):
    buckets[h].append(v)
print(buckets)



