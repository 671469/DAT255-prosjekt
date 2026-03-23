import torch

#dataset splitter
def train_val_split(data, split_ratio=0.9):

    n = int(split_ratio * len(data))

    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data

#tilfeldig batch generator
def get_batch(data, block_size, batch_size):

    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y
