import torch

def flatten_fig(input):
    result = []
    batch_size, channel, m, n = input.shape
    input = input.view(batch_size, channel, m*n)
    for batch_index in range(batch_size):
        result_this_batch = []
        for element_index in range(m*n):
            feature_element = torch.stack([input[batch_index][channel_index][element_index]
                                           for channel_index in range(channel)])
            result_this_batch.append(feature_element)
        result.append(torch.stack(result_this_batch))

    return torch.stack(result)

def flatten_encoder(input):
    result = []
    batch_size, channel, m = input.shape
    for batch_index in range(batch_size):
        result_this_batch = []
        for element_index in range(m):
            feature_element = torch.stack([input[batch_index][channel_index][element_index]
                                           for channel_index in range(channel)])
            result_this_batch.append(feature_element)
        result.append(torch.stack(result_this_batch))

    return torch.stack(result)
