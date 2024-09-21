import torch
import torch.nn as nn
import itertools
import math


def canonize_batch(batch):
    sorted_batch = batch.clone()
    for dim in reversed(range(batch.shape[2])):  # d=2, so we sort on dim=1 first, then dim=0
        _, indices = torch.sort(sorted_batch[..., dim], dim=1)
        sorted_batch = torch.gather(sorted_batch, 1, indices.unsqueeze(-1).expand_as(sorted_batch))
    return sorted_batch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net(x)

class EquivariantLinearLayer(nn.Module):

    def __init__(self, input_dim, output_dim,  sigma=0.01):
        super().__init__()
        self.w1 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.w2 = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.bias = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters(sigma=sigma)

    def reset_parameters(self, sigma=0.01):
        nn.init.normal_(self.w1, mean=0, std=sigma)
        nn.init.normal_(self.w2, mean=0, std=sigma)
        nn.init.normal_(self.bias, mean=0, std=0)

    def forward(self, x):
        x_sum = torch.sum(x, dim=1, keepdim=True)
        x_broadcast_sum = x_sum.expand(-1, 1, -1).repeat(1, x.size(1), 1)
        output = torch.matmul(x, self.w1) + torch.matmul(x_broadcast_sum, self.w2)+ self.bias
        return output


class CanonizationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        x = canonize_batch(x)
        return self.mlp(x)


class SymmetrizationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, x):
        set_size = x.shape[1]
        outputs = []
        for perm in itertools.permutations(range(set_size)):
            outputs.append(self.mlp(x[:, perm, :]))
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(dim=0)


class SampledSymmetrizationNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, sample_factor = 20):
        super().__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim, num_layers)
        self.sample_factor = sample_factor

    def forward(self, x):
        set_size = x.shape[1]
        outputs = []
        for i in range(set_size//self.sample_factor):
            permuted_indices = torch.randperm(set_size)
            outputs.append(self.mlp(x[:, permuted_indices, :]))
        outputs = torch.stack(outputs)
        return outputs.mean(dim=0)


class InvariantLinearNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([EquivariantLinearLayer(input_dim, hidden_dim)])
        for i in range(1, num_layers):
            self.layers.append(EquivariantLinearLayer(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim, bias=True))  # output layer

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = layer(x).relu()
        x_out = torch.sum(x, dim=1)
        x_out = self.layers[-1](x_out)
        return x_out


def get_model(model_type, input_dim, hidden_dim, output_dim, num_layers, device):
    if model_type == 'CanonizationNetwrok':
        model = CanonizationNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                                    num_layers=num_layers)

    elif model_type == 'SymmetrizationNetwork':
        model = SymmetrizationNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                                      num_layers=num_layers)

    elif model_type == 'SampledSymmetrizationNetwork':
        model = SampledSymmetrizationNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                                      num_layers=num_layers)

    elif model_type == 'InvariantLinearlNetwork':
        model = InvariantLinearNetwork(input_dim=input_dim, hidden_dim = hidden_dim, output_dim=output_dim,
                                       num_layers=num_layers)

    elif model_type == 'AugmentationNetwork':
        model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim,num_layers=num_layers)
    else:
        raise ValueError(f'type not implemented')
    return model.to(device)

