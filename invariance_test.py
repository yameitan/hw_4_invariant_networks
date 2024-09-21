import torch


def test_invariance(model, set_size, data_dim):
    x= torch.randn((100, set_size, data_dim))
    output = model(x)
    for i in range(10):
        permuted_indices = torch.randperm(set_size)
        x = x[:, permuted_indices, :]
        perm_output = model(x)
        error = (output - perm_output).abs().sum()
        if error > 1e-5:
            print(f"Failed :(")
            return False
    print(f"Passed :)")
    return True

