import torch


def test_invariance(model):
    x= torch.randn((100, 10, 5))
    output = model(x)
    for i in range(10):
        permuted_indices = torch.randperm(10)
        x = x[:, permuted_indices, :]
        perm_output = model(x)
        error = (output - perm_output).abs().sum()
        if error > 1e-5:
            print(f"Failed :(")
            return False
    print(f"Passed :)")
    return True

