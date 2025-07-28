import torch

import matplotlib.pyplot as plt

def check_tensor(tensor, tensor_name="Tensor", force_analyze=False, min=None, max=None):
    if not force_analyze:
        return True
    """
    Check if the given tensor contains any NaN or Inf values and print a specific message.

    Args:
    tensor (torch.Tensor): The tensor to check.
    tensor_name (str): The name of the tensor to display in messages.
    """
    has_nan = torch.isnan(tensor).any().item()  # Check for any NaNs and convert to Python bool
    has_inf = torch.isinf(tensor).any().item()  # Check for any Infs and convert to Python bool


    good = True
    # Print relevant messages based on the tensor's content
    if has_nan and has_inf:
        print(f"{tensor_name} contains both NaN and Inf values.")
        good = False
    elif has_nan:
        print(f"{tensor_name} contains NaN values.")
        good = False
    elif has_inf:
        print(f"{tensor_name} contains Inf values.")
        good = False



    if min is not None and torch.all(tensor >= min).item() is False:
        good = False
        print(f"{tensor_name} is smaller than {min}")

    if max is not None and torch.all(tensor <= max).item() is False:
        good = False
        print(f"{tensor_name} is bigger than {max}")


    if good and not force_analyze:
        return True

    # Checking for NaNs and Infs
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print("Contains NaN values:", has_nan)
    print("Contains Inf values:", has_inf)

    # Basic statistics
    tensor_min = torch.min(tensor).item()
    tensor_max = torch.max(tensor).item()
    tensor_mean = torch.mean(tensor.float()).item()  # Ensure tensor is float for mean calculation
    print("Minimum value:", tensor_min)
    print("Maximum value:", tensor_max)
    print("Average value:", tensor_mean)

    # Histogram of the tensor values
    tensor_np = tensor.detach().cpu().numpy()  # Convert tensor to NumPy array for histogram
    plt.figure(figsize=(10, 4))
    plt.hist(tensor_np.ravel(), bins=100, color='blue', alpha=0.7)  # Flatten the tensor and plot
    plt.title(f"Histogram of {tensor_name} with shape {tensor.shape}")
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    return good
