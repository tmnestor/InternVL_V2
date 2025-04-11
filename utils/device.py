"""
Utilities for device management.
"""
from typing import Optional, Tuple, Union

import torch


def get_device(cuda_id: Optional[int] = None) -> torch.device:
    """
    Get the appropriate device based on availability.
    
    Args:
        cuda_id: Optional specific CUDA device ID to use
        
    Returns:
        torch.device object for CPU or GPU
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    if cuda_id is not None:
        if cuda_id >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {cuda_id} not available. "
                             f"Only {torch.cuda.device_count()} devices found.")
        return torch.device(f"cuda:{cuda_id}")
    
    return torch.device("cuda")


def to_device(data: Union[torch.Tensor, Tuple, list, dict], device: torch.device) -> Union[torch.Tensor, Tuple, list, dict]:
    """
    Move tensors, or collections of tensors to the specified device.
    
    Args:
        data: Input data (tensor, tuple, list, or dict)
        device: Target device
        
    Returns:
        Data on the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (tuple, list)):
        return [to_device(x, device) for x in data]
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data


def get_device_properties() -> dict:
    """
    Get information about available GPU devices.
    
    Returns:
        Dictionary with device properties
    """
    properties = {
        "device_count": torch.cuda.device_count(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        properties["cuda_version"] = torch.version.cuda
        properties["current_device"] = torch.cuda.current_device()
        properties["devices"] = []
        
        for i in range(torch.cuda.device_count()):
            device_props = {
                "name": torch.cuda.get_device_name(i),
                "capability": torch.cuda.get_device_capability(i),
                "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                "memory_available": torch.cuda.memory_reserved(i) / (1024**3),  # GB
            }
            properties["devices"].append(device_props)
    else:
        properties["cpu_info"] = "No CUDA-compatible GPU found"
    
    return properties


def print_gpu_info():
    """
    Print detailed information about available GPUs.
    """
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("No CUDA-compatible GPU found")