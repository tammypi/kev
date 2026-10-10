"""Inference across CUDA GPUs, with each decoder layer and its cache on one device.

The checkpoint loader builds the model on CPU, including its adapter, before calling shard(). This avoids a transient
full copy on GPU 0. Accelerate moves activations between layers and returns hidden states to GPU 0 for the pointer head;
the DynamicCache stays distributed. No CPU/disk offload, CUDA graphs, or Kev fused rewrites on this path.
"""
import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory


def validate(device, opts):
    """Reject incompatible settings before loading any weights."""
    if opts.device_map != "balanced":
        raise ValueError("device_map must be 'balanced' or None (KEV_DEVICE_MAP=balanced / --device-map balanced)")
    if not str(device).startswith("cuda") or opts.backend == "mlx":
        raise ValueError("balanced device_map requires the torch backend on CUDA; pass --device cuda")
    if opts.cuda_graphs or opts.fused:
        raise ValueError("balanced device_map requires KEV_CUDA_GRAPHS=0 and KEV_FUSED=0")
    if torch.cuda.device_count() < 2:
        raise ValueError("balanced device_map needs at least two visible CUDA GPUs; set CUDA_VISIBLE_DEVICES=2,3, for example")


def balanced_device_map(lm, max_memory):
    """Balance weight bytes without splitting a residual block or silently spilling weights to CPU/disk.

    max_memory contains GPU weight budgets, excluding space reserved by the caller for caches and activations.
    The model's own no-split declarations also work through an unmerged PEFT wrapper.
    """
    no_split = lm._no_split_modules
    if not no_split:
        raise ValueError("balanced device_map needs a backbone with declared no-split decoder layers")
    budgets = get_balanced_memory(lm, max_memory=max_memory, no_split_module_classes=no_split)
    placement = infer_auto_device_map(lm, max_memory=budgets, no_split_module_classes=no_split)
    if any(device not in max_memory for device in placement.values()):
        raise ValueError("model weights do not fit the visible GPUs with 20% of free memory reserved for inference; "
                         "free GPU memory or expose more GPUs")
    if len(set(placement.values())) < 2:
        raise ValueError("balanced device_map could not place this model on at least two GPUs without splitting decoder layers")
    return placement


def shard(model):
    """Dispatch a CPU model. The head and request tensors use GPU 0; backbone caches stay with their owning layers."""
    budgets = {i: int(torch.cuda.mem_get_info(i)[0] * 0.8) for i in range(torch.cuda.device_count())}
    placement = balanced_device_map(model.lm, budgets)
    model.lm = dispatch_model(model.lm, device_map=placement, main_device=0,
                              skip_keys=["past_key_values", "cache_params"])
    model.device = "cuda:0"
    model.head.to(model.device)
    return model
