#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

"""FSDP2 utility functions for model setup and configuration."""

import functools
import itertools
import logging
from collections.abc import Iterable

import torch
import torch.nn as nn
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.utils.fsdp_utils import (
    fsdp2_load_full_state_dict,
    fsdp2_prepare_auto_wrap_policy,
)
from accelerate.utils.other import get_module_children_bottom_up
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from phi_agents.utils.logger import NullLogger


def parse_module_name(name: str) -> tuple[list[str], list[int]]:
    """Parse a module name into parts and extract numeric indices.

    Args:
        name: Module name like 'base_model.model.layers.0.self_attn.q_proj'

    Returns:
        Tuple of (parts, indices) where parts are string components and
        indices are positions of numeric parts in the name
    """
    parts = name.split(".")
    numeric_indices = []
    for i, part in enumerate(parts):
        if part.isdigit():
            numeric_indices.append(i)
    return parts, numeric_indices


def format_numeric_range(numbers: list[int]) -> str:
    """Format a list of numbers into a compact range notation.

    Args:
        numbers: List of integers

    Returns:
        String representation like '[0-2,4]' for mixed consecutive/non-consecutive
    """
    if len(numbers) <= 1:
        return str(numbers[0]) if numbers else ""

    sorted_nums = sorted(set(numbers))

    if len(sorted_nums) == 1:
        return str(sorted_nums[0])

    # Group consecutive numbers using itertools.groupby
    ranges = []
    for _, group in itertools.groupby(enumerate(sorted_nums), key=lambda x: x[1] - x[0]):
        group_nums = [num for _, num in group]
        if len(group_nums) == 1:
            ranges.append(str(group_nums[0]))
        else:
            ranges.append(f"{group_nums[0]}-{group_nums[-1]}")

    return f"[{','.join(ranges)}]"


def can_compress_together(name1: str, name2: str) -> bool:
    """Check if two names differ in exactly one numeric position."""
    parts1, indices1 = parse_module_name(name1)
    parts2, indices2 = parse_module_name(name2)

    # Must have same structure (same length and same numeric positions)
    if len(parts1) != len(parts2) or indices1 != indices2:
        return False

    # Count how many parts differ (should be exactly 1 numeric part)
    return sum(p1 != p2 for p1, p2 in zip(parts1, parts2, strict=True)) == 1


def compress_sequence(names: list[str]) -> str:
    """Compress a sequence of names that differ in one numeric position."""
    if len(names) == 1:
        return names[0]

    # Use first name as template and find the varying position
    template_parts, indices = parse_module_name(names[0])

    # Find which numeric position varies
    varying_values = []
    varying_pos = None

    for pos in indices:
        values = []
        for name in names:
            parts, _ = parse_module_name(name)
            values.append(int(parts[pos]))

        if len(set(values)) > 1:
            varying_pos = pos
            varying_values = values
            break

    if varying_pos is None:
        return names[0]  # Shouldn't happen

    # Create compressed name
    template_parts[varying_pos] = format_numeric_range(varying_values)
    return ".".join(template_parts)


def compress_module_names(names: list[str]) -> list[str]:
    """Compress module names by detecting patterns and using range notation.

    Args:
        names: List of module names to compress

    Returns:
        List of compressed module names
    """
    if len(names) <= 1:
        return names

    # Sort names for easier processing (output order doesn't matter)
    sorted_names = sorted(names, key=lambda v: v[::-1])
    compressed = []
    i = 0

    while i < len(sorted_names):
        # Start a new sequence with current name
        sequence = [sorted_names[i]]
        j = i + 1

        # Look for consecutive names that can be compressed with the current sequence
        while j < len(sorted_names):
            if can_compress_together(sequence[0], sorted_names[j]):
                sequence.append(sorted_names[j])
                j += 1
            else:
                break

        # Compress the sequence if it has more than one name
        if len(sequence) > 1:
            compressed.append(compress_sequence(sequence))
        else:
            compressed.append(sequence[0])

        i = j

    return compressed


def module_has_all_params_with_grad(module: nn.Module) -> bool:
    """Check if all parameters in a module require gradients."""
    all_params = list(module.parameters())
    return len(all_params) > 0 and all(p.requires_grad for p in all_params)


def extract_modules_with_all_grad(
    model: nn.Module, prefix: str | None = None
) -> list[tuple[str, nn.Module]]:
    """Extract modules where all parameters require gradients."""
    res = []
    for n, c in model.named_children():
        this_name = f"{prefix}.{n}" if prefix is not None else n
        if not module_has_all_params_with_grad(c):
            res.extend(extract_modules_with_all_grad(c, this_name))
        else:
            res.append((this_name, c))
    return res


def flatten_module_containers(modules: Iterable[nn.Module]) -> list[nn.Module]:
    """Flatten ModuleList and ModuleDict containers into a flat list."""
    res: list[nn.Module] = []
    for m in modules:
        match m:
            case nn.ModuleList():
                res.extend(flatten_module_containers(m))
            case nn.ModuleDict():
                res.extend(flatten_module_containers(m.values()))
            case _:
                res.append(m)
    return res


def get_auto_wrap_policy_type(fsdp2_plugin: FullyShardedDataParallelPlugin) -> str:
    """Determine the auto wrap policy type from the FSDP2 plugin."""
    if fsdp2_plugin.auto_wrap_policy is transformer_auto_wrap_policy:
        return "transformer"
    elif fsdp2_plugin.auto_wrap_policy is size_based_auto_wrap_policy:
        return "size"
    else:
        raise ValueError(f"Unknown auto_wrap_policy: {fsdp2_plugin.auto_wrap_policy!r}")


def apply_fsdp2_sharding(
    model: nn.Module,
    fsdp2_plugin: FullyShardedDataParallelPlugin,
    auto_wrap_policy_type: str,
) -> None:
    """Apply FSDP2 sharding to the model."""
    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
    }

    accelerate_fsdp2_policy = fsdp2_prepare_auto_wrap_policy(
        fsdp2_plugin, auto_wrap_policy_type, model
    )

    for module in get_module_children_bottom_up(model)[:-1]:
        if accelerate_fsdp2_policy(module):
            # If this is a module where some params require grad and others don't
            # We create two sharding groups. One for all params that require
            # grad and one for all other params.
            if not module_has_all_params_with_grad(module):
                submodules_with_grad = [m for _, m in extract_modules_with_all_grad(module)]
                fully_shard(flatten_module_containers(submodules_with_grad), **fsdp2_kwargs)

            fully_shard(module, **fsdp2_kwargs)

    fully_shard(model, **fsdp2_kwargs)


def convert_trainable_modules_to_fp32(
    model: nn.Module, logger: logging.Logger | NullLogger
) -> None:
    """Convert modules with all trainable parameters to FP32 and validate."""
    modules_with_all_grad = extract_modules_with_all_grad(model)
    for _, m in modules_with_all_grad:
        m.to(dtype=torch.float32)

    # Compress module names for cleaner logging
    module_names = [n for n, _ in modules_with_all_grad]
    compressed_names = compress_module_names(module_names)
    logger.info(f"Converted {compressed_names} to float32")

    # Validate that all trainable parameters are in FP32
    any_failed = False
    for n, p in model.named_parameters():
        if p.requires_grad and not p.dtype == torch.float32:
            logger.error(f"Parameters {n} requires a gradient but has dtype {p.dtype}.")
            any_failed = True

    if any_failed:
        raise RuntimeError(
            "Some parameters that require grad aren't in fp32. "
            "Parameters that require a gradient must be in a module where "
            "all params require gradient."
        )


def setup_fsdp2_model(
    model: nn.Module, accelerator: Accelerator, logger: logging.Logger | NullLogger
) -> nn.Module:
    """Complete FSDP2 setup for a model including sharding, checkpointing, and FP32 conversion.

    Args:
        model: The model to set up with FSDP2
        accelerator: The accelerator instance containing FSDP2 plugin
        logger: Logger for status messages

    Returns:
        The model after FSDP2 setup
    """
    original_sd = model.state_dict()

    fsdp2_plugin = accelerator.state.fsdp_plugin
    assert isinstance(fsdp2_plugin, FullyShardedDataParallelPlugin)

    # Get auto wrap policy type
    auto_wrap_policy_type = get_auto_wrap_policy_type(fsdp2_plugin)
    fsdp2_plugin.set_auto_wrap_policy(model)

    # Apply activation checkpointing if enabled
    if fsdp2_plugin.activation_checkpointing:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            auto_wrap_policy=fsdp2_plugin.auto_wrap_policy,  # type: ignore
        )

    # Apply FSDP2 sharding
    apply_fsdp2_sharding(model, fsdp2_plugin, auto_wrap_policy_type)

    # Load full state dict if CPU RAM efficient loading is enabled
    if fsdp2_plugin.cpu_ram_efficient_loading:
        # If `cpu_ram_efficient_loading` is enabled, only rank 0 loads the weights
        # Other ranks have an empty model on `meta` device, so we need to distribute the weights properly
        fsdp2_load_full_state_dict(accelerator, model, original_sd)

    # Convert trainable modules to FP32
    convert_trainable_modules_to_fp32(model, logger)

    return model


def setup_mixed_precision_policy(accelerator: Accelerator):
    from accelerate import FullyShardedDataParallelPlugin
    from torch.distributed.fsdp import MixedPrecisionPolicy

    if not accelerator.is_fsdp2:
        return

    fsdp2_plugin = accelerator.state.fsdp_plugin
    assert isinstance(fsdp2_plugin, FullyShardedDataParallelPlugin)
    mixed_precision_mapping = {
        "fp8": torch.bfloat16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if accelerator.mixed_precision != "no":
        param_dtype = mixed_precision_mapping[accelerator.mixed_precision]
    else:
        param_dtype = None

    fsdp2_plugin.mixed_precision_policy = MixedPrecisionPolicy(param_dtype, torch.float32, None)
