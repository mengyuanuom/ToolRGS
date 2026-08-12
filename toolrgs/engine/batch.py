"""Distributed batch-size helpers."""


def per_process_batch_size(global_batch_size, world_size, field_name):
    """Convert a configured global batch size to one process's batch size."""
    global_batch_size = int(global_batch_size)
    world_size = int(world_size)
    if global_batch_size <= 0:
        raise ValueError(f"TRAIN.{field_name} must be positive, got {global_batch_size}.")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}.")
    if global_batch_size % world_size:
        raise ValueError(
            f"TRAIN.{field_name} is global and must be divisible by world size: "
            f"{global_batch_size} % {world_size} != 0."
        )
    return global_batch_size // world_size
