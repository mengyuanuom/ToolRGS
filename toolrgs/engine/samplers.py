"""Distributed samplers used by the Ascend runner."""


class DistributedEvalSampler:
    """Shard evaluation indices without padding or duplicate samples."""

    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        if self.num_replicas <= 0:
            raise ValueError("num_replicas must be positive")
        if not 0 <= self.rank < self.num_replicas:
            raise ValueError(f"rank must be in [0, {self.num_replicas}), got {self.rank}")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.num_replicas))

    def __len__(self):
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch):
        del epoch
