from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from vllm import _custom_ops as ops


@dataclass
class KVCacheSlab:
    tensor: torch.Tensor
    start_block: int

    @property
    def num_blocks(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def end_block(self) -> int:
        return self.start_block + self.num_blocks


class AppendOnlyKVCache:

    def __init__(self, slabs: Sequence[KVCacheSlab]) -> None:
        if not slabs:
            raise ValueError("AppendOnlyKVCache requires at least one slab.")
        self.slabs: List[KVCacheSlab] = list(slabs)

    @classmethod
    def from_base_tensor(cls, tensor: torch.Tensor) -> "AppendOnlyKVCache":
        return cls([KVCacheSlab(tensor=tensor, start_block=0)])

    def append_slab(self, tensor: torch.Tensor) -> None:
        self.slabs.append(
            KVCacheSlab(tensor=tensor, start_block=self.total_num_blocks))

    @property
    def total_num_blocks(self) -> int:
        last = self.slabs[-1]
        return last.end_block

    @property
    def dtype(self) -> torch.dtype:
        return self.slabs[0].tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.slabs[0].tensor.device

    @property
    def layout_shape(self) -> torch.Size:
        base = self.slabs[0].tensor.shape
        return torch.Size((base[0], self.total_num_blocks, *base[2:]))

    def numel(self) -> int:
        return sum(slab.tensor.numel() for slab in self.slabs)


def is_append_only_kv_cache(kv_cache: object) -> bool:
    return isinstance(kv_cache, AppendOnlyKVCache)


def write_to_append_only_kv_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: AppendOnlyKVCache,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> None:
    if slot_mapping.numel() == 0:
        return

    flat_slot_mapping = slot_mapping.flatten()
    block_size = kv_cache.slabs[0].tensor.shape[2]
    valid_mask = flat_slot_mapping >= 0
    if not torch.any(valid_mask):
        return

    for slab in kv_cache.slabs:
        slab_start_slot = slab.start_block * block_size
        slab_end_slot = slab.end_block * block_size
        slab_mask = valid_mask & (flat_slot_mapping >= slab_start_slot) & (
            flat_slot_mapping < slab_end_slot)
        if not torch.any(slab_mask):
            continue
        local_slot_mapping = flat_slot_mapping[slab_mask] - slab_start_slot
        ops.reshape_and_cache_flash(
            key[slab_mask],
            value[slab_mask],
            slab.tensor[0],
            slab.tensor[1],
            local_slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )


def _seq_lens_to_list(seq_lens: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(seq_lens, torch.Tensor):
        return [int(x) for x in seq_lens.detach().cpu().tolist()]
    return [int(x) for x in seq_lens]


def _collect_used_block_ids(
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor | Sequence[int],
    block_size: int,
) -> list[int]:
    seq_lens_list = _seq_lens_to_list(seq_lens)
    block_tables_cpu = block_tables.detach().cpu()
    used_block_ids: list[int] = []
    for row, seq_len in enumerate(seq_lens_list):
        num_blocks = (seq_len + block_size - 1) // block_size
        if num_blocks <= 0:
            continue
        row_ids = block_tables_cpu[row, :num_blocks].tolist()
        for block_id in row_ids:
            block_id = int(block_id)
            if block_id >= 0:
                used_block_ids.append(block_id)
    return sorted(set(used_block_ids))


def gather_flash_attn_kv_cache(
    kv_cache: AppendOnlyKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor | Sequence[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    used_block_ids = _collect_used_block_ids(block_tables, seq_lens, block_size)
    if not used_block_ids:
        empty_shape = (2, 0, *kv_cache.slabs[0].tensor.shape[2:])
        dense_kv_cache = torch.empty(
            empty_shape,
            dtype=kv_cache.dtype,
            device=kv_cache.device,
        )
        return dense_kv_cache[0], dense_kv_cache[1], block_tables

    dense_kv_cache = torch.empty(
        (2, len(used_block_ids), *kv_cache.slabs[0].tensor.shape[2:]),
        dtype=kv_cache.dtype,
        device=kv_cache.device,
    )

    block_id_to_dense = {block_id: idx for idx, block_id in enumerate(used_block_ids)}
    dense_positions = torch.tensor(
        [block_id_to_dense[block_id] for block_id in used_block_ids],
        dtype=torch.long,
        device=kv_cache.device,
    )

    for slab in kv_cache.slabs:
        slab_block_ids = [
            block_id for block_id in used_block_ids
            if slab.start_block <= block_id < slab.end_block
        ]
        if not slab_block_ids:
            continue
        src_indices = torch.tensor(
            [block_id - slab.start_block for block_id in slab_block_ids],
            dtype=torch.long,
            device=kv_cache.device,
        )
        dst_indices = torch.tensor(
            [block_id_to_dense[block_id] for block_id in slab_block_ids],
            dtype=torch.long,
            device=kv_cache.device,
        )
        dense_kv_cache.index_copy_(1, dst_indices, slab.tensor.index_select(1, src_indices))

    remapped_block_tables = block_tables.clone()
    seq_lens_list = _seq_lens_to_list(seq_lens)
    remapped_cpu = remapped_block_tables.detach().cpu()
    for row, seq_len in enumerate(seq_lens_list):
        num_blocks = (seq_len + block_size - 1) // block_size
        for col in range(num_blocks):
            global_block_id = int(remapped_cpu[row, col].item())
            remapped_cpu[row, col] = block_id_to_dense[global_block_id]
    remapped_block_tables.copy_(remapped_cpu.to(remapped_block_tables.device))
    return dense_kv_cache[0], dense_kv_cache[1], remapped_block_tables
