# SPDX-License-Identifier: Apache-2.0
"""CacheEngine class for managing the KV cache."""
from typing import List
import time
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)
import torch.nn.functional as F
import sys
import triton
import triton.language as tl
logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        print(f"cache engine ! self.num_gpu_blocks: {self.num_gpu_blocks}")
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free,
                                             use_mla=model_config.use_mla)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []

        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            layer_kv_cache = torch.zeros(kv_cache_shape,
                                         dtype=self.dtype,
                                         pin_memory=pin_memory,
                                         device=device)

            # view back to (TOTAL_PAGES, PAGE_SIZE, entry_shape...) for cases
            # when entry_shape is higher than 1D
            kv_cache.append(layer_kv_cache)
        return kv_cache


    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry = num_heads * head_size

        # For MLA there is no value cache, since the latent vector
        # is joint keys and values.
        value_cache_entry = key_cache_entry if not model_config.use_mla else 0
        total = num_attention_layers * cache_config.block_size * \
            (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    def migrate_blocks_inplace_with_triton(self, cache_tensor, block_migration_map, block_size):
        """使用 Triton 加速的原地 KV 缓存块迁移方法
        
        Args:
            cache_tensor: 缓存张量
            block_migration_map: 块迁移映射 {old_id: new_id}
            block_size: 每个块的大小
        """
        # 需要先解决循环依赖问题
        # 按照拓扑排序的方式处理映射，避免数据覆盖

        processed_map = self._resolve_migration_dependencies(block_migration_map)
        
        # 转换映射为张量
        old_ids = torch.tensor(list(processed_map.keys()), 
                            dtype=torch.int32, 
                            device=self.device_config.device_type)
        new_ids = torch.tensor(list(processed_map.values()), 
                            dtype=torch.int32, 
                            device=self.device_config.device_type)
        
        # 获取头部大小和数量
        num_heads = self.num_kv_heads
        head_size = self.head_size
        
        # 计算参数
        num_mappings = len(processed_map)
        block_elements = block_size * num_heads * head_size
        
        # 优化的块大小
        BLOCK_SIZE = 128  # 可根据实际硬件调整
        
        # 启动 Triton kernel
        grid = (num_mappings, (block_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
        kv_cache_inplace_migration_kernel[grid](
            cache_tensor,
            old_ids, new_ids,
            num_mappings,
            block_size, num_heads, head_size,
            BLOCK_SIZE
        )

    def decrease_gpu_blocks(self, decrease_num_blocks: int, block_migration_map=None) -> None:
        """通过迁移方式减少GPU块的数量，按照迁移映射移动KV缓存值
        
        Args:
            decrease_num_blocks: 要减少的块数量
            block_migration_map: 从block_manager获得的块迁移映射 {old_id: new_id}
        """
        time_start = time.time()
        # PyTorch回退版本
        # 基本检查
        decrease_num_blocks = min(decrease_num_blocks, self.num_gpu_blocks)
        if decrease_num_blocks <= 0:
            return
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"before decrease_gpu_blocks 显存使用情况: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")   
        new_num_blocks = self.num_gpu_blocks - decrease_num_blocks
        print(f"decrease_num_blocks: {decrease_num_blocks}, new_num_blocks: {new_num_blocks}")
        # 如果有迁移映射，先执行迁移
        if block_migration_map:
            print(f"block_migration_map: {block_migration_map}")
            # 过滤映射，确保所有索引都在有效范围内
            valid_map = {old: new for old, new in block_migration_map.items() 
                        if old < self.num_gpu_blocks and new < new_num_blocks}
            if valid_map:
                for i in range(self.num_attention_layers):
                    # 使用 Triton 执行原地迁移
                    self.migrate_blocks_inplace_with_triton(
                        self.gpu_cache[i],
                        valid_map, 
                        self.block_size
                    )
                    
                    # 迁移完成后直接切片
                    block_migration_map = None
            else:
                # 跳转到外层else的处理逻辑
                block_migration_map = None
        else:
            shape = list(self.gpu_cache[0].shape)
            shape[1] = new_num_blocks
            new_cache_list = []
            # 如果没有迁移映射，创建新缓存并彻底删除旧缓存
            for i in range(self.num_attention_layers-1, -1, -1):
                # 创建全新的张量，不保留与旧张量的任何关联
                # self.gpu_cache[i] = self.gpu_cache[i][:, :new_num_blocks]
                new_shape = list(self.gpu_cache[i].shape)
                new_shape[1] = new_num_blocks  # 修改第二个维度，与块数对应
                new_cache = torch.empty(new_shape, 
                                      dtype=self.dtype,
                                      device=self.device_config.device_type)
               
                new_cache.copy_(self.gpu_cache[i][:, :new_num_blocks])
                new_cache_list.append(new_cache)
                
                # 明确删除原始缓存
                del self.gpu_cache[i]
            
            # 逆序重建cache列表
            self.gpu_cache = new_cache_list[::-1]

        # 更新块数
        self.num_gpu_blocks = new_num_blocks
        
        # 强制执行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"decrease_gpu_blocks 显存使用情况: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
        end_time = time.time()
        logger.info(f"decrease_gpu_blocks 时间: {end_time - time_start:.2f} 秒")

    
    def increase_gpu_blocks(self, increase_num_blocks: int) -> None:
        """通过原地扩展方式增加GPU块的数量，避免大规模内存复制
        
        Args:
            increase_num_blocks: 要增加的块数量
        """
        if increase_num_blocks <= 0:
            return
        
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"before increase_gpu_blocks 显存使用情况: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
        begin_time = time.time()
        # 计算新的块数量
        new_num_blocks = self.num_gpu_blocks + increase_num_blocks
        
        def expand_cache_layer(i):
            old_shape = list(self.gpu_cache[i].shape)
            new_shape = old_shape.copy()
            new_shape[1] = increase_num_blocks
            new_cache = torch.zeros(new_shape, 
                                dtype=self.dtype,
                                device=self.device_config.device_type)
            self.gpu_cache[i] = torch.cat([self.gpu_cache[i], new_cache], dim=1)
        with ThreadPoolExecutor() as executor:
            list(executor.map(expand_cache_layer, range(self.num_attention_layers)))
        # # 对于每个注意力层分别处理
        # for i in range(self.num_attention_layers):
        #     old_shape = list(self.gpu_cache[i].shape)
        #     new_shape = old_shape.copy()
        #     new_shape[1] = increase_num_blocks
        #     # 0.4630403518676758 seconds
        #     new_cache = torch.zeros(new_shape, 
        #                            dtype=self.dtype,
        #                            device=self.device_config.device_type)
        #     self.gpu_cache[i] = torch.cat([self.gpu_cache[i], new_cache], dim=1)
            # self.gpu_cache[i] = F.pad(self.gpu_cache[i], 
            #              (0, 0,    # 第4维度(dim=128)不填充
            #              0, 0,    # 第3维度(blocks=4)不填充
            #              0, 0,    # 第2维度(heads=16)不填充
            #              0, increase_num_blocks))  # 第1维度(seq_len)末尾填充2982
            
        # 更新块数量
        self.num_gpu_blocks = new_num_blocks
        
        # 执行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"after increase_gpu_blocks 显存使用情况: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
        end_time = time.time()
        logger.info(f"increase_gpu_blocks 时间: {end_time - begin_time:.2f} 秒")
    def _resolve_migration_dependencies(self, block_migration_map):
        """解析块迁移映射中的依赖关系，确保按正确顺序执行
        
        Args:
            block_migration_map: 原始块迁移映射 {old_id: new_id}
            
        Returns:
            处理后的映射，保证执行顺序正确
        """
        if not block_migration_map:
            return {}
            
        # 复制映射进行处理
        processed_map = {}
        remaining = dict(block_migration_map)
        
        # 构建依赖图 - 找出哪些目标块是其他块的源块
        dependencies = {new_id: [] for old_id, new_id in block_migration_map.items()}
        for old_id, new_id in block_migration_map.items():
            if new_id in block_migration_map:  # 如果某个块的目标是另一个块的源
                dependencies[new_id].append(old_id)  # 记录依赖
        
        # 使用拓扑排序处理
        while remaining:
            # 找到无依赖的节点（那些目标块不是其他块的源块）
            independent = []
            for old_id, new_id in list(remaining.items()):
                if new_id not in remaining and not any(old_id in deps for deps in dependencies.values()):
                    independent.append(old_id)
            
            # 如果无法找到无依赖的节点，说明存在循环依赖
            # 需要打破循环，可以使用临时存储
            if not independent:
                # 选择一个节点打破循环
                old_id = next(iter(remaining.keys()))
                new_id = remaining[old_id]
                
                # 在临时空间存储内容，然后再移动
                # 这里只添加到处理队列，不实际执行移动
                processed_map[old_id] = new_id
                del remaining[old_id]
                continue
                
            # 处理无依赖的节点
            for old_id in independent:
                processed_map[old_id] = remaining[old_id]
                del remaining[old_id]
                
        return processed_map
    def decrease_gpu_blocks_with_triton(self,decrease_num_blocks: int, block_migration_map=None) -> None:
        """使用 Triton 加速的 GPU 块减少方法
        
        Args:
            decrease_num_blocks: 要减少的块数量
            block_migration_map: 从 block_manager 获得的块迁移映射 {old_id: new_id}
        """
        # 基本检查
        decrease_num_blocks = min(decrease_num_blocks, self.num_gpu_blocks)
        if decrease_num_blocks <= 0:
            return
            
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"before decrease_gpu_blocks_with_triton 显存使用情况: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")
        
        # 计算新的块数量
        new_num_blocks = self.num_gpu_blocks - decrease_num_blocks
        
        # 确定CUDA上的计算参数
        BLOCK_SIZE = 128  # 可根据实际硬件调整
        
        # 如果有迁移映射，先执行迁移
        if block_migration_map:
            # 过滤映射，确保所有索引都在有效范围内
            valid_map = {old: new for old, new in block_migration_map.items() 
                        if old < self.num_gpu_blocks and new < new_num_blocks}
            
            if valid_map:
                for i in range(self.num_attention_layers):
                    # 使用 Triton 执行原地迁移
                    self.migrate_blocks_inplace_with_triton(
                        self.gpu_cache[i],
                        valid_map, 
                        self.block_size
                    )
                    
                    # 创建新形状
                    new_shape = list(self.gpu_cache[i].shape)
                    new_shape[1] = new_num_blocks
                    
                    # 创建新缓存张量
                    new_cache = torch.empty(new_shape, 
                                        dtype=self.dtype,
                                        device=self.device_config.device_type)
                    
                    # 获取每个块的大小
                    block_size = self.block_size
                    num_heads = self.num_kv_heads
                    head_size = self.head_size
                    
                    # 计算总的数据元素数
                    block_elements = block_size * num_heads * head_size
                    
                    # 计算线程网格大小
                    grid = (new_num_blocks, (block_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
                    
                    # 调用 Triton 内核进行数据复制
                    kv_cache_trim_kernel[grid](
                        self.gpu_cache[i],
                        new_cache,
                        self.num_gpu_blocks,
                        new_num_blocks,
                        block_size, 
                        num_heads, 
                        head_size,
                        BLOCK_SIZE
                    )
                    
                    # 替换缓存
                    self.gpu_cache[i] = new_cache
            else:
                # 如果没有有效映射，使用常规方法
                block_migration_map = None
        
        # 如果没有迁移映射，直接裁剪缓存
        if not block_migration_map:
            for i in range(self.num_attention_layers):
                # 创建新形状
                new_shape = list(self.gpu_cache[i].shape)
                new_shape[1] = new_num_blocks
                
                # 创建新缓存张量
                new_cache = torch.empty(new_shape, 
                                    dtype=self.dtype,
                                    device=self.device_config.device_type)
                
                # 获取每个块的大小
                block_size = self.block_size
                num_heads = self.num_kv_heads
                head_size = self.head_size
                
                # 计算总的数据元素数
                block_elements = block_size * num_heads * head_size
                
                # 计算线程网格大小
                grid = (new_num_blocks, (block_elements + BLOCK_SIZE - 1) // BLOCK_SIZE)
                # 调用 Triton 内核进行数据复制
                kv_cache_trim_kernel[grid](
                    self.gpu_cache[i],
                    new_cache,
                    self.num_gpu_blocks,
                    new_num_blocks,
                    block_size, 
                    num_heads, 
                    head_size,
                    BLOCK_SIZE
                )
                
                # 替换缓存
                old_cache = self.gpu_cache[i]
                self.gpu_cache[i] = new_cache
                
                # 删除旧缓存
                del old_cache
                torch.cuda.empty_cache()
        
        # 更新块数量
        self.num_gpu_blocks = new_num_blocks
        
        # 执行垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
        
        allocated = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # 转换为GB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)
        logger.info(f"after decrease_gpu_blocks_with_triton 显存使用情况: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB")

   
@triton.jit
def kv_cache_inplace_migration_kernel(
    cache_ptr,
    migration_map_old_ptr, migration_map_new_ptr,
    num_mappings, block_size, num_heads, head_size,
    BLOCK_SIZE: tl.constexpr
):
    # 获取线程索引
    pid = tl.program_id(0)
    if pid >= num_mappings:
        return
        
    # 获取源块和目标块
    old_block_id = tl.load(migration_map_old_ptr + pid)
    new_block_id = tl.load(migration_map_new_ptr + pid)
    
    # 跳过原地块（不需要移动的块）
    if old_block_id == new_block_id:
        return
        
    # 计算每个块的元素数量
    block_elements = block_size * num_heads * head_size
    
    # 计算块内偏移
    for idx in range(0, block_elements, BLOCK_SIZE):
        offset = idx + tl.program_id(1) * BLOCK_SIZE
        if offset < block_elements:
            # 计算源地址和目标地址
            src_addr = old_block_id * block_elements + offset
            dst_addr = new_block_id * block_elements + offset
            
            # 复制数据
            val = tl.load(cache_ptr + src_addr.to(tl.int32))
            tl.store(cache_ptr + dst_addr.to(tl.int32), val)

@triton.jit
def kv_cache_extend_kernel(
    src_cache_ptr, dst_cache_ptr,
    old_num_blocks, new_num_blocks,
    block_size, num_heads, head_size,
    BLOCK_SIZE: tl.constexpr
):
    # 获取线程索引
    block_id = tl.program_id(0)  # 处理的块ID
    
    # 只处理原有的块
    if block_id >= old_num_blocks:
        return
        
    # 计算每个块包含的元素数量
    block_elements = block_size * num_heads * head_size
    
    # 循环处理块内每个数据点
    for idx in range(tl.program_id(1) * BLOCK_SIZE, block_elements, tl.num_programs(1) * BLOCK_SIZE):
        # 创建一个向量加载范围
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_elements
        
        # 计算源地址和目标地址
        src_addr = block_id * block_elements + offsets
        dst_addr = block_id * block_elements + offsets
        
        # 加载数据 - 修复索引类型问题
        x = tl.load(src_cache_ptr + src_addr.to(tl.int32), mask=mask)
        
        # 存储数据 - 修复索引类型问题
        tl.store(dst_cache_ptr + dst_addr.to(tl.int32), x, mask=mask)

@triton.jit
def kv_cache_trim_kernel(
    src_cache_ptr, dst_cache_ptr,
    old_num_blocks, new_num_blocks,
    block_size, num_heads, head_size,
    BLOCK_SIZE: tl.constexpr
):
    # 获取线程索引
    block_id = tl.program_id(0)  # 处理的块ID
    
    # 只处理需要保留的块
    if block_id >= new_num_blocks:
        return
    
    # 计算每个块包含的元素数量
    block_elements = block_size * num_heads * head_size
    
    # 循环处理块内每个数据点
    for idx in range(tl.program_id(1) * BLOCK_SIZE, block_elements, tl.num_programs(1) * BLOCK_SIZE):
        # 创建一个向量加载范围
        offsets = idx + tl.arange(0, BLOCK_SIZE)
        mask = offsets < block_elements
        
        # 计算源地址和目标地址
        src_addr = block_id * block_elements + offsets
        dst_addr = block_id * block_elements + offsets
        
        # 加载数据 - 修复索引类型问题
        x = tl.load(src_cache_ptr + src_addr, mask=mask)
        
        # 存储数据 - 修复索引类型问题
        tl.store(dst_cache_ptr + dst_addr, x, mask=mask)


@triton.jit
def tensor_slice_copy_kernel(
    src_ptr, dst_ptr,
    src_shape_0, src_shape_1, src_shape_2, src_shape_3,
    dst_shape_0, dst_shape_1, dst_shape_2, dst_shape_3,
    copy_shape_0, copy_shape_1, copy_shape_2, copy_shape_3,
    BLOCK_SIZE: tl.constexpr
):
    """高效复制4D张量切片的Triton内核

    这个内核函数的用途是将源张量的一个子区域复制到目标张量，
    用于替代 `dst.copy_(src[:, :n, ...])` 这样的操作。
    """
    # 计算全局线性索引
    pid = tl.program_id(0)
    
    # 计算源张量和目标张量的总元素数
    src_stride_1 = src_shape_2 * src_shape_3
    src_stride_0 = src_shape_1 * src_stride_1
    
    dst_stride_1 = dst_shape_2 * dst_shape_3
    dst_stride_0 = dst_shape_1 * dst_stride_1
    
    # 计算复制区域的总元素数
    total_elements = copy_shape_0 * copy_shape_1 * copy_shape_2 * copy_shape_3
    
    # 计算每个线程处理的元素数
    elements_per_thread = (total_elements + tl.num_programs(0) - 1) // tl.num_programs(0)
    
    # 计算当前线程的起始元素索引
    start_element = pid * elements_per_thread
    
    # 处理当前线程的元素
    for i in range(elements_per_thread):
        element_idx = start_element + i
        
        # 如果超出复制区域，则停止
        if element_idx >= total_elements:
            break
            
        # 计算4D索引
        idx_0 = element_idx // (copy_shape_1 * copy_shape_2 * copy_shape_3)
        remainder = element_idx % (copy_shape_1 * copy_shape_2 * copy_shape_3)
        
        idx_1 = remainder // (copy_shape_2 * copy_shape_3)
        remainder = remainder % (copy_shape_2 * copy_shape_3)
        
        idx_2 = remainder // copy_shape_3
        idx_3 = remainder % copy_shape_3
        
        # 计算源地址和目标地址
        src_addr = idx_0 * src_stride_0 + idx_1 * src_stride_1 + idx_2 * src_shape_3 + idx_3
        dst_addr = idx_0 * dst_stride_0 + idx_1 * dst_stride_1 + idx_2 * dst_shape_3 + idx_3
        
        # 复制数据 - 修复索引类型问题
        val = tl.load(src_ptr + src_addr.to(tl.int32))
        tl.store(dst_ptr + dst_addr.to(tl.int32), val)

def copy_tensor_slice_with_triton(src_tensor, dst_tensor, copy_shape=None):
    """使用Triton高效复制张量切片
    
    Args:
        src_tensor: 源张量
        dst_tensor: 目标张量
        copy_shape: 要复制的形状 (如果为None则使用dst_tensor的形状)
    """
    # 如果未指定复制形状，则使用目标张量的形状
    if copy_shape is None:
        copy_shape = list(dst_tensor.shape)
    
    # 确保是4D张量，如果维度不足则填充为1
    src_shape = list(src_tensor.shape)
    while len(src_shape) < 4:
        src_shape.append(1)
    
    dst_shape = list(dst_tensor.shape)
    while len(dst_shape) < 4:
        dst_shape.append(1)
    
    copy_shape_padded = list(copy_shape)
    while len(copy_shape_padded) < 4:
        copy_shape_padded.append(1)
    
    # 计算总元素数
    total_elements = 1
    for dim in copy_shape_padded:
        total_elements *= dim
    
    # 确定网格大小和块大小
    BLOCK_SIZE = 128
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # 启动内核
    tensor_slice_copy_kernel[grid](
        src_tensor, dst_tensor,
        src_shape[0], src_shape[1], src_shape[2] if len(src_shape) > 2 else 1, src_shape[3] if len(src_shape) > 3 else 1,
        dst_shape[0], dst_shape[1], dst_shape[2] if len(dst_shape) > 2 else 1, dst_shape[3] if len(dst_shape) > 3 else 1,
        copy_shape_padded[0], copy_shape_padded[1], copy_shape_padded[2], copy_shape_padded[3],
        BLOCK_SIZE
    )