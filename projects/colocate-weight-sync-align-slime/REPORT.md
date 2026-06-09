# Weight-Sync Performance Optimization Report

**Date**: 2026-06-07 ~ 2026-06-08  
**Target**: vime colocate/non-colocate weight-sync vs slime baseline  
**Model**: Qwen3-30B-A3B MoE (bf16 + FP8), H200  
**PR**: [#176](https://github.com/vllm-project/vime/pull/176) (base = #170)  
**Issue**: [#175](https://github.com/vllm-project/vime/issues/175)

---

## 结果

| Mode | Before | After | slime | 改善 |
|------|--------|-------|-------|------|
| **Colocate bf16** | 33s | **24.7s** | 20s | 1.34x↑ |
| **Colocate FP8** | 42s | **31.5s** | 22s | 1.33x↑ |
| **Non-colocate bf16** | 53s | 未修 | 12s | 已定位根因 |

50步验证（bf16 40步 + FP8 50步）通过，metrics无regression。

---

## 代码修改详解 (commit 81cbad50)

修改了2个文件，核心是把**每参数一个IPC handle → 每chunk一个packed buffer**。

### 问题

vime每个chunk（~170个参数tensor）逐个调用`reduce_tensor`创建CUDA IPC handle，engine端逐个调用`rebuild_cuda_tensor`（即`cudaIpcOpenMemHandle`）打开。30B MoE模型115 chunks × ~170参数 ≈ **19,500次IPC open**。`cudaIpcOpenMemHandle`是driver-level串行调用，这是weight-sync比slime慢4.3x的主因。

### 修复

使用vLLM自带的`pack_tensors`/`unpack_tensor`（`vllm.distributed.weight_transfer.packed_tensor`），把整个chunk的所有tensor拼接成一个contiguous uint8 buffer，只导出一个IPC handle。

#### 文件1: `update_weight_from_tensor.py`

**Trainer端 — `_build_ipc_update_info_from_named_tensors`**

修改前：逐参数创建IPC handle
```python
# 旧：每个参数一个handle
for name, tensor in named_tensors:
    weight = tensor.detach().contiguous()
    rebuild_func, ipc_args = reduce_tensor(weight)  # 逐个导出
    ipc_handles.append({gpu_uuid: (rebuild_func, ipc_args)})
return {"names": ..., "ipc_handles": ipc_handles}, weight_refs
```

修改后：整chunk打包成一个buffer
```python
# 新：整个chunk一个packed buffer
from vllm.distributed.weight_transfer.packed_tensor import pack_tensors

chunk = pack_tensors(iter(named_tensors), lambda item: item[1].detach(), 1 << 40)
rebuild_func, ipc_args = reduce_tensor(chunk.packed_tensor)  # 只导出一次
return {
    "packed_handle": {gpu_uuid: (rebuild_func, ipc_args)},  # 1个handle
    "names": chunk.names,
    "shapes": chunk.shapes,
    "dtype_names": [str(dt).split(".")[-1] for dt in chunk.dtypes],
    "tensor_sizes": chunk.tensor_sizes,  # 新增：每个tensor在buffer中的字节数
}, [chunk.packed_tensor]
```

`pack_tensors`把所有tensor flatten成uint8后cat到一个连续buffer中，记录每个tensor的name/shape/dtype/size用于解包。payload schema从`ipc_handles: list[dict]`（每参数一个）变成`packed_handle: dict`（整chunk一个）+ `tensor_sizes: list[int]`。

**Trainer端 — `_merge_ipc_update_infos`**

修改前：逐参数合并handle dict（N个参数 × M个rank → N个合并dict）
```python
for i in range(num_params):
    combined = {}
    for info in infos:
        combined.update(info["ipc_handles"][i])
    merged_handles.append(combined)
```

修改后：只合并一个packed_handle dict（M个rank → 1个合并dict）
```python
combined = {}
for info in infos:
    combined.update(info["packed_handle"])
return {**base, "packed_handle": combined}
```

**Engine端 — `vLLMColocateWorkerExtension.update_weights_chunk`**

修改前：逐参数打开IPC handle + 重建tensor
```python
for name, shape, ipc_handle in zip(names, shapes, ipc_handles):
    func, args = ipc_handle[physical_gpu_id]
    list_args[6] = device_index
    weight = func(*list_args)              # cudaIpcOpenMemHandle × N
    weights.append((name, weight))
```

修改后：打开一个packed buffer + unpack切分
```python
from vllm.distributed.weight_transfer.packed_tensor import unpack_tensor

func, args = packed_handle[physical_gpu_id]
list_args[6] = device_index
packed_buf = func(*list_args)              # cudaIpcOpenMemHandle × 1

dtypes = [getattr(torch, dn) for dn in inner["dtype_names"]]
weights = unpack_tensor(                   # 按tensor_sizes切片，zero-copy views
    packed_buf, inner["names"], inner["shapes"], dtypes, inner["tensor_sizes"]
)
```

`unpack_tensor`根据`tensor_sizes`对packed buffer做`narrow` + `view` + `reinterpret_cast`，产生per-param的tensor view，无额外内存分配。

#### 文件2: `vllm_engine.py`

**VLLMEngine.update_weights_from_tensor** — Ray actor的bridge方法

修改前：接收`ipc_handles: list[dict]`，cloudpickle序列化后POST到engine
```python
def update_weights_from_tensor(self, *, names, dtype_names, shapes, ipc_handles, ...):
    payload = {"names": ..., "ipc_handles_pickled": base64(cloudpickle(ipc_handles))}
```

修改后：接收`packed_handle: dict` + `tensor_sizes: list[int]`
```python
def update_weights_from_tensor(self, *, packed_handle, names, shapes, dtype_names, tensor_sizes, ...):
    payload = {
        "names": ..., "shapes": ..., "dtype_names": ..., "tensor_sizes": ...,
        "packed_handle_pickled": base64(cloudpickle(packed_handle)),
    }
```

`packed_handle`只有一个`{gpu_uuid: (rebuild_func, args)}`条目（vs 旧的170个），cloudpickle序列化量大幅减小。`tensor_sizes`是新增字段，纯int list，JSON-native不需要pickle。

### 效果量化

| 指标 | 修改前 | 修改后 | 改善 |
|------|--------|--------|------|
| IPC handle数/sync | ~19,500 | 115 | **170x↓** |
| Trainer build时间 | 1.2-1.7s | 0.5-0.7s | ~2.5x↓ |
| Engine rayget时间 | 10.9-11.8s | 4.3-5.3s | ~2.2x↓ |
| Engine per-chunk rebuild | 0.035s | — (unpack ~0) | — |
| Engine per-chunk load_weights | 0.03-0.086s | 0.03-0.086s | 不变 |
| **Loop total (115 chunks)** | **12-13.5s** | **5.3s** | **2.3x↓** |
| **update_weights_time** | **33s** | **24.7s** | **1.34x↓** |

剩余的~19s是loop外的固定开销（weights_getter megatron→HF转换、pause/flush/continue engine、layerwise init/finalize），与IPC无关。

---

## 未解决 & 根因分析

### FP8 colocate / Non-colocate 为什么不能进一步优化

剩余差距的根因在vLLM的layerwise reload pipeline。每次weight sync时：
1. `initialize_layerwise_reload` — clone所有参数（~17GB）、restore到meta device、wrap weight_loaders
2. `_layerwise_process` — per-layer materialize + buffer分配 + `process_weights_after_loading`
3. `finalize_layerwise_reload` — 清理wrapper

尝试bypass layerwise失败，原因链：
- `replace_parameter()`（`vllm/model_executor/utils.py:92`）创建plain `nn.Parameter`时只复制`weight_loader`，丢失`quant_method`等属性 → FP8 crash
- `replace_parameter()`把`ModelWeightParameter`子类降级为`nn.Parameter`，丢失`load_row_parallel_weight`等类方法 → TP sharding crash  
- `process_weights_after_loading`做weight layout变换（`swap_w13_to_w31`），bypass后第二次`model.load_weights`往kernel格式buffer写checkpoint格式数据 → shape mismatch

**vLLM上游**: [#42821](https://github.com/vllm-project/vllm/issues/42821) (issue) / [#42823](https://github.com/vllm-project/vllm/pull/42823) (PR) / [#44353](https://github.com/vllm-project/vllm/pull/44353) (weight sync重构)

**其它框架做法**:
- **verl**: trainer端预量化成FP8再发送（`quant_weights`）；monkey-patch `replace_parameter`保留子类属性 + `__class__`交换
- **SkyRL**: 直接调`model_runner.reload_weights()`（内部走layerwise），不自己优化
- **slime**: 用sglang，没有vLLM layerwise问题

### 后续优化方向

1. **学verl在trainer端预量化** — 改vime的weights_getter，发FP8格式权重，engine端不需要layout变换
2. **优化vLLM layerwise本身** — 跳过meta restore/materialize（参数已在device）、缓存FusedMoE buffer
3. **跟踪vLLM upstream** — PR #44353 重构weight sync，移除`is_checkpoint_format`

---

## 测试工具

`h200-0:/home/aoshen/vime/projects/slime-vime-comparison/` 下：

- `launch_wsprof_vime_flatten.sh` / `_fp8.sh` — colocate WSPROF launcher
- `launch_noncolo_vime.sh` / `_slime.sh` — 2-node non-colocate launcher  
- `vime-worktree-flatten/` — fix分支代码 (aoshen/colocate-ipc-flatten)
