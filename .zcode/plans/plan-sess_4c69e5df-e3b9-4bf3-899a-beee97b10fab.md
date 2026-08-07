# 多 GPU 多任务并行方案

## 目标
让 ComfyUI 支持"每个 GPU 负责一个任务、多任务并行"。把当前单线程 `prompt_worker` 改造成**每 GPU 一个 worker 线程**的线程池：空闲 GPU 自动认领队列里的下一个任务，每个 worker 启动时用 `set_torch_device(device)` 绑定自己的 GPU，各自拥有独立的 `PromptExecutor`。任务默认自动分配到空闲 GPU，若 prompt 在 `extra_data` 里带 `device` 指定（如 `"gpu:1"`）则路由到对应 worker。

## 设计原则
- **最小改动、复用现有机制**：复用 `MultiGPUThreadPool` 的 per-thread `set_torch_device` 模式、`PromptQueue` 已有的多消费者安全、`load_models_gpu`/`free_memory` 已有的 per-device 显存查询、`SelectModelDevice`/`deepclone_multigpu` 已有的 per-GPU 模型放置。
- **不破坏单 GPU 行为**：只有 1 个 GPU 时退化为现状（1 worker），不引入额外开销。MPS/CPU/DirectML 路径保持单 worker。
- **per-task 隔离而非全局拆分**：共享一份模型注册表（加锁，保留跨任务模型复用），但中断令牌、执行上下文（client_id / last_node_id）做成 per-task。

---

## 改动清单

### 1. 新增 `comfy/parallel_executor.py`（核心新文件，约 150 行）
封装"每 GPU 一个 worker 线程"的执行池，职责单一：
- `ParallelPromptExecutor(devices)`：构造时为每个 device 起一个 daemon 线程，线程入口先 `set_torch_device(device)`（沿用 `multigpu.py:38-40` 的写法），再构造自己的 `execution.PromptExecutor`（沿用 `main.py:349` 的构造参数）。
- 每个 worker 跑一个改造版的 `prompt_worker` 循环：`q.get()` → 执行 → `q.task_done()`，并在执行前后维护 worker 自己的 `InterruptToken`、`client_id`、`last_node_id`、`last_prompt_id`（per-worker 字段，不再读写 `server` 全局属性）。
- 设备亲和调度：`get()` 之后检查 item 是否带 `extra_data["device"]`，若带且不是自己的设备则放回队列并重新等待（用一个轻量的"归还/认领"逻辑，避免忙等）。无指定时任意 worker 认领。
- 提供 `interrupt(prompt_id)`：找到持有该 prompt 的 worker，置它的 `InterruptToken`。
- 提供 `shutdown()`：发 sentinel 优雅退出。

### 2. `execution.py` — 中断与执行上下文 per-task 化
- 把 `nodes.interrupt_processing(False)`（`execute_async:733`）和所有 `server.client_id` / `server.last_node_id` 读写改成走一个 **per-executor 的执行上下文对象**。
  - 新增 `PromptExecutor.context`，持有 `client_id`、`last_node_id`、`last_prompt_id`、`interrupted: bool`。
  - `execute_async` 开头 reset 自己的 `context` 而非全局 `nodes.interrupt_processing`。
  - `execution.py` 里所有 `server.client_id` / `server.last_node_id` 的引用（433, 436, 494-496, 536, 577-578, 683-684, 835）改成读 `PromptExecutor.context`（通过把 `server` 引用传进来后用 `executor.context` 取值，或把 context 作为参数透传到 `execute()`）。
  - `add_message`（677-684）改用 `self.context.client_id`。
- 中断检查点：现有代码通过 `nodes.interrupt_processing()` / `throw_exception_if_processing_interrupted()`（`model_management.py:2090-2102`）轮询全局布尔。新增一个 per-task 检查路径：在 worker 线程里把 `InterruptProcessingException` 的触发条件接到本 worker 的 token。为减少改动面，保留全局 `interrupt_processing` 作为"中断所有"通道，并让每个 worker 的 `execute_async` 开头**既 reset 全局也 reset 自己的 token**；中断检查点（`throw_exception_if_processing_interrupted`）扩展为"全局 or 本 worker token 命中则抛"——通过 `contextvars` 让检查函数读到当前 worker 的 token。

### 3. `comfy/model_management.py` — 给共享全局加锁
- 给 `current_loaded_models`（611）加一个模块级 `threading.RLock`（`_loaded_models_lock`），在 `load_models_gpu`（901-1001 的 insert at 1000）、`free_memory`（855-899 的 pop at 886）、`cleanup_models`（1045-1053）、`cleanup_models_gc`（1017-1034）的列表增删处加锁。`free_memory` 内部已 per-device 过滤，锁只保护 list 操作本身，粒度小。
- `soft_empty_cache()`（2025）/ `synchronize()`（2017）：改成 `torch.cuda.synchronize(device=None)` 时若能拿到当前 worker 设备则指定设备；保留无参版本用于全局清理路径。`free_memory` 里调用 `soft_empty_cache()`（893, 898）保持现状（它已在 per-device 上下文里，且 `empty_cache()` 本身是当前设备）。
- `vram_state` 保持全局（两 GPU 同策略，简单且够用）。

### 4. `main.py` — 启动逻辑切换
- `start_comfyui`（538）把单 `prompt_worker` 线程的启动替换为：根据 `comfy.model_management.get_all_torch_devices()` 构造 `ParallelPromptExecutor(devices)` 并启动。只有 1 个设备时仍走原单线程路径（行为等价，避免无谓复杂度）。
- 新增 CLI 参数 `--parallel-execution`（`cli_args.py`，store_true，默认 True 当 device_count>1，否则 False）控制是否启用多 worker；`--no-parallel-execution` 可关闭退回单线程。
- 原 `prompt_worker` 里的 GC / `free_memory` flag / `unload_models` 处理逻辑迁移到 `ParallelPromptExecutor`：每个 worker 处理自己的 flag，`unload_all_models` 已是多设备安全的（`model_management.py:2043-2045`）。

### 5. `server.py` — `/interrupt` 路由按 prompt_id 路由
- `/interrupt`（1160-1190）：带 `prompt_id` 时改成调 `parallel_executor.interrupt(prompt_id)`（per-task token），不再依赖全局 `nodes.interrupt_processing()`。不带 `prompt_id` 时仍走全局中断（中断所有 worker）。
- `send_sync`（1392）已经是线程安全的（`call_soon_threadsafe`），多 worker 并发发消息无需改动。

### 6. `comfy/multigpu.py` — 复用 `set_torch_device` 模式
- 不改 `MultiGPUThreadPool`（它是 intra-prompt CFG 分割，与本特性正交）。新文件 `parallel_executor.py` 直接复用 `comfy.model_management.set_torch_device`，不引入新依赖。

---

## 不改动的部分（明确边界）
- `ModelPatcher` / `deepclone_multigpu` / `cached_patcher_init` / `SelectModelDevice` 节点：已经 per-device，无需改动。用户若想把特定模型钉到特定 GPU 仍用现有节点。
- `load_models_gpu` / `free_memory` 的 per-device 显存查询逻辑：已经正确，只补锁。
- `PromptQueue`：已经多消费者安全，只补"设备亲和认领"逻辑（在 executor 侧，不动 queue 本身）。
- 单 GPU / MPS / CPU / DirectML 路径：device_count<=1 时退化为单 worker，零行为变化。

---

## 风险与权衡
- **`soft_empty_cache` 跨设备同步**：`torch.cuda.synchronize()` 无参时 sync 当前设备。在 per-thread-device 模型下，每个 worker 的"当前设备"是自己绑定的 GPU，所以 `free_memory` 在 worker 线程内调 `soft_empty_cache` 实际 sync 的是自己的设备——风险可控。全局 `/free` 路径会从主线程调，需在 `unload_all_models` 内对每个设备显式 sync（它已 iterate `get_all_torch_devices`，补一个 per-device sync）。
- **模型跨任务复用**：共享 `current_loaded_models` 意味着 GPU0 上跑完的模型若 GPU1 的任务也需要同模型，会在 GPU1 上各加载一份（显存独立），这是正确的——不同 GPU 显存不共享。同一 GPU 上的连续任务仍复用缓存。这与现有 `SelectModelDevice` 行为一致。
- **进度消息**：多个 worker 并发 `send_sync` 会把不同 prompt 的 `executing`/`executed`/`progress` 事件混在一起推给前端，但每条都带 `prompt_id`，前端按 `prompt_id` 区分即可（前端已是按 prompt_id 路由的）。`client_id` 改 per-task 后，每个 worker 的消息只发给发起该 prompt 的客户端。
- **AIMDO/dynamic patcher**：`ModelPatcherDynamic.load` 断言 `device_to == self.load_device`（`model_patcher.py:1856`），每个 GPU 必须有自己的 patcher 实例——这由现有 `deepclone_multigpu` 保证，多 worker 各自加载时也会各自构造 patcher，不冲突。

---

## 验证方式
1. `python -c "import main"` 语法/导入检查。
2. 单 GPU 环境启动，确认行为与改动前一致（1 worker，串行）。
3. 多 GPU 环境：提交 2 个 prompt，确认分别在两个 GPU 上并行执行（看日志的 device 和执行时间重叠）；中断其中一个确认不影响另一个。
4. 显存：两个任务用不同模型时，确认各自 GPU 各加载一份；用相同模型时确认同 GPU 上复用。

---

## 实现顺序（建议）
1. `comfy/model_management.py` 加锁（最小、独立、可单独验证）。
2. `execution.py` 执行上下文 per-task 化（中断 token + client_id/last_node_id）。
3. 新增 `comfy/parallel_executor.py`（worker 池 + 设备亲和调度）。
4. `main.py` + `cli_args.py` 启动切换 + CLI 开关。
5. `server.py` `/interrupt` 路由。
6. 多 GPU 环境联调验证。