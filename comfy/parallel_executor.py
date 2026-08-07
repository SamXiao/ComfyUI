"""
Per-GPU parallel prompt execution.

Spawns one worker thread per visible GPU. Each worker pins itself to its GPU
via set_torch_device() (CUDA's current device is per-thread), owns its own
PromptExecutor, and pulls prompts from the shared PromptQueue. This gives
inter-task parallelism: N independent prompts run on N GPUs concurrently.

Device affinity: a prompt may request a specific GPU by setting
extra_data["device"] to "gpu:N". A worker that pulls such a prompt but is not
on the requested device puts it back and waits; prompts without a device hint
are claimed by any idle worker.
"""

import threading
import logging

import comfy.model_management


class TaskInterruptFlag:
    """Mutable per-task interrupt flag, readable across threads.

    The owning worker thread binds it via
    comfy.model_management.bind_task_interrupt_flag() so the existing
    throw_exception_if_processing_interrupted() checks honor it. The parallel
    executor sets .interrupted from another thread to stop a single prompt
    without affecting prompts running on other GPUs.
    """
    __slots__ = ("interrupted", "prompt_id")

    def __init__(self, prompt_id):
        self.interrupted = False
        self.prompt_id = prompt_id


class WorkerState:
    """Per-worker bookkeeping exposed to the worker loop and the interrupt router."""
    __slots__ = ("device", "thread", "current_flag", "lock")

    def __init__(self, device):
        self.device = device
        self.thread = None
        self.current_flag = None
        self.lock = threading.Lock()


class ParallelPromptExecutor:
    """Manages one prompt-execution worker thread per GPU device.

    The worker loop body is supplied by the caller via `worker_target`, which
    keeps GPU/queue/asset concerns in main.py and out of this module. The
    worker_target is invoked as `worker_target(device, worker_state, executor)`
    on a thread that has already pinned its torch device and bound a task
    interrupt flag slot. It is expected to run an infinite loop pulling from
    the prompt queue and returning only on shutdown.
    """

    def __init__(self, devices, worker_target):
        self._devices = list(devices)
        self._worker_target = worker_target
        self._workers: list[WorkerState] = []
        self._shutdown = threading.Event()
        # prompt_id -> WorkerState, for routing per-task interrupts.
        self._assignment_lock = threading.Lock()
        self._assignments: dict[str, WorkerState] = {}

        for device in self._devices:
            ws = WorkerState(device)
            t = threading.Thread(target=self._run_worker, args=(ws,), daemon=True,
                                 name=f"prompt-worker-{device}")
            ws.thread = t
            self._workers.append(ws)
            t.start()

    def _run_worker(self, ws: WorkerState):
        try:
            comfy.model_management.set_torch_device(ws.device)
        except Exception as e:
            logging.error(f"prompt-worker-{ws.device}: failed to set device: {e}")
            return
        # Bind a task interrupt slot for this thread; the per-prompt flag is
        # swapped in by assign_prompt() before execution and reset after.
        comfy.model_management.bind_task_interrupt_flag(_NullFlag())
        try:
            self._worker_target(ws.device, ws, self)
        except Exception as e:
            logging.exception(f"prompt-worker-{ws.device}: worker loop crashed: {e}")

    def assign_prompt(self, prompt_id: str, ws: WorkerState) -> TaskInterruptFlag:
        """Bind a fresh interrupt flag for a prompt to the given worker thread.

        Called by the worker loop right before executing a prompt, on the
        worker's own thread. The flag is registered so interrupt(prompt_id)
        can target it from another thread.
        """
        flag = TaskInterruptFlag(prompt_id)
        with ws.lock:
            ws.current_flag = flag
        comfy.model_management.bind_task_interrupt_flag(flag)
        with self._assignment_lock:
            self._assignments[prompt_id] = ws
        return flag

    def finish_prompt(self, prompt_id: str, ws: WorkerState):
        """Clear the prompt's interrupt flag after execution."""
        with self._assignment_lock:
            self._assignments.pop(prompt_id, None)
        with ws.lock:
            ws.current_flag = None
        comfy.model_management.bind_task_interrupt_flag(_NullFlag())

    def interrupt(self, prompt_id: str) -> bool:
        """Interrupt a single running prompt by id. Returns True if found."""
        with self._assignment_lock:
            ws = self._assignments.get(prompt_id)
        if ws is None:
            return False
        with ws.lock:
            flag = ws.current_flag
        if flag is None:
            return False
        comfy.model_management.interrupt_current_task(flag)
        logging.info(f"Interrupting prompt {prompt_id} on {ws.device}")
        return True

    def interrupt_all(self):
        """Interrupt every currently running prompt (global interrupt)."""
        nodes_interrupt = comfy.model_management.interrupt_current_processing
        with self._assignment_lock:
            workers = list(self._assignments.values())
        for ws in workers:
            with ws.lock:
                flag = ws.current_flag
            if flag is not None:
                comfy.model_management.interrupt_current_task(flag)
        # Also flip the global flag so workers between prompts pick it up.
        nodes_interrupt(True)

    def shutdown(self):
        self._shutdown.set()
        # Workers are daemon threads; they exit with the process. No graceful
        # join is wired here because the worker loop blocks on queue.get().

    @property
    def shutdown_event(self):
        return self._shutdown

    @property
    def num_workers(self):
        return len(self._workers)

    @property
    def worker_devices(self):
        """Set of devices that have a worker bound to them. Used by device
        affinity routing to detect prompts pinned to a GPU with no worker."""
        return {ws.device for ws in self._workers}


class _NullFlag:
    """A no-op task interrupt flag bound to a worker thread when no prompt is
    running, so throw_exception_if_processing_interrupted() does not carry over
    a stale flag from the previous prompt."""
    __slots__ = ("interrupted",)
    def __init__(self):
        self.interrupted = False
