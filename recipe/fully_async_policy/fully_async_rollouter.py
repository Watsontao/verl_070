# Copyright 2025 Meituan Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import time
import json
import re
from pprint import pformat
from collections import defaultdict

import numpy as np
import ray
import torch
from ray import ObjectRef

from verl import DataProto
from recipe.fully_async_policy.detach_utils import (
    RolloutSample,
    ValidateMetrics,
    prepare_single_generation_data,
)
from recipe.fully_async_policy.message_queue import MessageQueueClient
from recipe.fully_async_policy.ray_trainer import FullyAsyncRayPPOTrainer
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.profiler import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger


@ray.remote(num_cpus=10, max_concurrency=100)
class FullyAsyncRollouter(FullyAsyncRayPPOTrainer):
    """
    Asynchronous sample generator, responsible for continuously generating training samples
    and putting them into MessageQueue
    Based on the mature implementation improvements of OneStepOffRayTrainer
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        device_name=None,
    ):
        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        self.val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        assert not self.hybrid_engine
        assert self.config.data.train_batch_size == 0, "train_batch_size must be zero"
        assert self.config.data.gen_batch_size == 1, "gen_batch_size must be one"
        assert self.config.async_training.staleness_threshold >= 0, "staleness_threshold must larger than 0"
        assert self.config.async_training.trigger_parameter_sync_step >= 1, (
            "trigger_parameter_sync_step must larger than 1"
        )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = False
        self.kl_ctrl_in_reward = False
        self.use_critic = False
        self.use_reference_policy = False
        self.use_rm = False

        print("[FullyAsyncRollouter] Creating datasets...")
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor)
        val_dataset = create_rl_dataset(config.data.val_files, config.data, tokenizer, processor)
        train_sampler = create_rl_sampler(config.data, train_dataset)

        self._validate_config()
        print(f"[FullyAsyncRollouter] Rollouter _create_dataloader...\n{train_dataset}\n{val_dataset}")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ==================== fully async config ====================

        self.total_rollout_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.rollout.total_rollout_steps is not None:
            self.total_rollout_steps = min(self.config.rollout.total_rollout_steps, self.total_rollout_steps)
        print(f"[FullyAsyncRollouter] Total rollout steps: {self.total_rollout_steps}")
        self.total_train_steps = None

        # Rollouter parameter configuration
        self.message_queue_client = None

        # Worker groups: rollout_wg is same to actor_rollout_wg
        self.rollout_wg = None
        self.actor_rollout_wg = None
        self.async_rollout_manager = None

        # Config
        self.staleness_threshold: float = config.async_training.get("staleness_threshold", 1)
        # required_samples use ppo_mini_batch_size*require_batches as the minimum number of samples.
        self.require_batches = config.async_training.require_batches
        self.required_samples = config.actor_rollout_ref.actor.ppo_mini_batch_size * self.require_batches
        self.max_required_samples = None
        self.max_concurrent_samples = None
        # queue size
        self.max_queue_size = None

        # Statistics
        self.current_param_version = 0
        self.total_generated_samples = 0
        self.staleness_samples = 0
        self.dropped_stale_samples = 0
        self.processed_sample_count = 0
        # we start from step 1
        self.global_steps = 1
        self.idle_start_time = None
        self.version_start_time = None

        # Concurrency control
        # Modified by self.pause() or self._should_pause_generation()
        self.paused = False
        self.running = True
        self.monitor_loop_trigger = True

        # Add dataloader lock
        self.dataloader_lock = asyncio.Lock()

        # Statistics for long-tail analysis
        self.total_long_probes = 0
        self.total_long_matches_sum = 0.0 # Sum of ratios (count_long_rest / (n-1))
        self.long_tail_threshold = 512 # Define long-tail as > 512 tokens

        # P-DSR State
        self.sample_buffer = {}
        
        # P-DSR Aggregation (v3.0)
        self.fake_batch_size = 128
        self.batch_aggregator = defaultdict(list) # {batch_id: [probe_lengths]}

        # Initialize async queues
        # P-DSR: probe_queue for initial sampling, re_queue for Rest/Retry (high priority)
        self.probe_queue = asyncio.Queue(maxsize=0)
        self.re_queue = asyncio.Queue(maxsize=0)
        self.active_tasks = set()
        self.cancel_queue = asyncio.Queue()

    def _init_async_objects(self):
        # Initialize asyncio synchronization primitives.
        # We let asyncio.Condition create the Lock internally to ensure they share the same Event Loop.
        # This avoids 'ValueError: loop argument must agree with lock' which can occur in Ray environments
        # where the lock's captured loop (get_running_loop) differs from Condition's default loop check.
        # Explicitly passing the loop is deprecated/removed in Python 3.10+, so this reverse-initialization
        # is the most robust workaround.
        self.condition = asyncio.Condition()
        self.lock = self.condition._lock

    async def set_message_queue_client(self, message_queue_client: MessageQueueClient):
        """Set message queue client"""
        async with self.lock:
            self.message_queue_client = message_queue_client

    async def set_max_required_samples(self):
        async with self.lock:
            self.max_required_samples = int(
                self.required_samples
                * (self.staleness_threshold + 1)
                * self.config.async_training.trigger_parameter_sync_step
            )
            self.total_train_steps = int(
                self.total_rollout_steps
                / (self.required_samples * self.config.async_training.trigger_parameter_sync_step)
            )

            self.max_concurrent_samples = len(self.async_rollout_manager.server_handles) * 128
            self.max_concurrent_samples = min(self.max_concurrent_samples, self.max_required_samples)
            self.max_queue_size = self.max_required_samples

            print(
                f"[FullyAsyncRollouter] required_samples : {self.required_samples} "
                f"max_required_samples: {self.max_required_samples} "
                f"max_queue_size: {self.max_queue_size} "
                f"total_train_steps: {self.total_train_steps} "
                f"total_rollout_steps: {self.total_rollout_steps} "
                f"max_concurrent_samples: {self.max_concurrent_samples} "
            )

    def get_rollout_wg(self):
        """Get rollout worker group"""
        return self.rollout_wg

    def get_max_queue_size(self):
        return self.max_queue_size

    def get_total_train_steps(self):
        return self.total_train_steps

    async def update_param_version(self, version: int, validate: bool = False, global_steps: int = 0):
        """Update current parameter version"""
        async with self.lock:
            old_version = self.current_param_version
            self.current_param_version = version
            # every time param change, reset staleness_samples
            self.staleness_samples = (
                len(self.active_tasks) + self.cancel_queue.qsize() + await self.message_queue_client.get_queue_size()
            )
            timing_raw = {}
            idle_ratio = None
            if self.idle_start_time is not None and self.version_start_time is not None:
                rollout_active_time = self.idle_start_time - self.version_start_time
                rollout_version_time = time.time() - self.version_start_time
                idle_ratio = 1 - rollout_active_time / rollout_version_time
                timing_raw["rollouter/active_time"] = rollout_active_time
                timing_raw["rollouter/version_time"] = rollout_version_time
                timing_raw["rollouter/idle_ratio"] = idle_ratio
                self.idle_start_time = None
            print(
                f"[FullyAsyncRollouter][Public][update_param_version] "
                f"Parameter version updated from {old_version} to {version} "
                f",reset staleness_samples to: {self.staleness_samples}"
                f",idle_ratio: {idle_ratio}"
            )
            val_metrics = None
            if (
                self.val_reward_fn is not None
                and self.config.rollout.test_freq > 0
                and self.current_param_version % self.config.rollout.test_freq == 0
                and self.current_param_version > 0  # don't test here in the initial parameter sync
            ) or (validate and self.val_reward_fn is not None):
                with marked_timer("rollouter/validate_time", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
            data = ValidateMetrics(
                timing_raw=timing_raw, metrics=val_metrics, global_steps=global_steps, param_version=version
            )
            await self.message_queue_client.put_validate(ray.cloudpickle.dumps(data))

            self.version_start_time = time.time()

    async def save_checkpoint(self, local_global_step_folder: str):
        # WARNING!: Due to the asynchronous nature, there are some in-flight samples
        # (pending/cancel/result queue and message queue).
        # Therefore, directly saving the state of the dataloader will result in losing these
        # samples when resuming training.
        # TODO: Implement dataloader recovery without losing in-flight samples.
        from verl.utils.fs import local_mkdir_safe

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        async with self.dataloader_lock:
            dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        print(f"[FullyAsyncRollouter] Saved dataloader checkpoint to {dataloader_local_path}")

    def load_checkpoint(self):
        """Load checkpoint including dataloader state based on resume mode"""

        if self.config.trainer.resume_mode == "disable":
            print("[FullyAsyncRollouter] Resume mode is disabled, starting from scratch")
            return 0

        # Determine checkpoint folder path
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("[FullyAsyncRollouter] Load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        # Find and validate global_step_folder based on resume mode
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("[FullyAsyncRollouter] Training from scratch (no checkpoint found)")
                return 0
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), (
                "[FullyAsyncRollouter] resume_from_path must be str type"
            )
            assert "global_step_" in self.config.trainer.resume_from_path, (
                "[FullyAsyncRollouter] resume_from_path must specify the global_steps"
            )
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                working_dir = os.getcwd()
                global_step_folder = os.path.join(working_dir, global_step_folder)
        else:
            raise ValueError(f"[FullyAsyncRollouter] Unknown resume_mode: {self.config.trainer.resume_mode}")

        print(f"[FullyAsyncRollouter] Loading checkpoint from: {global_step_folder}")

        # Extract and set global step
        trainer_global_steps = int(global_step_folder.split("global_step_")[-1])
        self.global_steps = (
            trainer_global_steps * self.required_samples * self.config.async_training.trigger_parameter_sync_step + 1
        )
        print(f"[FullyAsyncRollouter] Setting global_steps to {self.global_steps}")

        # Load dataloader state
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"[FullyAsyncRollouter] Loaded dataloader state from {dataloader_local_path}")
        else:
            print(
                f"[FullyAsyncRollouter] Warning: No dataloader state found at {dataloader_local_path}, "
                f"will start from scratch"
            )

    def _validate_config(self):
        # Validate asynchronous training configuration
        if not hasattr(self.config, "async_training"):
            raise ValueError("[FullyAsyncRollouter] Missing async_training configuration")
        assert self.config.actor_rollout_ref.rollout.calculate_log_probs, "must rollout calculate log_probs"

    async def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self._init_async_objects()
        self._init_resource_pools()
        self._create_worker_classes()
        self._init_worker_groups()
        self._init_models()
        await self._init_async_rollout_manager()

    def _create_actor_rollout_classes(self):
        # only create rollout
        for role in [Role.Rollout]:
            resource_pool = self.resource_pool_manager.get_resource_pool(role)
            role_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[role],
                config=self.config.actor_rollout_ref,
                role=str(role),
            )
            self.resource_pool_to_cls[resource_pool][str(role)] = role_cls

    def _init_models(self):
        self.rollout_wg = self.all_wg[str(Role.Rollout)]
        self.rollout_wg.init_model()
        self.actor_rollout_wg = self.rollout_wg

    def _create_continuous_iterator(self):
        """
        Create a continuous data iterator across epoch
        """
        for epoch in range(self.config.rollout.total_epochs):
            iterator = iter(self.train_dataloader)
            for batch_dict in iterator:
                yield epoch, batch_dict

    async def _init_async_rollout_manager(self):
        # create async rollout manager and request scheduler
        assert self.config.actor_rollout_ref.rollout.mode == "async"
        from recipe.fully_async_policy.agent_loop import FullyAsyncAgentLoopManager

        self.async_rollout_mode = True
        self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
            config=self.config,
            worker_group=self.rollout_wg,
        )

    # Add samples to the pending_queue
    async def _feed_samples(self):
        continuous_iterator = self._create_continuous_iterator()
        
        sample_global_idx = 0

        for epoch, batch_dict in continuous_iterator:
            # P-DSR: Generate PROBE first (N=1)
            full_batch = prepare_single_generation_data(batch_dict, self.config, override_n=1)

            sample_id = f"sample_{epoch}_{self.global_steps}"
            
            # P-DSR: Fake Batch ID
            fake_batch_id = sample_global_idx // self.fake_batch_size
            sample_global_idx += 1
            
            # Cache the raw batch_dict for generating the rest (N-1) later
            self.sample_buffer[sample_id] = batch_dict

            rollout_sample = RolloutSample(
                full_batch=full_batch,
                agent_loop_output_list=[None] * 1, # Probe only needs 1 slot
                sample_id=sample_id,
                epoch=epoch,
                param_version=0,
                param_version_start=[],
                param_version_end=[],
                processing_times=[],
                tool_calls=[],
                rollout_status={'is_probe': True, 'fake_batch_id': fake_batch_id}, # Mark as Probe with Batch ID
            )

            await self.probe_queue.put(rollout_sample)

            # Check if have reached the last step
            if self.global_steps >= self.total_rollout_steps:
                print(
                    f"[FullyAsyncRollouter][Feed] "
                    f"Maximum count has been reached, stop adding new samples"
                    f"{self.global_steps} >= {self.total_rollout_steps}"
                )
                break

            self.global_steps += 1

        # End signal
        await self.probe_queue.put("DONE")
        print(f"[FullyAsyncRollouter][Feed] Sample addition is complete, {self.global_steps} samples have been added")

    async def _processor_worker(self):
        """
        Streaming worker coroutines, a sample is submitted for processing without waiting for batches
        """
        while True:
            try:
                # 1. Determine where to get the next sample
                # ALWAYS prioritize existing tasks (cancel or rest/retry)
                # These must be processed to avoid deadlocks
                simple_from_cancel_queue = False
                from_re_queue = False
                rollout_sample = None

                if not self.cancel_queue.empty():
                    rollout_sample = await self.cancel_queue.get()
                    simple_from_cancel_queue = True
                    # --- P-DSR Fix: Sanitize Cancelled Samples ---
                    if not isinstance(rollout_sample.agent_loop_output_list, list):
                        rollout_sample.agent_loop_output_list = [None] * len(rollout_sample.full_batch)
                    # ---------------------------------------------
                elif not self.re_queue.empty():
                    rollout_sample = await self.re_queue.get()
                    from_re_queue = True
                
                # 2. If no existing tasks, check for pause before taking NEW probes
                if rollout_sample is None:
                    if self.paused or await self._should_pause_generation():
                        if not self.paused:
                            print("[FullyAsyncRollouter][Processor] 达到样本限制，暂停接收新 Probe...")
                        async with self.lock:
                            self.paused = True
                            
                        # Wait for existing active tasks to finish if we are paused
                        while self.active_tasks:
                            async with self.lock:
                                if self.active_tasks:
                                    done_tasks, self.active_tasks = await asyncio.wait(
                                        self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                                    )
                                for task in done_tasks:
                                    await task

                        async with self.lock:
                            while self.paused:
                                self.idle_start_time = time.time()
                                # P-DSR Fix: Do not wait indefinitely. Wake up to check re_queue.
                                try:
                                    await asyncio.wait_for(self.condition.wait(), timeout=1.0)
                                except asyncio.TimeoutError:
                                    # Timeout means we should check if re_queue has items
                                    pass
                                
                                # If re_queue has items, we must break the pause loop to process them
                                if not self.re_queue.empty() or not self.cancel_queue.empty():
                                    break
                                    
                        continue # Re-check queues after resume
                    
                    # Take NEW probe from queue
                    rollout_sample = await self.probe_queue.get()
                    if rollout_sample != "DONE":
                        self.staleness_samples += 1

                # 3. Handle Termination Signal
                if rollout_sample == "DONE":
                    print("[FullyAsyncRollouter][Processor] 收到结束信号，等待剩余任务完成...")
                    while self.active_tasks:
                        async with self.lock:
                            if self.active_tasks:
                                done_tasks, self.active_tasks = await asyncio.wait(
                                    self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                                )
                            for task in done_tasks:
                                await task
                    break

                # 4. Concurrency Control (Flow Control)
                while len(self.active_tasks) >= self.max_concurrent_samples:
                    async with self.lock:
                        if self.active_tasks:
                            done_tasks, self.active_tasks = await asyncio.wait(
                                self.active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                        for task in done_tasks:
                            await task

                # 5. Submit sample processing
                async with self.lock:
                    while self.paused and rollout_sample.rollout_status.get('is_probe'):
                        await self.condition.wait()
                        
                    task = asyncio.create_task(
                        self._process_single_sample_streaming(rollout_sample),
                        name=rollout_sample.sample_id,
                    )
                    self.active_tasks.add(task)

                # 6. Cleanup
                if simple_from_cancel_queue:
                    self.cancel_queue.task_done()
                elif from_re_queue:
                    self.re_queue.task_done()
                else:
                    self.probe_queue.task_done()

            except Exception as e:
                import traceback
                print(f"[FullyAsyncRollouter] Processor loop crashed: {e}")
                print(traceback.format_exc())
                await asyncio.sleep(1)

    async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
        """Process a single sample streamingly"""
        print(f"[P-DSR Debug] Start processing {rollout_sample.sample_id} (Priority: {rollout_sample.rollout_status.get('priority', 'Default')})")
        
        # Fix: If retrying from cancel_queue, agent_loop_output_list might be a DataProto. Reset it.
        if not isinstance(rollout_sample.agent_loop_output_list, list):
            rollout_sample.agent_loop_output_list = [None] * len(rollout_sample.full_batch)

        # Calling asynchronous generation methods
        rollout_sample.full_batch.non_tensor_batch["param_version"] = [self.current_param_version] * len(
            rollout_sample.full_batch
        )
        
        # --- P-DSR: Inject Priority into Meta Info ---
        priority = rollout_sample.rollout_status.get('priority', 'HEAVY')
        rollout_sample.full_batch.meta_info['priority'] = priority
        # ---------------------------------------------

        print(f"[Rollouter] 📤 发送任务 {rollout_sample.sample_id} -> Worker. 优先级: {priority}, 探路: {rollout_sample.rollout_status.get('is_probe', False)}")

        ret, is_cancel = await self.async_rollout_manager.generate_single_sample_async(
            rollout_sample.full_batch, rollout_sample.agent_loop_output_list
        )
        
        print(f"[Rollouter] 📥 任务 {rollout_sample.sample_id} 返回. 取消状态: {is_cancel}")
        
        # --- P-DSR Debug: Check Status ---
        print(f"[P-DSR Debug] 任务 {rollout_sample.sample_id} 状态: {rollout_sample.rollout_status}")
        # ---------------------------------
        
        # --- P-DSR Debug: 检查是否发生取消 ---
        if is_cancel:
            print(f"[P-DSR Debug] ⚠️ 任务 {rollout_sample.sample_id} 被 vLLM 取消! ret类型={type(ret)}")
        # ------------------------------------
        
        # --- P-DSR: Circuit Breaker & Retry (熔断重试) ---
        if not is_cancel and priority == 'FAST':
            # 检查是否撞墙 (Cap = 1024)
            if 'response_mask' in ret.batch:
                response_mask = ret.batch['response_mask']
                if hasattr(response_mask, 'cpu'): response_mask = response_mask.cpu()
                lengths = response_mask.sum(dim=-1)
                
                # 只要有一个样本被截断（达到或超过 1024），整个 Batch 重试
                if (lengths >= 1024).any():
                    print(f"[P-DSR] 🚨 样本 {rollout_sample.sample_id} 触发熔断！(长度 >= 1024) -> 正在重回 Heavy 队列...")
                    
                    # 1. 升级优先级
                    rollout_sample.rollout_status['priority'] = 'HEAVY'
                    
                    # 2. 重新入队 (使用 re_queue)
                    await self.re_queue.put(rollout_sample)
                    
                    # 3. 提前结束 (丢弃本次生成的残次品 ret)
                    return 
        # ------------------------------------------------
        
        if not is_cancel:
            # --- P-DSR Logic: Reassembly (Rollouter-Side) ---
            
            # 1. Handle Probe Completion
            is_probe = rollout_sample.rollout_status.get('is_probe', False)
            if is_probe:
                # Get Basic Info
                if 'response_mask' in ret.batch:
                    response_mask = ret.batch['response_mask']
                    if hasattr(response_mask, 'cpu'): response_mask = response_mask.cpu()
                    probe_len = response_mask.sum(dim=-1).item()
                else:
                    probe_len = 0 # Fallback

                # P-DSR Aggregation
                batch_id = rollout_sample.rollout_status.get('fake_batch_id', 0)
                
                # Cache the sample object (we need it later for dispatch) and the result
                if rollout_sample.sample_id in self.sample_buffer:
                    # Update buffer with Probe Result (for final reassembly)
                    stored_data = self.sample_buffer[rollout_sample.sample_id]
                    
                    # P-DSR v0.7.0 Fix: In v0.7.0, 'input_ids' doesn't exist in raw batch_dict.
                    # We check if it's already wrapped. If not, wrap it.
                    if isinstance(stored_data, dict) and 'input' not in stored_data: 
                         self.sample_buffer[rollout_sample.sample_id] = {
                             'input': stored_data,
                             'probe_ret': ret
                         }
                    else:
                         # Already wrapped or other structure
                         self.sample_buffer[rollout_sample.sample_id]['probe_ret'] = ret

                self.batch_aggregator[batch_id].append((rollout_sample.sample_id, probe_len))
                
                # Check if Batch is Full
                if len(self.batch_aggregator[batch_id]) >= self.fake_batch_size:
                    # --- TRIGGER BATCH DISPATCH ---
                    batch_data = self.batch_aggregator.pop(batch_id)
                    
                    # Sort by Length Descending
                    batch_data.sort(key=lambda x: x[1], reverse=True)
                    
                    # Top 20% Cutoff
                    n_total = len(batch_data)
                    cut_idx = int(n_total * 0.20)
                    if cut_idx == 0: cut_idx = 1
                    
                    heavy_samples = batch_data[:cut_idx]
                    fast_samples = batch_data[cut_idx:]
                    
                    # Calculate Dynamic Cap (1.5x of the boundary)
                    boundary_len = heavy_samples[-1][1]
                    dynamic_cap = int(boundary_len * 1.5)
                    
                    print(f"[P-DSR] ⚖️ Batch {batch_id} 排序完成. Cutoff: {boundary_len}, Cap: {dynamic_cap}. Heavy: {len(heavy_samples)}, Fast: {len(fast_samples)}")

                    # Dispatch Helper
                    async def dispatch(sid, prio, cap=None):
                        if sid not in self.sample_buffer: return
                        
                        # Retrieve Input Data
                        buf_data = self.sample_buffer[sid]
                        input_batch = buf_data['input']
                        
                        # Generate Rest
                        rest_n = self.config.actor_rollout_ref.rollout.n - 1
                        if rest_n > 0:
                            full_batch_rest = prepare_single_generation_data(input_batch, self.config, override_n=rest_n)
                            
                            # Inject Cap if needed
                            if cap:
                                full_batch_rest.meta_info['max_tokens'] = cap
                            
                            # Create Sample
                            rest_sample = RolloutSample(
                                full_batch=full_batch_rest,
                                agent_loop_output_list=[None] * rest_n,
                                sample_id=sid,
                                epoch=0,
                                param_version=self.current_param_version,
                                param_version_start=[],
                                param_version_end=[],
                                processing_times=[],
                                tool_calls=[],
                                rollout_status={'priority': prio, 'is_rest': True}, 
                            )
                            await self.re_queue.put(rest_sample)

                    # Dispatch Heavy (No Cap)
                    for sid, _ in heavy_samples:
                        await dispatch(sid, 'HEAVY')
                        
                    # Dispatch Fast (With Dynamic Cap)
                    for sid, _ in fast_samples:
                        await dispatch(sid, 'FAST', cap=dynamic_cap)

                # IMPORTANT: Wait for batch to fill, do not continue individual processing
                return 

            # 2. Handle Rest Completion (Reassembly)
            is_rest = rollout_sample.rollout_status.get('is_rest', False)
            if is_rest:
                if rollout_sample.sample_id in self.sample_buffer:
                    stored_data = self.sample_buffer.pop(rollout_sample.sample_id)
                    probe_ret = stored_data['probe_ret']
                    
                    # Concatenate Probe (1) + Rest (N-1) -> Total (N)
                    # DataProto.concat expects a list
                    merged_batch = DataProto.concat([probe_ret, ret])
                    
                    # Update the sample with the merged batch
                    rollout_sample.full_batch = merged_batch
                    # Update status
                    rollout_sample.rollout_status['is_merged'] = True
                    print(f"[P-DSR] 📦 样本 {rollout_sample.sample_id} 拼接完成 (8 seqs). 准备发送给 Trainer.")
                else:
                    print(f"[P-DSR] ❌ Error: Buffer missing for Rest sample {rollout_sample.sample_id}")
                    return

            # ------------------------------------------------------------------
            # [日志统计] 抽离出的独立方法
            # ------------------------------------------------------------------
            await self._log_rollout_metrics(rollout_sample, priority)

            rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
                [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
            )
            rollout_sample.param_version = self.current_param_version
            rollout_sample.rollout_status = await self.get_statistics()
            rollout_sample.agent_loop_output_list = []

            success = await self.message_queue_client.put_sample(
                sample=ray.cloudpickle.dumps(rollout_sample),
                param_version=rollout_sample.param_version,
            )
            if success:
                self.total_generated_samples += 1
            else:
                self.dropped_stale_samples += 1
        else:
            rollout_sample.agent_loop_output_list = ret
            await self.cancel_queue.put(rollout_sample)

        self.processed_sample_count += 1

    async def _log_rollout_metrics(self, rollout_sample, priority):
        """Helper method to log length statistics and decoded responses."""
        try:
            # Use the merged full_batch to get all 8 responses
            if 'response_mask' in rollout_sample.full_batch.batch:
                response_mask = rollout_sample.full_batch.batch['response_mask']
                if hasattr(response_mask, 'cpu'):
                    response_mask = response_mask.cpu()
                lengths = response_mask.sum(dim=-1).tolist()
                
                # 1. Processing times (Note: full_batch might not have processing_times merged perfectly, 
                # but we try to get what we can or pad 0s)
                # For simplicity, we just log 0s if missing, as merging times is complex
                rounded_times = [0.0] * len(lengths)

                # 2. Calculate Conditional Long-tail Probability
                if len(lengths) > 1:
                    is_first_long = lengths[0] > self.long_tail_threshold
                    if is_first_long:
                        self.total_long_probes += 1
                        rest_lengths = lengths[1:]
                        long_rest_count = sum(1 for l in rest_lengths if l > self.long_tail_threshold)
                        match_ratio = long_rest_count / len(rest_lengths)
                        self.total_long_matches_sum += match_ratio
                        
                        avg_prob = (self.total_long_matches_sum / self.total_long_probes) * 100
                        print(f"[P-DSR Analysis] Sample: {rollout_sample.sample_id} | Probe is LONG. "
                              f"Conditional Prob (N-1 also long): {avg_prob:.2f}% (Total Long Probes: {self.total_long_probes})")

                # 3. Decode Text
                try:
                    responses_tensor = rollout_sample.full_batch.batch['responses']
                    if hasattr(responses_tensor, 'cpu'): responses_tensor = responses_tensor.cpu()
                    decoded_texts = self.tokenizer.batch_decode(responses_tensor, skip_special_tokens=True)
                except:
                    decoded_texts = ["<Decode Error>"] * len(lengths)

                # 4. Dynamic Filename
                import re
                model_path = self.config.actor_rollout_ref.model.path
                model_match = re.search(r'(\d+(?:\.\d+)?b)', model_path.lower())
                model_suffix = model_match.group(1) if model_match else "model"
                filename = f"rollout_length_stats_async_{model_suffix}.jsonl"
                filename_text = f"rollout_responses_async_{model_suffix}.jsonl"

                log_entry = {
                    "timestamp": time.time(),
                    "param_version": self.current_param_version,
                    "sample_id": rollout_sample.sample_id,
                    "n_responses": len(lengths),
                    "lengths": lengths,
                    "times": rounded_times,
                    "priority": priority
                }
                
                with open(filename, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    
                log_entry_text = {
                    "sample_id": rollout_sample.sample_id,
                    "responses": decoded_texts
                }
                with open(filename_text, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry_text, ensure_ascii=False) + "\n")
                    
        except Exception as e:
            print(f"[FullyAsyncRollouter] Failed to log stats: {e}")

    async def _streaming_generation_main(self):
        """The main entry method for stream processing"""

        if self.async_rollout_manager is None:
            await self._init_async_rollout_manager()

        # Start the streaming loop
        print(f"[FullyAsyncRollouter] Start streaming mode, maximum concurrent samples: {self.max_concurrent_samples}")

        # Start sample feed coroutine, streaming process coroutine
        self.feed_task = asyncio.create_task(self._feed_samples())
        self.processor_task = asyncio.create_task(self._processor_worker())

        try:
            # Wait for sample feed to complete
            # Use asyncio.wait to monitor all tasks. If processor exits early,
            # detect it instead of blocking on feed_task (it might be stuck on a full queue).
            done, pending = await asyncio.wait(
                [self.feed_task, self.processor_task], return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    raise task.exception()

            if self.feed_task not in done:
                raise RuntimeError("Processor task exited prematurely")

            print("[FullyAsyncRollouter] Sample feed completed")

            # Wait for streaming to complete
            await self.processor_task
            print("[FullyAsyncRollouter] Streaming process completed")

        except Exception as e:
            import traceback
            print(f"[FullyAsyncRollouter] Streaming process exception: {repr(e)}")
            print(traceback.format_exc())

        finally:
            if self.processor_task:
                self.processor_task.cancel()

            await asyncio.gather(self.processor_task, return_exceptions=True)

        # Send a finish signal
        await self.message_queue_client.put_sample(
            sample=None,
            param_version=self.current_param_version,
        )

        async with self.lock:
            self.running = False

    async def fit(self):
        """
        Start the async rollouter - entry point that sets up and runs async tasks
        Main async fit method that coordinates all coroutines
        """

        print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

        if self.message_queue_client is None:
            raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

        # Set the running status flag
        async with self.lock:
            self.paused = False
            self.running = True

        # Create the main asynchronous task
        generation_task = asyncio.create_task(self._streaming_generation_main())
        monitor_task = asyncio.create_task(self._async_monitor_loop())

        try:
            # Run build and monitoring tasks concurrently
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
        except Exception as e:
            print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
        finally:
            if not generation_task.done():
                generation_task.cancel()
            if not monitor_task.done():
                monitor_task.cancel()

            # Wait for the task to complete
            await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        print("[FullyAsyncRollouter] Rollouter fit completed")

    async def _async_monitor_loop(self):
        """
        Async coroutine for monitoring:
        Function 1: Log information output
        Function 2: Trigger rollout recovery
        """
        last_stats_time = time.time()
        stats_interval = 60.0
        check_interval = 10.0

        while True:
            async with self.lock:
                if not self.running:
                    break
            await asyncio.sleep(check_interval)
            # Print statistics periodically
            current_time = time.time()
            if current_time - last_stats_time >= stats_interval:
                stats = await self.get_statistics()
                print(f"[FullyAsyncRollouter][MonitorLoop][Statistics] {pformat(stats)}")
                last_stats_time = current_time

            # Trigger rollout recovery
            if self.monitor_loop_trigger:
                if not await self._should_pause_generation():
                    async with self.lock:
                        self.paused = False
                        self.condition.notify_all()

    async def _should_pause_generation(self) -> bool:
        """Determine whether the build should be paused"""
        queue_stats = self.message_queue_client.get_statistics_sync()
        queue_size = queue_stats["queue_size"]

        if queue_size >= self.max_queue_size:
            if not self.paused:
                print(
                    f"[FullyAsyncRollouter][ShouldPause]  "
                    f"due to full queue: size={queue_size}, max={self.max_queue_size}"
                )
            return True

        if self.staleness_samples >= self.max_required_samples:
            if not self.paused:
                print(
                    "[FullyAsyncRollouter][ShouldPause] "
                    f"due to "
                    f"staleness_samples {self.staleness_samples} >= max_required_samples {self.max_required_samples} "
                )
            return True

        return False

    async def pause(self):
        """pause rollout"""
        print("[FullyAsyncRollouter][Public][Pause]")
        async with self.lock:
            self.paused = True
            # Cancel all rollout tasks
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.cancel()
            if self.active_tasks:
                await asyncio.gather(*self.active_tasks, return_exceptions=True)
                self.active_tasks.clear()
                print("[FullyAsyncRollouter][Public][Pause] All active tasks completed")
            await self.async_rollout_manager.clear_kv_cache()
            self.monitor_loop_trigger = False

    async def resume(self, dependency_ref: ObjectRef = None):
        if dependency_ref is not None:
            ray.get(dependency_ref)
        print("[FullyAsyncRollouter][Public][Resume]")
        async with self.lock:
            if self.config.async_training.partial_rollout:
                await self.async_rollout_manager.resume()
            self.paused = False
            self.monitor_loop_trigger = True
            self.condition.notify_all()

    async def get_statistics(self) -> dict:
        queue_stats = self.message_queue_client.get_statistics_sync()
        
        # P-DSR Debug: Aggregator State
        agg_stats = {k: len(v) for k, v in self.batch_aggregator.items()}

        stats = {
            # monitor stats
            "monitor/active_tasks_size": len(self.active_tasks),
            "monitor/queue/pending_queue_size": self.probe_queue.qsize(), # Use probe_queue for stats
            "monitor/queue/re_queue_size": self.re_queue.qsize(), # Add re_queue stats
            "monitor/queue/cancel_queue_size": self.cancel_queue.qsize(),
            "monitor/queue/mq_queue_size": queue_stats["queue_size"],
            "monitor/aggregator_state": str(agg_stats), # Log aggregator state
            # counting stats
            "count/current_param_version": self.current_param_version,
            "count/total_generated_samples": self.total_generated_samples,
            "count/staleness_samples": self.staleness_samples,
            "count/dropped_stale_samples": self.dropped_stale_samples,
            # static stats
            "static/max_required_samples": self.max_required_samples,
            "static/required_samples": self.required_samples,
            "static/staleness_threshold": self.staleness_threshold,
            "static/max_queue_size": self.max_queue_size,
            "static/max_concurrent_samples": self.max_concurrent_samples,
        }

        return stats