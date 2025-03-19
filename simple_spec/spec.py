import heapq
from collections import deque
from typing import List, Tuple, Optional
import json
from transformers import PreTrainedTokenizerBase

class Request:
    def __init__(self, input_ids, max_output_len):
        self.input_ids = input_ids
        self.output_ids = []
        self.kv_cache = None
        self.alpha_history = deque(maxlen=20)  # 移动平均窗口
        self.proposed_lengths = []
        self.max_output_len = max_output_len

class SmartSpecScheduler:
    def __init__(self, target_model, draft_model=None, 
                 alpha=0.01, gamma=0.02, delta=0.005,
                 max_proposed_length=5):
        # 模型参数 (根据实际模型分析得到)
        self.alpha = alpha    # 上下文加载系数
        self.gamma = gamma    # 批处理计算系数
        self.delta = delta    # 模型加载系数
        
        self.target_model = target_model
        self.draft_model = draft_model
        self.max_proposed_length = max_proposed_length
        self.pending_requests = []
        self.running_requests = []
        
        # 维护token接受率历史
        self.global_alpha_history = deque(maxlen=100)
    
    def estimate_goodput(self, batch, proposed_lengths):
        """
        Algorithm 1: Goodput Estimation
        """
        # 计算移动平均token接受率
        if len(self.global_alpha_history) == 0:
            avg_alpha = 0.7  # 默认值
        else:
            avg_alpha = sum(self.global_alpha_history) / len(self.global_alpha_history)
        
        # 估计生成token数
        generated_tokens = 0
        for req, k in zip(batch, proposed_lengths):
            # 处理特殊情况：当avg_alpha接近1时
            if abs(1 - avg_alpha) < 1e-6:
                l = k + 1  # 当接受率接近1时，预期生成k+1个token
            else:
                l = (1 - avg_alpha**(k+1)) / (1 - avg_alpha)
            generated_tokens += l
        
        # 估计执行时间
        context_tokens = sum(len(req.input_ids)+len(req.output_ids) for req in batch)
        batched_tokens = sum(proposed_lengths)
        
        # 目标模型执行时间
        target_time = self.alpha*context_tokens + self.gamma*batched_tokens + self.delta
        
        # 草稿模型执行时间（如果使用）
        draft_time = 0
        if self.draft_model:
            draft_steps = max(proposed_lengths)
            for s in range(draft_steps):
                current_context = context_tokens + s
                current_batched = sum(k > s for k in proposed_lengths)
                draft_time += self.alpha*current_context + self.gamma*current_batched + self.delta
        
        total_time = draft_time + target_time
        
        return generated_tokens / total_time if total_time > 0 else 0
    
    def schedule_step(self):
        """
        Algorithm 2: 核心调度逻辑
        """
        if not self.pending_requests:
            return []
        
        best_goodput = -1
        best_batch = []
        best_proposals = []
        
        # 生成候选批次（简化实现）
        batch_candidates = []
        for batch_size in range(1, len(self.pending_requests)+1):
            candidate_batch = self.pending_requests[:batch_size]
            batch_candidates.append(candidate_batch)
        
        # 评估每个候选批次
        for batch in batch_candidates:
            # 枚举可能的推测长度 (0到max_proposed_length)
            for k in range(self.max_proposed_length+1):
                proposals = [k] * len(batch)
                
                # 检查KV缓存容量（简化实现）
                if not self.has_kv_cache_space(batch, proposals):
                    continue
                
                # 计算goodput
                current_goodput = self.estimate_goodput(batch, proposals)
                
                if current_goodput > best_goodput:
                    best_goodput = current_goodput
                    best_batch = batch
                    best_proposals = proposals
        
        # 执行最佳批次
        if best_batch:
            self.execute_batch(best_batch, best_proposals)
            return best_batch
        return []
    
    def execute_batch(self, batch, proposals):
        """修改执行批次的逻辑"""
        draft_outputs = []
        kv_cache_updates = []
        
        # 草稿模型生成阶段
        for req, k in zip(batch, proposals):
            remaining_tokens = req.max_output_len - len(req.output_ids)
            k = min(k, remaining_tokens)  # 确保不超过最大输出长度
            
            if k > 0:
                tokens, new_kv = self.draft_model.generate(
                    req.input_ids + req.output_ids,
                    max_length=k,
                    past_key_values=req.kv_cache
                )
                draft_outputs.append(tokens[:k])  # 限制长度
                kv_cache_updates.append(new_kv)
            else:
                draft_outputs.append([])
                kv_cache_updates.append(req.kv_cache)
        
        # 目标模型验证阶段
        accepted_lengths = []
        for req, draft_tokens, new_kv in zip(batch, draft_outputs, kv_cache_updates):
            verified, target_kv = self.target_model.verify(
                req.input_ids + req.output_ids,
                draft_tokens,
                past_key_values=req.kv_cache
            )
            
            # 更新请求状态
            verified = verified[:req.max_output_len - len(req.output_ids)]  # 确保不超过最大长度
            req.output_ids.extend(verified)
            req.kv_cache = target_kv
            
            # 更新接受率历史
            accept_rate = len(verified) / len(draft_tokens) if draft_tokens else 1.0
            req.alpha_history.append(accept_rate)
            self.global_alpha_history.append(accept_rate)
            
            accepted_lengths.append(len(verified))
        
        return accepted_lengths


    def _merge_kv_cache(self, draft_kv, target_kv):
        """合并草稿模型和目标模型的KV缓存（示例实现）"""
        # 实际需要根据模型结构实现缓存合并逻辑
        return target_kv  # 简化为使用目标模型的KV缓存
    
    def has_kv_cache_space(self, batch, proposals):
        """
        简化KV缓存检查
        """
        # 实际实现需要考虑具体缓存管理策略
        return True  # 暂时假设总有足够空间


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ModelWrapper:
    def __init__(self, model_path, draft=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.is_draft = draft
        
        # 配置生成参数（根据论文参数设置）
        self.generation_config = {
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": False,
            "temperature": 1.0,
            "top_k": 1
        }

    def verify_generate(self, input_ids, max_length, past_key_values=None):
        """验证模型生成实现（带KV缓存支持）"""
        with torch.no_grad():
            inputs = self._prepare_inputs(input_ids, past_key_values)
            
            # 使用更严格的生成配置，适合验证模型
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                **self.generation_config 
            )
            
            # 正确处理输出格式
            generated_sequence = outputs.sequences[0]  # 获取第一个序列
            input_length = inputs["input_ids"].shape[1]
            new_tokens = generated_sequence[input_length:].tolist()  # 只获取新生成的tokens
            
            return new_tokens, outputs.past_key_values
     
    def generate(self, input_ids, max_length, past_key_values=None):
        """草稿模型生成实现（带KV缓存支持）"""
        with torch.no_grad():
            inputs = self._prepare_inputs(input_ids, past_key_values)
            
            # 使用speculative decoding专用生成配置
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,    
                **self.generation_config
            )
            
            # 提取新生成的token IDs
            new_tokens = outputs[0, inputs["input_ids"].shape[1]:].tolist()
            return new_tokens, outputs.past_key_values

        

    def verify(self, input_ids, draft_tokens, past_key_values=None):
        """目标模型验证实现（带辅助头支持）"""
        with torch.no_grad():
            inputs = self._prepare_inputs(input_ids, past_key_values)
            
            # 启用辅助头进行快速验证
            outputs = self.model(
                **inputs,
                return_dict=True,
                output_hidden_states=False,
                use_cache=True
            )
            
            # 使用最后一个token的logits进行预测
            logits = outputs.logits[:, -1:, :]  # 只取最后一个位置的logits
            next_token = torch.argmax(logits, dim=-1)[0].item()
            
            # 验证草稿tokens
            verified_tokens = []
            current_input = input_ids.copy()
            
            for draft_token in draft_tokens:
                inputs = self._prepare_inputs(current_input, outputs.past_key_values)
                current_outputs = self.model(**inputs)
                pred_token = torch.argmax(current_outputs.logits[:, -1:], dim=-1)[0].item()
                
                if pred_token == draft_token:
                    verified_tokens.append(draft_token)
                    current_input.append(draft_token)
                else:
                    verified_tokens.append(pred_token)
                    break
            
            return verified_tokens, outputs.past_key_values

    
    def _prepare_inputs(self, input_ids, past_key_values):
        """准备模型输入（适配不同格式）"""
        if not isinstance(input_ids, torch.Tensor):
            input_tensor = torch.tensor([input_ids], device=self.device)
        else:
            input_tensor = input_ids.to(self.device)
            
        return {
            "input_ids": input_tensor,
            "past_key_values": past_key_values
        }

    def _get_accept_mask(self, logits, draft_tokens):
        """实现DeepSeek-R1的token接受逻辑（示例实现）"""
        # 实际实现需要根据模型具体结构调整
        # 这里简化展示逻辑：比较预测token与草稿token
        predicted_tokens = torch.argmax(logits[:, :-1], dim=-1)
        draft_tensor = torch.tensor([draft_tokens], device=self.device)
        return (predicted_tokens == draft_tensor).cpu().numpy()[0]

    def _process_accept_mask(self, draft_tokens, accept_mask, logits):
        """处理接受掩码并添加bonus token"""
        accepted = []
        for token, valid in zip(draft_tokens, accept_mask):
            if valid:
                accepted.append(token)
            else:
                break
        # 添加bonus token（如果适用）
        if len(accepted) == len(draft_tokens):
            accepted.append(torch.argmax(logits[-1]).item())
        return accepted

from transformers import AutoTokenizer

# 初始化目标模型和草稿模型
target_model = ModelWrapper("/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B")
draft_model = ModelWrapper("alamios/DeepSeek-R1-DRAFT-Qwen2.5-0.5B", draft=True)

# 初始化调度器
scheduler = SmartSpecScheduler(
    target_model=target_model,
    draft_model=draft_model,
    max_proposed_length=5
)

# 初始化tokenizer（使用目标模型的tokenizer）
tokenizer = AutoTokenizer.from_pretrained("/data/model/deepseek-aiDeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int] = None,
) -> List[Tuple[str, int, int, None]]:
    # Load the dataset.
    with open(dataset_path, encoding='utf-8') as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    # random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or (fixed_output_len is None and output_len < 4):
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len, None))

    return filtered_dataset
# return prompt, prompt_len, output_len
dataset = sample_sharegpt_requests("/data/sharegpt.json", 10, tokenizer)

requests = []
for prompt, prompt_len, output_len, _ in dataset[:1]:
    print(prompt, prompt_len, output_len)
    input_ids = tokenizer(prompt).input_ids
    requests.append(Request(input_ids=input_ids, max_output_len=output_len))

def generate_with_target_model(requests):
    # 执行调度步骤
    for step in range(10):  # 假设最多执行10个调度步骤
        #print(f"\nStep {step + 1}:")
    
        # 直接使用target_model生成
        for req in requests:
            if len(req.output_ids) >= req.max_output_len:
                continue
                
            new_tokens, new_kv = target_model.verify_generate(
                req.input_ids + req.output_ids,
                max_length=req.max_output_len - len(req.output_ids),
                past_key_values=req.kv_cache
            )
            
            req.output_ids.extend(new_tokens)
            req.kv_cache = new_kv
            
            # 打印当前输出
            output_text = tokenizer.decode(req.output_ids, skip_special_tokens=True)
            #print(f"Generated text: {output_text}")
        
        # 如果所有请求都完成，提前退出
        if all(len(req.output_ids) >= req.max_output_len for req in requests):
            break

    # 最终输出结果
    print("\nFinal Results:")
    for i, req in enumerate(requests):
        output_text = tokenizer.decode(req.output_ids, skip_special_tokens=True)
        print(f"Request {i + 1}: {output_text}")

def generate_with_draft_model(requests):
    # 添加请求到调度器
    scheduler.pending_requests = requests.copy()  # 使用副本避免修改原始请求
    
    # 执行调度步骤
    step = 0
    max_steps = 100  # 添加最大步数限制，避免死循环
    
    while scheduler.pending_requests and step < max_steps:
        step += 1
        print(f"Step {step}, Pending requests: {len(scheduler.pending_requests)}")
        
        # 调度并执行一个批次
        scheduled_batch = scheduler.schedule_step()
        
        # 更新pending_requests列表，移除已完成的请求
        scheduler.pending_requests = [
            req for req in scheduler.pending_requests 
            if len(req.output_ids) < req.max_output_len
        ]
        
        # 如果没有成功调度任何请求，可能出现了问题
        if not scheduled_batch:
            print("Warning: No requests were scheduled in this step")
            break
    
    if step >= max_steps:
        print(f"Warning: Reached maximum steps ({max_steps})")
    
    # 最终输出结果
    print("\nFinal Results:")
    for i, req in enumerate(requests):
        output_text = tokenizer.decode(req.output_ids, skip_special_tokens=True)
        print(f"Request {i + 1}: {output_text}")
    
generate_with_draft_model(requests)
# generate_with_target_model(requests)