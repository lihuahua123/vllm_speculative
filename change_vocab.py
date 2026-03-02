import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. 设定路径
original_model_path = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct"  # 或者你本地的路径
save_path = "/root/autodl-tmp/Qwen2.5-0.5B-Instruct-152064"        # 新模型的保存路径
target_vocab_size = 152064                          # 目标 32B 的词表大小

print(f"正在加载原始模型: {original_model_path} ...")
# 注意：这里加载时不需要 trust_remote_code=True，除非你用的是很旧的版本
model = AutoModelForCausalLM.from_pretrained(
    original_model_path, 
    torch_dtype="auto", 
    device_map="cpu"  # 0.5B 很小，CPU 处理即可，防止显存碎片
)
tokenizer = AutoTokenizer.from_pretrained(original_model_path)

current_vocab_size = model.config.vocab_size
print(f"当前词表大小: {current_vocab_size}")

if current_vocab_size == target_vocab_size:
    print("词表大小已经一致，无需修改。")
else:
    print(f"正在将词表从 {current_vocab_size} 扩展到 {target_vocab_size} ...")
    
    # 核心步骤：调整 Embedding 和 lm_head 的权重形状
    # Transformers 会自动用 0 或随机噪声填充新增的部分
    # 对于投机采样，这些新增的 Token 永远不会被 Draft 预测出来，所以不影响精度
    model.resize_token_embeddings(target_vocab_size)
    
    # 强制更新 config 中的 vocab_size
    model.config.vocab_size = target_vocab_size
    
    print(f"正在保存修改后的模型到: {save_path} ...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("✅ 完成！现在可以用这个新路径作为 Draft Model 了。")