由于DeepSeek团队选择DeepSeek-V3作为创建R1 Zero和R1的基础模型，但它相当庞大，因为**685 GB的大小**显然不是我们所能企及的。

为了简单起见，我们将使用更小的基础模型 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)（0.9 GB 大小）。如果您的 GPU RAM 较大，甚至可以加载未量化的 LLM，那么您可以使用更大的模型，如 [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。

让我们来看看基本型号的一些规格：

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "data/Qwen-GRPO-training"  # 为了保存我们训练好的模型

# 如果输出目录不存在，则创建它
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 使用聊天模板初始化标记器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)

# 如果未设置，则设置 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocabulary size: {len(tokenizer)}")
print(f"Model max length: {tokenizer.model_max_length}")
print(f"Pad token: {tokenizer.pad_token}")
print(f"EOS token: {tokenizer.eos_token}")


#### 输出 ####
Vocabulary size: 151665
Model max length: 131072
Pad token: <|endoftext|>
EOS token: <|im_end|>
#### 输出 ####
```
以上是有关模型的一些基本信息，请看看我们的基本模型共有多少个参数。

```python
# 初始化基础模型
model = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME,
  trust_remote_code=True,
  torch_dtype=torch.bfloat16
)

print(f"Model parameters: {model.num_parameters():,}")


#### 输出 ####
Model parameters: 494,032,768
#### 输出 ####
```

接近 0.5B 的参数，让我们从中打印一个简单的响应，然后进入下一步。

```python
# 检查 CUDA 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 将模型移动到适当的设备
model.to(device)

# 测试基本推理
def test_model_inference(user_input: str):
    """Test basic model inference with the loaded model and tokenizer."""
    messages = [
        {"role": "system", "content": "You are Qwen, a helpful assistant."},
        {"role": "user", "content": user_input}
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 标记化并生成
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试模型
test_input = "how are you?"
response = test_model_inference(test_input)
print(f"Test Input: {test_input}")
print(f"Model Response: {response}")


#### 输出 ####
"Test Input: how are you?
Model Response: As an AI language model I dont have feelings ..."
#### 输出 ####
```
因此，这个微小模型的输出相当可靠，肯定适合我们的 DeepSeek lookalike 模型训练。