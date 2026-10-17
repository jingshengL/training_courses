其中一种技巧是**长思维链（CoT）的少量提示**，在这种技巧中，我们尝试向 DeepSeek-V3 Base（或者在我们的案例中，Qwen2-0.5B）展示少量问题示例，并配以超级详细的逐步解决方案。这就是思维链（CoT）。


<img src='https://cdn-images-1.medium.com/max/4068/1*SAhvB0JqaK4d45IiIcj1Ow.png' width=800px>

这种方法的目标是让模型通过实例学习，并开始模仿这种彻底的推理方式。

对于我们的例题 “2 + 3 * 4 是多少？”，我们可以创建包含一些已解问题作为示例的提示。让我们看看在 Python 中是如何实现的：
```python
# 加载模型和分词器
# 定义要使用的模型名称
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# 从预训练模型中加载分词器
# trust_remote_code=True 表示信任远程代码，用于处理一些自定义代码
# padding_side="right" 表示在分词时从右侧进行填充
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
# 如果分词器没有设置填充标记（pad_token），则将其设置为结束标记（eos_token）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 从预训练模型中加载基于因果语言模型的模型
# trust_remote_code=True 表示信任远程代码
# torch_dtype=torch.bfloat16 表示使用 bfloat16 数据类型，可减少内存占用和加速计算
# .to("cuda" if torch.cuda.is_available() else "cpu") 将模型移动到可用的 GPU 上，如果没有则使用 CPU
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda" if torch.cuda.is_available() else "cpu")

# 生成详细的思维链响应
def generate_response(prompt_text):
    # 构建对话消息列表，包含系统提示和用户输入
    messages = [
        # 系统提示，告知模型要提供逐步的解决方案
        {"role": "system", "content": "You are a helpful assistant that provides step-by-step solutions."},
        # 用户输入的文本内容
        {"role": "user", "content": prompt_text}
    ]
    # 使用分词器的聊天模板处理消息列表，生成符合模型输入格式的文本
    # tokenize=False 表示不进行分词，仅生成文本
    # add_generation_prompt=True 表示添加生成提示
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # 对生成的文本进行分词，并转换为 PyTorch 张量
    # return_tensors="pt" 表示返回 PyTorch 张量
    # .to(model.device) 将分词后的输入数据移动到模型所在的设备上
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    # 使用模型进行文本生成
    # **inputs 解包输入参数
    # max_new_tokens=200 表示最多生成 200 个新标记
    # do_sample=False 表示不使用采样策略，生成结果是确定性的
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    # 将生成的标记解码为文本
    # skip_special_tokens=True 表示跳过特殊标记，只保留实际的文本内容
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 提取出助手的回复内容
    # 从解码后的文本中分割出以 "<|im_start|>assistant\n" 开头的最后一部分内容，并去除首尾空格
    return response.split("<|im_start|>assistant\n")[-1].strip()
```
让我们针对所提问题，对少样本进行相应的定义：
```python
# 示例问题及解决方案（使用 | special_token | 作为分隔符）
few_shot_prompt = """
# 第一个示例问题：求 9 的平方根加 5 的结果
Problem: What's the square root of 9 plus 5?
# 对应的解决方案，<|special_token|> 用于分隔解题步骤和总结
Solution: <|special_token|> First, find the square root of 9, which is 3. Then, add 5 to 3.  3 + 5 equals 8. <|special_token|> Summary: The answer is 8.

# 第二个示例问题：火车以每小时 60 英里的速度行驶 2 小时，求行驶的距离
Problem: Train travels at 60 mph for 2 hours, how far?
# 对应的解决方案，按照距离 = 速度 × 时间的公式计算
Solution: <|special_token|> Use the formula: Distance = Speed times Time. Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. <|special_token|> Summary: Train travels 120 miles.

# 第三个示例问题：计算 2 + 3 * 4 的值
Problem: What is 2 + 3 * 4?
Solution:
"""
```

现在，使用我们的基本模型，我们的样本集合看起来是这样的：
```python
# 利用少样本示例为目标问题生成回复
# 将少样本提示和目标问题拼接起来，形成完整的提示
target_problem_prompt = few_shot_prompt + "What is 2 + 3 * 4?"
# 调用之前定义的 generate_response 函数，使用拼接后的提示让模型生成回复
model_response_few_shot = generate_response(target_problem_prompt)

# 打印少样本提示内容
print("Few-shot Prompt:")
print(target_problem_prompt)
# 打印模型基于少样本提示生成的回复
print("\nModel Response (Few-shot CoT):")
print(model_response_few_shot)
```

它输出这些结构化数据：

```
Few-shot Prompt:
Problem: What's the square root of 9 plus 5?
Solution: <|special_token|> First, find the square root of 9, 
which is 3. Then, add 5 to 3.  3 + 5 equals 8. 
<|special_token|> Summary: The answer is 8.

Problem: Train travels at 60 mph for 2 hours, how far?
Solution: <|special_token|> Use the formula: Distance = Speed times Time. 
Speed is 60 mph, Time is 2 hours. Distance = 60 * 2 = 120 miles. 
<|special_token|> Summary: Train travels 120 miles.

Problem: What is 2 + 3 * 4?
Solution: 

Model Response (Few-shot CoT):
<|special_token|> To solve 2 + 3 * 4, we need to follow the order 
of operations (PEMDAS/BODMAS). Multiplication should be performed 
before addition.
Step 1: Multiply 3 by 4, which equals 12.
Step 2: Add 2 to the result from Step 1: 2 + 12 = 14.
<|special_token|> Summary: The answer is 14.
```

看看模型在看到示例后，是如何开始用 <|special_token|> 分隔符构建答案结构，并逐步推理出摘要和最终答案的！

->这就是少样本学习引导模型实现理想输出格式的威力。