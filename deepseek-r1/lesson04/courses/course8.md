训练完成后，我们可以保存训练好的模型，用于推理。
```python
import os
from transformers import AutoTokenizer

# 定义训练好的模型的保存路径和模型名称
TRAINED_MODEL_PATH = "data/Qwen-GRPO-training"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 创建保存路径（如果不存在）
os.makedirs(TRAINED_MODEL_PATH, exist_ok=True)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right"
)

# 如果分词器的 pad_token 没有被设置，则将其设置为 eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 保存分词器到指定路径
tokenizer.save_pretrained(TRAINED_MODEL_PATH)

```

为了将其用于推理：

```python
# 测试训练好的模型的推理能力
def test_trained_model_inference(user_input: str):
    """使用加载的训练好的模型和分词器进行推理测试。"""
    # 构建对话消息列表
    messages = [
        # 系统提示信息，复用之前定义的系统提示
        {"role": "system", "content": SYSTEM_PROMPT}, 
        # 用户输入的内容
        {"role": "user", "content": user_input}
    ]

    # 使用分词器应用聊天模板
    # 将消息列表转换为符合模型输入要求的文本格式
    # tokenize=False 表示不进行分词，仅生成文本
    # add_generation_prompt=True 表示添加生成提示
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 对输入文本进行分词
    # return_tensors="pt" 表示返回 PyTorch 张量
    # .to(device) 将分词后的输入数据移动到指定设备（如 GPU）上
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # 使用训练好的模型生成输出
    outputs = trained_model.generate(
        # 解包输入参数
        **inputs,
        # 最大新生成的标记数量，这里设置为 200，可能会生成稍长的文本
        max_new_tokens=200, 
        # 是否进行采样，设置为 True 表示使用采样策略
        do_sample=True,
        # 采样温度，控制生成文本的随机性，值越大越随机
        temperature=0.7
    )

    # 将生成的标记解码为文本
    # skip_special_tokens=True 表示跳过特殊标记，只保留实际的文本内容
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 返回生成的文本响应
    return response
```

## R1 zero 的两个主要问题

现在，我们已经使用我们的基础模型 Qwen2-0.5B 代替他们的 DeepSeek V3（原始基础模型）完成了 R1 Zero 训练方法。

我们无法确定我们训练的模型存在哪些问题，但 DeepSeek 的研究人员发现，R1 Zero 模型在推理测试中表现非常出色，甚至在**AIME 2024**等任务中的得分与**OpenAI-01-0912**等更高级的模型相差无几。

这表明，在语言模型中使用强化学习（RL）来鼓励推理是一种很有前景的方法。

但他们也注意到DeepSeek-R1-Zero存在一些关键问题，需要在实际应用和更广泛的研究中加以解决。

![Problem with R1 Zero (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6378/1*_NdVhpb9cgT3-8o3Qn7mMA.png)

DeepSeek 的研究人员指出，该模板的*意图是简单和结构集中*。它*避免*对*推理过程本身*强加任何*特定内容*的限制。例如，它并没有说：

* 你*必须*使用分步推理”（它只说 “推理过程”，让模型来定义其含义）

* 你*必须*使用反思推理

* 你*必须*使用特定的问题解决策略

主要问题在于，“<思考>”标签内的推理过程很难读懂，人类很难理解和分析。

另一个问题是语言混合，当被问及多语言问题时，模型有时会在同一回答中混合使用多种语言，导致输出不一致和混乱。

如果你用西班牙语向它提问。突然之间，它的 “思维 ”就会变成**英语和西班牙语的杂乱混合体** ，并不完美！这些问题、混乱的推理和语言混淆是明显的障碍。
> 这就是他们将最初的 R1 zero模型转变为 R1 的两个主要原因
