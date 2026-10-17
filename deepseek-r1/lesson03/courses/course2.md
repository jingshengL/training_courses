
我们正在使用与 DeepSeek 用于 GRPO 算法的思维提示模板相同的模板来构建 R1 Zero，因此让我们来定义一下：

```python
# DeepSeek system prompt for GRPO based training
SYSTEM_PROMPT = (
  f"""A conversation between User and Assistant. The user asks a question, 
      and the Assistant solves it. The assistant
      first thinks about the reasoning process in the mind and 
      then provides the user with the answer. The reasoning
      process and answer are enclosed within <think> </think> 
      and <answer> </answer> tags, respectively, i.e., 
      <think> reasoning process here </think><answer> answer here </answer>
   """
)
```
这个**系统提示**告诉基本模型（Qwen/Qwen2.5-7B-Instruct）它的角色是一个在回答前逐步推理的有用助手。

思考 “和 ”回答 "标签用于组织模型的回答，将其内部推理与最终答案分开，以获得更好的评估和奖励。