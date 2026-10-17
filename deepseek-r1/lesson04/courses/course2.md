精度奖励最容易理解，但需要的代码有点复杂。在这个奖励模型中，我们要检查我们的基础模型响应在数学上是否等同于地面实况解。


<img src='https://cdn-images-1.medium.com/max/7860/1*A3tW-OZSZ4m10EEzogjy8Q.png' width=800px>

如果模型答案在数学上是正确的，我们会给予**1.0**的奖励。如果答案不正确，奖励则为**0.0**。在无法解析地面实况解的情况下，我们会分配一个 **0.5** 的中性奖励，以避免不公平的惩罚。

现在，让我们来实现这个函数。
```python
def accuracy_reward(completions, solution, **kwargs):
    """
    奖励函数，用于检查模型的响应在数学上是否等同于真实答案。
    使用 latex2sympy2 进行解析，使用 math_verify 进行验证。
    """
    # 从 completions 中提取模型的响应内容
    # completions 是一个列表，其中每个元素是一个包含字典的列表，字典中有 "content" 键存储响应内容
    contents = [completion[0]["content"] for completion in completions]
    # 用于存储每个响应对应的奖励值的列表
    rewards = []

    # 遍历模型的响应内容和对应的真实答案
    for content, sol in zip(contents, solution):
        # 解析真实答案
        # 使用 parse 函数，以 "first_match" 模式提取，使用 LatexExtractionConfig 配置
        gold_parsed = parse(sol, extraction_mode="first_match", 
                            extraction_config=[LatexExtractionConfig()])
        # 检查真实答案是否解析成功
        if gold_parsed:
            # 以宽松的归一化配置解析模型的答案
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        # 配置归一化参数
                        normalization_config=NormalizationConfig(
                            # 不进行 nits 归一化
                            nits=False,
                            # 不处理格式错误的运算符
                            malformed_operators=False,
                            # 处理基本 LaTeX 表达式
                            basic_latex=True,
                            # 处理方程
                            equations=True,
                            # 处理所有带框的表达式
                            boxed="all",
                            # 处理单位
                            units=True,
                        ),
                        # 带框匹配的优先级
                        boxed_match_priority=0,
                        # 不尝试在没有锚点的情况下提取
                        try_extract_without_anchor=False,
                    )
                ],
                # 以 "first_match" 模式提取
                extraction_mode="first_match",
            )
            # 如果模型答案和真实答案匹配，奖励为 1.0，否则为 0.0
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # 如果真实答案无法解析，给予中性奖励 0.5
            reward = 0.5
            # 打印警告信息，提示真实答案解析失败
            print("Warning: Failed to parse gold solution:", sol)
        # 将当前响应的奖励值添加到奖励列表中
        rewards.append(reward)
    # 返回所有响应对应的奖励值列表
    return rewards
```
在这个函数中，我们检查模型回答**是否**等同于正确答案。与比较原始文本不同，我们

 1. 使用 **latex2sympy2** 将解答转换为结构化数学格式。

 2. 如果解析失败，则给予**0.5**的中性奖励。

 3. 提取模型输出并进行归一化处理，以提高稳健性。

 4. 使用 **math_verify** 检查解析后的响应是否与解析后的解决方案相匹配。

 5. 如果正确，则赋值为 **1** 如果不正确，则赋值为 **0**。

这确保了准确性评估不仅仅是文本相似性，而是**真正的数学正确性**。