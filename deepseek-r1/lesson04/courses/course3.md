格式奖励的目的是确保我们的模型遵循指令并正确构造其输出。我们要求它将推理放在“<思考>”标签中，将最终答案放在“<答案>”标签中，对吗？这个奖励函数正是要检查这一点！


<img src='https://cdn-images-1.medium.com/max/6620/1*DbUraziwiOoAj6SvtSJmpw.png' width=800px>

如果模型正确使用了这些标签，我们就给它 1 的奖励！这将鼓励模型关注我们想要的输出结构。

让我们来编写代码：
```python
# 实现格式奖励函数
def format_reward(completions, **kwargs):
    """
    此奖励函数用于检查完成结果是否符合特定格式：<think>...</think> <answer>...</answer>。
    """
    # 定义所需格式的正则表达式模式
    # 该模式要求字符串以 <think> 开头，中间可以包含任意字符（非贪婪匹配），接着是 </think>
    # 之后可以有零个或多个空白字符，然后是 <answer>，再接着是任意字符（非贪婪匹配），最后以 </answer> 结尾
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

    # 从每个完成结果中提取内容
    # completions 是一个列表，每个元素通常是一个列表，其中第一个元素是包含 "content" 键的字典
    # 这里通过列表推导式提取出每个完成结果的内容
    completion_contents = [completion[0]["content"] for completion in completions]

    # 检查每个完成结果的内容是否与定义的模式匹配
    # 使用 re.match 函数，结合 re.DOTALL 和 re.MULTILINE 标志
    # re.DOTALL 使得 . 可以匹配包括换行符在内的任意字符
    # re.MULTILINE 影响 ^ 和 $ 的匹配行为，使其可以匹配每行的开头和结尾
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
               for content in completion_contents]

    # 根据匹配结果给予奖励
    # 如果匹配成功，奖励为 1.0；否则奖励为 0.0
    return [1.0 if match else 0.0 for match in matches]
```

在此函数中

* 我们使用正则表达式（regex）定义一个模式。该模式的基本意思是 “内容应以 <think> 开始，在 </think>之前有 *anything* 内容，然后是一些 *spaces* ，接着是 <answer>，在 </answer>之前有 *anything* 内容，然后 *end* 到此为止”。

* 我们从每个完成模型中获取实际文本内容。

* re.DOTALL 可以帮助 regex 中的 .匹配换行符，re.MULTILINE 可以让 ^ 和 $ 匹配整个字符串的开始/结束，而不仅仅是行。

* 最后，如果与格式完全匹配，我们会给予 1 的奖励；如果不匹配，则给予 0 的奖励。这是对格式正确性的严格开/关奖励。