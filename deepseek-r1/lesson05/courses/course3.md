另一种方法是**直接提示**。在这里，我们直接指示模型不仅要解决问题，还要明确地逐步展示其推理过程，然后验证其答案。

这样做的目的是鼓励学生采用更谨慎、更深思熟虑的方法来解决问题。


<img src='https://cdn-images-1.medium.com/max/4656/1*IYyk7UWgDNADFe_djWcXow.png' width=800px>

让我们为 “2 + 3 * 4 是多少？”制作一个提示，明确要求推理和验证。下面是 Python 代码的实际应用：
```python
# 直接提示示例
# 定义直接提示文本，要求模型解决问题、展示逐步推理过程并进行验证
direct_prompt_text = """
Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?
"""

# 调用之前定义的 generate_response 函数，使用直接提示文本让模型生成回复
model_response_direct = generate_response(direct_prompt_text)

# 打印直接提示文本
print("Direct Prompt:")
print(direct_prompt_text)
# 打印模型基于直接提示生成的回复
print("\nModel Response (Direct Prompting):")
print(model_response_direct)
```
直接提示输出非常容易理解，就是这个样子：
```
Direct Prompt:
Problem: Solve this, show reasoning step-by-step, and verify:
What is 2 + 3 * 4?

Model Response (Direct Prompting):
<|special_token|> Reasoning: To solve 2 + 3 * 4, I need to follow 
the order of operations, which states that multiplication should 
be done before addition.
Step 1: Multiply 3 by 4, which equals 12.
Step 2: Add 2 to the result from Step 1: 2 + 12 = 14.
Verification: To verify the answer, I can double-check the 
order of operations and the calculations. Multiplication is 
indeed performed before addition, and the calculations are correct.
<|special_token|> Summary: The answer is 14.
```
正如你所看到的，通过直接要求推理和验证，模型提供了一个更全面的输出，包括一个 “验证 ”部分。

这种方法直接引导模型产生我们所需的详细推理。
