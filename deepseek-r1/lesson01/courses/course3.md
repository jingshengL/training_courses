虽然论文中没有明确说明用于 RL（Reinforcement Learning-强化学习） 预训练的初始数据集，但我们认为它应该以推理为主。

因此，为了尽可能接近原始复制，我们将使用这两个开源推理Hugging-Face数据集:

 1. [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR) (For R1 Zero Training)

 2. [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) (For R1 Training)

AI-MO/NuminaMath-TIR 包含 70K 道数学题，其中的信息栏显示了解题背后的 COT（推理链）。

| Field    | Description |  
|----------|------------|  
| problem  | The math problem |  
| solution | Step-by-step solution |  
| messages    | Chat to solve the problem |


样本如下：
```python
# 从 DigitalLearningGmbH 加载“AI-MO/NuminaMath-TIR”数据集
MATH_le = load_dataset("AI-MO/NuminaMath-TIR", "default")  

# 访问训练集中的第一个样本
MATH_le['train'][0]


#### 输出 ####
{
'problem': 'What is the degree of the polynomial 4 +5x^3 ... ',
'solution': 'This polynomial is not written in ...',
'messages': [{'from': 'user', 'value': 'The problem ...'}]
}
#### 输出 ####
```

Bespoke-Stratos 包含 17K 个数学和代码问题。

| Field        | Description |  
|-------------|------------|  
| system      | Guidelines for math and code problems |  
| conversation | Chat to solve the problem |

并且它的样本看起来像：
```python
# 从 bespokelabs 加载“Bespoke-Stratos-17k”数据集
bespoke_rl = load_dataset("bespokelabs/Bespoke-Stratos-17k", "default")

# 访问训练集中的第一个样本
bespoke_rl['train'][0]


#### 输出 ####
{
'system': 'Your role as an assistant involves ... ',
'conversations': [{'from': 'user', 'value': 'Return your ...'}]
}
 #### 输出 ####
```

您可以选择任何数据集，不一定非要选择这些数据集，只要以推理为重点（**问题逐步解决方案**）即可。
