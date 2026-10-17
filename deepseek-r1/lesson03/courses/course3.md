现在，我们已经准备好了系统提示，需要根据模板转换训练数据。

![Preprocessing dataset overview (Created by [Fareed Khan](undefined))](https://cdn-images-1.medium.com/max/6160/1*XnM7v4dPD4LtyAh2MLuInA.png)

我们需要创建 make_conversation 函数，它将为我们处理对话。
```python

# 构造训练数据的函数
def make_conversation(example):
    """将数据集示例转换为对话格式。"""
  return {
      "prompt": [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": example["problem"]},
      ],
  }
```

它将从我们的训练数据集中获取每个问题列的值，并返回一个包含系统提示和每行附加问题的字典。让我们创建这个函数来准备我们的数据集。
```python
# 加载并准备数据集
def load_math_dataset():
    """加载并准备数学数据集。"""
    dataset = load_dataset(
        "AI-MO/NuminaMath-TIR",
        name="default",
        split=['train', 'test']
    )
    
    # 将拆分转换为字典
    dataset = {
        'train': dataset[0],
        'test': dataset[1]
    }
    
    # 应用对话格式
    for split in dataset:
        dataset[split] = dataset[split].map(make_conversation)

        # 如果存在“messages”列，则删除
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")
    
    return dataset
```
一切准备就绪，让我们把训练数据转换成所需的格式，并打印出训练和测试大小。

```python
#加载我们的训练数据集并打印训练/测试大小
dataset = load_math_dataset()

print(f"Train set size: {len(dataset['train'])}")
print(f"Test set size: {len(dataset['test'])}")



#### 输出 ####
Train set size: 72441
Test set size: 99
#### 输出 ####
```
既然我们已经拆分了训练数据集，在进入下一步之前，我们需要验证数据集（**检查用户/助手对话是否存在**）。
```python
def validate_dataset(dataset):
    """对数据集执行基本的验证检查。"""
    
    # 定义数据集所需的字段
    required_fields = ["problem", "prompt"]

    # 遍历数据集的 'train' 和 'test' 划分
    for split in ['train', 'test']:
        print(f"\n正在验证 {split} 划分:")

        # 从数据集中获取列名
        fields = dataset[split].column_names

        # 检查是否有任何必需的字段缺失
        missing = [field for field in required_fields if field not in fields]
        if missing:
            print(f"警告: 缺失的字段: {missing}")  # 若字段缺失则发出警告
        else:
            print("✓ 所有必需字段均存在")  # 确认所有字段都存在

        # 从数据集划分中获取第一个样本
        sample = dataset[split][0]

        # 提取 'prompt' 字段，该字段包含一个消息列表
        messages = sample['prompt']

        # 验证提示格式：
        # - 它应该至少包含两条消息
        # - 第一条消息应该来自 'system' 角色
        # - 第二条消息应该来自 'user' 角色
        if (len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'):
            print("✓ 提示格式正确")  # 确认格式正确
        else:
            print("警告: 提示格式不正确")  # 若格式不正确则发出警告

# 验证数据集
validate_dataset(dataset)
```

它的输出是这样的：

```
验证列车拆分：

✓ 所有必填字段均已填写
✓ 提示格式正确

验证测试拆分：

✓ 所有必填字段均已填写
✓ 提示格式正确
```
我们的训练数据集验证成功🙌，这意味着我们已经成功地转换了训练数据集。
