from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.modifiers.transform.awq import AWQModifier

MODEL_ID = "Qwen/Qwen3-0.6B"

# 加载模型与 tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# AWQ recipe：AWQModifier 负责 activation-aware 缩放，
# QuantizationModifier 负责按 W4A16 量化权重（group_size 默认 128）
recipe = [
    AWQModifier(), #duo_scaling="both"),
    QuantizationModifier(
        ignore=["lm_head"],        # lm_head 不量化，保证输出层精度
        scheme="W4A16_ASYM",
        targets=["Linear"],
    ),
]

# 一键量化（用内置 perfectblend 校准集）
oneshot(
    model=model,
    dataset="ultrachat-200k",
    splits="train_sft[:128]",
    recipe=recipe,
    max_seq_length=1024,
    num_calibration_samples=128,
)

# 保存为 compressed-tensors 格式
SAVE_DIR = "Qwen3-0.6B-awq"
model.save_pretrained(SAVE_DIR, save_compressed=False) # maos Flase, Cuda True
tokenizer.save_pretrained(SAVE_DIR)
print(f"量化模型已保存到 {SAVE_DIR}")


