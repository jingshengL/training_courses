#!/usr/bin/env bash
# 按课程顺序 [1] -> [9] 跑完整套教程。
# 全程离线、无需数据集；CPU 可跑，有 GPU 更快（第 4 课训练最耗时）。
# 想单步学习就按 README 第 1 节逐个执行。
set -e
cd "$(dirname "$0")"

echo "========== [1] 量化数学 =========="
python 01_quant_math.py

echo "========== [2] 数值格式 =========="
python 02_number_formats.py

echo "========== [3] PTQ / QAT / QLoRA =========="
python 03_ptq_qat_qlora.py

echo "========== [4] 被试模型（训练迷你 GPT） =========="
python 04_tiny_llm_train.py

echo "========== [5] 激活离群与破解 =========="
python 05_llm_activation.py

echo "========== [6] weight-only 算法 =========="
python 06_weight_only.py

echo "========== [7] KV cache =========="
python 07_kv_cache.py

echo "========== [8] 整网量化实战 =========="
python 08_llm_e2e.py

echo "========== [9] 真实打包与 GGUF 风格 =========="
python 09_deploy_formats.py

echo
echo "全部完成，图表在 outputs/，权重在 ckpt/。"
