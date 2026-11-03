# 第二次实验：找回的补充评估资料（不是完整验证通过）

本补充包从此前提供的 `third_experiment` 数据包中提取，与本次 `新实验 2.zip` 中的原始图像和模型进行独立交叉校验。原压缩包未修改。

## 资料来源与范围

- `dataset/manifest.csv`：保留第三次实验包中的 **原始字节**。SHA-256 等于第二次实验 `prepare_dataset.py` 和 `verify_results.py` 指定的固定清单 SHA-256：`80960c1cb7d2b5388694864e2b1c076a7d35a72ee7edb609a22162c5a11b6b8a`。
- `second_reference/metrics.csv`：第三次实验期间，以**第二次实验原始保存模型**在相同 887 张独立于训练受试者的测试图像上重新评估得到的六组指标表，保留原始 CSV 字节。这不是第二次实验原来生成但丢失的指标表。
- `second_reference/raw/` 与 `second_reference/processed/`：六组逐图预测记录的**公开路径版本**。只把 `sample_path` 中的本地绝对路径改为 `selected/类别/文件名`；原始数值、标签、图像/受试者 ID、分数和哈希不变。原始 CSV 仍保存在第三次实验包中。
- `provenance_public.json`：记录原第二次模型哈希与本次交叉核对信息，去除了原始 JSON 中的私人 Windows 路径。
- `audit_checks.json`：各模型的指标重算、受试者划分与来源完整性核对摘要。

## 关键边界

- 可以确认两次实验 5000 张 JPG 文件内容的 SHA-256 逐张一致；第三次实验的原始 `manifest.csv` 与第二次代码内的指定哈希一致。
- 第二次实验中缺失的 `dataset/summary.json`、`results/source_config.json`、原有训练期 `epoch_metrics.csv`、完整成功运行记录以及原始 `source_split_manifest.csv` **不能通过本补充包声称已经恢复**。
- 第三次实验另有完整的评估、配置信息，不能直接冒充第二次实验的训练配置；不要覆盖 `results/verification.json`，也不要把它修改成 `passed`。
- 所有这些结果都是 **同一 OASIS 数据集按受试者隔离的 held-out test**，不是独立医院或外部队列验证。
- 要提供给 GitHub，建议放入单独的 `Experiment_2/recovered_evaluation/`，并保留这份说明。请检查数据集的再分发许可，勿无说明上传本地路径记录。
