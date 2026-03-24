# PIBT 蒸馏数据 · 远程训练说明（给朋友）

本文说明如何在收到压缩包后，在本地 GPU 上训练 / 微调 MAPF-GPT 2M 模型。假设压缩包内包含 **MAPF-GPT 代码** 与 **`dataset_pibt_48agents/`** 数据。

---

## 1. 发送方需要打包什么

### 1.1 代码（推荐：整仓库）

最省事且不易缺文件的做法：**打包整个 MAPF-GPT 仓库**（可删掉下面「不必带」的内容以减小体积）。

**必须包含的目录/文件（训练会用到）：**

| 路径 | 说明 |
|------|------|
| `train.py` | 训练入口 |
| `gpt/` | 模型、config、`MapfArrowDataset`、configurator |
| `tokenizer/` | `tokenizer.py`、`parameters.py`（构建词表维度） |
| `docker/requirements.txt` | Python 依赖 |

**建议一并带上：**

| 路径 | 说明 |
|------|------|
| `CLAUDE.md` / `README.md` | 环境与命令参考 |
| `gpt/config-2M.py` | 已在 `gpt/` 内，2M 默认超参 |

**可删以减小体积（对方不需要也能训）：**

- `out/`、`wandb/`、`__pycache__/`、`.pytest_cache/`
- `PIBT_Distill_data/`（若已只发 Arrow 数据集）
- `eval_configs/` 里特别大的附件、`svg/` 等
- `lacam/`、`pogema_toolbox/`（**仅训练 Arrow 数据时不必**；若对方要跑 `benchmark.py` / 完整评测再带上）

### 1.2 数据

需要目录结构（与你本地一致）：

```text
dataset_pibt_48agents/
  train/          # 若干 *.arrow
  val/            # 至少 1 个 *.arrow
```

数据体积极大时，可用 **`tar` + 分卷** 或网盘单独传，代码与数据分两个包也可以。

### 1.3 可选：初始权重

若希望从官方 2M 起步而不是随机初始化：

- 放 `weights/model-2M.pt`，或让对方第一次跑推理时由 `huggingface_hub` 自动下载（需联网）。

### 1.4 打包示例（在仓库**上一级**执行）

```bash
cd /path/to
tar --exclude='MAPF-GPT/.git' \
    --exclude='MAPF-GPT/out' \
    --exclude='MAPF-GPT/wandb' \
    --exclude='MAPF-GPT/__pycache__' \
    -czvf MAPF-GPT-pibt-training.tar.gz MAPF-GPT
```

仅数据：

```bash
tar -czvf dataset_pibt_48agents.tar.gz -C MAPF-GPT dataset_pibt_48agents
```

---

## 2. 接收方：环境准备

### 2.1 系统要求

- Linux + NVIDIA GPU + 驱动正常（`nvidia-smi` 可用）
- Python **3.10**（与上游一致）
- 建议 **CUDA 与 PyTorch 匹配**（按你机器安装的 PyTorch 官方说明选择）

### 2.2 解压

```bash
tar -xzvf MAPF-GPT-pibt-training.tar.gz
cd MAPF-GPT
```

确认存在：

```bash
ls train.py gpt/config-2M.py dataset_pibt_48agents/train dataset_pibt_48agents/val
```

### 2.3 虚拟环境与依赖

推荐使用 `uv`（或 `python -m venv`）：

```bash
uv venv --python 3.10
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r docker/requirements.txt
```

若安装 PyTorch 失败，请到 [pytorch.org](https://pytorch.org) 按 CUDA 版本安装 `torch`，再安装其余依赖。

---

## 3. 训练命令（单卡）

在 **`MAPF-GPT` 仓库根目录** 执行。

### 3.1 从官方 2M 权重微调（推荐）

需有 `weights/model-2M.pt` 或能联网下载 Hugging Face 权重（`train.py` 中 `pretrained` 逻辑会读该文件）。

```bash
python train.py gpt/config-2M.py \
  --train_data_file=dataset_pibt_48agents/train \
  --valid_data_file=dataset_pibt_48agents/val \
  --out_dir=out/finetune-pibt-2m \
  --init_from=pretrained \
  --pretrained_path=weights/model-2M.pt \
  --compile=False \
  --max_iters=50000 \
  --lr_decay_iters=50000 \
  --batch_size=1024 \
  --gradient_accumulation_steps=16
```

### 3.2 从对方提供的 `out/ckpt.pt` 继续训

若打包里包含已有 checkpoint：

```bash
python train.py gpt/config-2M.py \
  --train_data_file=dataset_pibt_48agents/train \
  --valid_data_file=dataset_pibt_48agents/val \
  --out_dir=out \
  --init_from=resume \
  --compile=False \
  --max_iters=60000 \
  --lr_decay_iters=60000 \
  --batch_size=1024 \
  --gradient_accumulation_steps=16
```

**重要：** `max_iters` 必须 **大于** checkpoint 里已保存的 `iter_num`，否则会很快退出（只评一次 / 几乎不再更新）。

### 3.3 显存与 batch

- `gpt/config-2M.py` 默认 `batch_size=4096` 很吃显存；**12GB 级 GPU** 建议 `batch_size=512`～`1024`，并相应增大 `gradient_accumulation_steps` 保持等效 batch 接近原设定。
- 若遇 OOM：先降 `batch_size`，再升 `gradient_accumulation_steps`。
- **`--compile=False`**：若 `torch.compile` + 分布式等组合报错，先关掉编译。

### 3.4 多卡（可选）

```bash
torchrun --standalone --nproc_per_node=4 train.py gpt/config-2M.py \
  --train_data_file=dataset_pibt_48agents/train \
  --valid_data_file=dataset_pibt_48agents/val \
  --out_dir=out/finetune-pibt-2m \
  --init_from=pretrained \
  --pretrained_path=weights/model-2M.pt \
  --compile=False \
  --max_iters=50000 \
  --lr_decay_iters=50000 \
  --batch_size=1024 \
  --gradient_accumulation_steps=16
```

注意：`MapfArrowDataset` 仅在训练数据路径名 **包含子串 `train`** 时会对 DDP 切分文件；`dataset_pibt_48agents/train` 满足该条件。

---

## 4. 输出与回传

- 训练过程中会往 **`--out_dir`**（默认 `out/`）写入 **`ckpt.pt`**。
- 请把 **`out/ckpt.pt`**（或整个 `out/`）发回；体积大可用网盘。
- 日志里会周期性打印 **`train loss` / `val loss`**，可作为是否过拟合的参考。

---

## 5. 常见问题

| 现象 | 可能原因 |
|------|----------|
| `IndexError` / 找不到 `.arrow` | `train/`、`val/` 路径错或目录为空；应在上述子目录下放好 shard。 |
| 立刻结束 | `resume` 时 `max_iters` ≤ 已有 `iter_num`；提高 `max_iters`。 |
| CUDA OOM | 减小 `batch_size`，增大 `gradient_accumulation_steps`。 |
| `torch.compile` 相关报错 | 加 `--compile=False`。 |

---

## 6. 版本与致谢

- 基于仓库内 `train.py`、`gpt/fast_data_loader.py` 与 `gpt/config-2M.py`。
- 数据格式：PyArrow，列 `input_tensors`、`gt_actions`，与上游 MAPF-GPT 训练管线一致。

若对方完全从零只训 Arrow、不跑 LaCAM / 评测，**不必**编译 `lacam/` 或安装完整仿真栈；仅需 Python 依赖与 GPU 即可。
