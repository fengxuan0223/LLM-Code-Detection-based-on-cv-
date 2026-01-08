1. **实验 1：Loss 曲线**
2. **实验 2：Backbone 对比**
3. **实验 3：Frozen vs Finetune CodeBERT**
4. **整理结果表 + 写实验分析**

------

# ✅ Step 1：完成实验 1 —— Loss 曲线



✅ **实验 1 到此为止，结束。**

------

# ✅ Step 2：实验 2 —— Backbone 对比（最关键）

1️⃣ SimpleMLP（baseline）

```bash
python train.py \
  --arch simplemlp \
  --dataroot ./code_dataset \
  --gpu_ids -1 \
  --niter 10 \
  --name exp_simplemlp
```

------

## 2️⃣ BiLSTM（如果你项目里有）

```bash
python train.py \
  --arch bilstm \
  --dataroot ./code_dataset \
  --gpu_ids -1 \
  --niter 10 \
  --name exp_bilstm
```

## 3️⃣ CodeBERT

```bash
python train.py \
  --arch codebert \
  --dataroot ./code_dataset \
  --gpu_ids -1 \
  --niter 10 \
  --name exp_codebert
```

------

# ✅ Step 3：实验 3 —— Frozen vs Finetune（加分项）

------



### 🔹 Frozen 版本

```python
for p in self.encoder.parameters():
    p.requires_grad = False
```

保存为：

```bash
--name exp_codebert_frozen
```

------

### 🔹 Finetune 版本（你现在这个）

```bash
--name exp_codebert_finetune
```

------


记录：

| Setting  | Val Acc | Val Loss |
| -------- | ------- | -------- |
| Frozen   | 0.6526  | 0.6440   |
| Finetune | 0.9356  | 0.1499   |

------


        )
        cls = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        logits = self.classifier(cls)
        return logits.squeeze(1)

    
```
