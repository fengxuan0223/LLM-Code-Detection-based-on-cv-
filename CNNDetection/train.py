import os
import sys
import time
import torch
import torch.nn as nn
import argparse
from PIL import Image
from tensorboardX import SummaryWriter
import numpy as np
from collections import Counter
from torch.utils.data import DataLoader
from validate import validate
from data import create_dataloader
from networks.trainer import Trainer
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util import Logger

# test config
vals = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
multiclass = [1, 1, 1, 0, 1, 0, 0, 0]


# ============================================================
# 辅助函数定义区（在main()之前）
# ============================================================

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.phase = 'val'
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    return val_opt


def check_dataset(dataset, name="Dataset"):
    """检查数据集标签分布"""
    print(f"\n{'=' * 60}")
    print(f"[检查] {name}")
    print(f"{'=' * 60}")

    # 根据数据集类型提取标签
    if hasattr(dataset, 'samples'):
        labels = [dataset.samples[i][1] for i in range(len(dataset))]
    elif hasattr(dataset, 'fake_samples') and hasattr(dataset, 'real_samples'):
        labels = [1] * len(dataset.fake_samples) + [0] * len(dataset.real_samples)
    else:
        print(f"⚠️ 无法识别数据集类型: {type(dataset)}")
        return

    print(f"总样本数: {len(labels)}")
    print(f"标签为0的数量: {labels.count(0)}")
    print(f"标签为1的数量: {labels.count(1)}")
    print(f"标签分布: {Counter(labels)}")

    # 检查前10个样本
    print(f"\n前10个样本:")
    for i in range(min(10, len(dataset))):
        if hasattr(dataset, 'samples'):
            path, label = dataset.samples[i]
            print(f"  {i}: {os.path.basename(path)} → label={label}")
    print(f"{'=' * 60}\n")


def debug_model_output(model, val_loader, device):  # ✅ 添加device参数
    """分析模型输出分布"""
    model.eval()
    all_outputs = []

    print(f"\n{'=' * 60}")
    print("[DEBUG] 模型输出分析（前5个batch）")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 5:
                break

            # 适配不同的batch格式
            if isinstance(batch, dict):
                if 'input_ids' in batch:  # Transformer模型
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = model(input_ids, attention_mask)
                else:  # 图像模型
                    data = batch['img'].to(device)
                    outputs = model(data)
            else:  # tuple格式
                data, _ = batch
                data = data.to(device)
                outputs = model(data)

            outputs = outputs.squeeze()
            all_outputs.extend(outputs.cpu().numpy())

    print(f"  输出范围: [{np.min(all_outputs):.4f}, {np.max(all_outputs):.4f}]")
    print(f"  输出均值: {np.mean(all_outputs):.4f}")
    print(f"  输出标准差: {np.std(all_outputs):.4f}")
    print(f"  >0.5的比例: {np.mean(np.array(all_outputs) > 0.5):.4f}")
    print(f"{'=' * 60}\n")


def validate_with_debug(model, val_loader, device):  # ✅ 添加device参数
    """带调试信息的验证函数"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    criterion = nn.BCELoss()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            # 适配不同的batch格式
            if isinstance(batch, dict):
                if 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(input_ids, attention_mask)
                else:
                    data = batch['img'].to(device)
                    labels = batch['label'].to(device)
                    outputs = model(data)
            else:
                data, labels = batch
                data = data.to(device)
                labels = labels.to(device)
                outputs = model(data)

            outputs = outputs.squeeze()

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    # 调试输出
    print(f"\n[DEBUG 验证集分析]")
    print(f"  预测为1的比例: {np.mean(all_preds):.4f}")
    print(f"  真实标签1的比例: {np.mean(all_labels):.4f}")
    print(f"  准确率: {accuracy:.4f}")

    return avg_loss, accuracy


# ============================================================
# 主函数
# ============================================================

def main():
    """主训练流程"""

    # --------------------------------------------------
    # 1. 初始化配置和设备
    # --------------------------------------------------
    opt = TrainOptions().parse()
    use_fake = 'code_dataset' in opt.dataroot
    Testdataroot = os.path.join(opt.dataroot, 'test')

    # # ✅ 统一设备管理
    # if hasattr(opt, 'gpu_ids') and len(opt.gpu_ids) > 0:
    #     device = torch.device(f'cuda:{opt.gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cpu")

    print(f"\n{'=' * 60}")
    print(f"[设备信息] 使用设备: {device}")
    if device.type == 'cuda':
        print(f"[设备信息] GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"[设备信息] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")
    print(f"{'=' * 60}\n")

    Logger(os.path.join(opt.checkpoints_dir, opt.name, 'log.log'))
    print('  '.join(list(sys.argv)))

    val_opt = get_val_opt()
    Testopt = TestOptions().parse(print_options=False)

    # --------------------------------------------------
    # 2. 创建数据加载器
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("初始化训练集...")
    print("=" * 60)

    opt.phase = 'train'
    train_loader = create_dataloader(opt)
    print(f'#training samples = {len(train_loader.dataset)}')

    print(f"[DEBUG] train_loader.dataset 类型: {type(train_loader.dataset)}")
    if hasattr(train_loader.dataset, 'fake_samples'):
        print(f"[DEBUG] fake样本数: {len(train_loader.dataset.fake_samples)}")
    if hasattr(train_loader.dataset, 'real_samples'):
        print(f"[DEBUG] real样本数: {len(train_loader.dataset.real_samples)}")

    check_dataset(train_loader.dataset, "训练集")

    # 创建验证集
    print("\n" + "=" * 60)
    print("初始化验证集...")
    print("=" * 60)

    opt.phase = 'val'
    val_loader = create_dataloader(opt)
    print(f'#val samples = {len(val_loader.dataset)}')

    check_dataset(val_loader.dataset, "验证集")

    # --------------------------------------------------
    # 3. 初始化模型和日志
    # --------------------------------------------------
    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))

    model = Trainer(opt)

    # ✅ 确保模型在正确的设备上
    print(f"\n[模型] 将模型移动到 {device}")
    model.model = model.model.to(device)

    # ✅ 如果Trainer有optimizer，也需要更新
    if hasattr(model, 'optimizer'):
        # 重新初始化optimizer以适配新设备
        for state in model.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # --------------------------------------------------
    # 4. 定义测试函数
    # --------------------------------------------------
    def testmodel():
        print('*' * 25)
        accs = []
        aps = []
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

        for v_id, val in enumerate(vals):
            Testopt.dataroot = '{}/{}'.format(Testdataroot, val)
            Testopt.classes = os.listdir(Testopt.dataroot) if multiclass[v_id] else ['']
            Testopt.no_resize = True
            Testopt.no_crop = True

            acc, ap, _, _, _, _ = validate(model.model, Testopt)
            accs.append(acc)
            aps.append(ap)
            print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc * 100, ap * 100))

        print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(
            v_id + 1, 'Mean', np.array(accs).mean() * 100, np.array(aps).mean() * 100))
        print('*' * 25)
        print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    # --------------------------------------------------
    # 5. ⭐ 训练前检查模型输出分布
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("训练前模型输出检查")
    print("=" * 60)
    debug_model_output(model.model, val_loader, device)  # ✅ 传递device

    # --------------------------------------------------
    # 6. 训练循环
    # --------------------------------------------------
    model.train()
    print(f'\n当前工作目录: {os.getcwd()}')
    print(f"开始训练，共 {opt.niter} 个epoch\n")

    for epoch in range(opt.niter):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{opt.niter}")
        print(f"{'=' * 60}")

        epoch_start_time = time.time()
        epoch_iter = 0

        # 训练阶段
        for i, data in enumerate(train_loader):
            model.total_steps += 1
            epoch_iter += opt.batch_size

            # ✅ 确保数据在正确设备上（如果Trainer.set_input没有处理）
            if isinstance(data, dict):
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in data.items()}
            elif isinstance(data, (tuple, list)):
                data = [d.to(device) if isinstance(d, torch.Tensor) else d
                        for d in data]

            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                      f"Train loss: {model.loss:.4f} at step: {model.total_steps} lr {model.lr}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)

        # 学习率调整
        if epoch % opt.delr_freq == 0 and epoch != 0:
            print(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()),
                  f'Changing lr at the end of epoch {epoch}, iters {model.total_steps}')
            model.adjust_learning_rate()

        # --------------------------------------------------
        # 7. ⭐ 验证阶段（带调试信息）
        # --------------------------------------------------
        model.eval()
        val_opt.phase = 'val'

        # 使用原始validate函数
        val_loss, val_acc = validate(model.model, val_opt)

        # 或使用带调试的版本
        # val_loss, val_acc = validate_with_debug(model.model, val_loader, device)  # ✅ 传递device

        val_writer.add_scalar('loss', val_loss, epoch)
        val_writer.add_scalar('accuracy', val_acc, epoch)

        print(f"\n(Val @ epoch {epoch + 1}) val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

        # ⭐ 每5个epoch详细分析一次
        if (epoch + 1) % 5 == 0:
            debug_model_output(model.model, val_loader, device)  # ✅ 传递device

        model.train()

    # --------------------------------------------------
    # 8. 训练结束，最终测试
    # --------------------------------------------------
    model.eval()

    if not use_fake:
        print("\n" + "=" * 60)
        print("开始最终测试")
        print("=" * 60)
        testmodel()
    else:
        print("\n>>> Skip testmodel (使用FakeCodeDataset)")

    model.save_networks('last')

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # ============================================================
    # 程序入口
    # ============================================================

if __name__ == '__main__':
    main()
