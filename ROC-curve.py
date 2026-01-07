import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

# ================== 全局字体设置（关键修改部分） ==================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,          # 全局基础字体
    'axes.titlesize': 20,     # 标题字体
    'axes.labelsize': 18,     # 坐标轴标签字体
    'xtick.labelsize': 15,    # x 轴刻度字体
    'ytick.labelsize': 15,    # y 轴刻度字体
    'legend.fontsize': 14,    # 图例字体
})

# ================== 预测图像基本路径配置 ==================
gt_dir = "IRSTD-PR/Label1"
pred_dirs = {
    "ACM": "IRSTD-PR/ACM1",
    "RDIAN": "IRSTD-PR/RDIAN1",
    "DNANet": "IRSTD-PR/DNANet1",
    "UIUNet": "IRSTD-PR/UIUNet1",
    "RPCANet": "IRSTD-PR/RPCANet1",
    "MSHNet": "IRSTD-PR/MSHNet1",
    "SCTransNet": "IRSTD-PR/SCTransNet1",
    "IDUNet": "IRSTD-PR/IDUNet1",
    "MPCNet": "IRSTD-PR/MPCNet1",
}

# ================== ROC 计算函数 ==================
def evaluate_ROC(pred_dir, gt_dir):
    y_true_all, y_score_all = [], []
    filenames = [f for f in os.listdir(gt_dir) if f.endswith(".png")]

    for filename in tqdm(filenames, desc=f"Evaluating {os.path.basename(pred_dir)}"):
        gt_path = os.path.join(gt_dir, filename)
        base = os.path.splitext(filename)[0].replace("_GT", "")
        possible_preds = [f for f in os.listdir(pred_dir) if base in f]
        if not possible_preds:
            continue

        pred_path = os.path.join(pred_dir, possible_preds[0])
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if gt is None or pred is None:
            continue

        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]))

        gt_bin = (gt >= 128).astype(np.uint8)
        pred_norm = np.clip(pred.astype(np.float32) / 255.0, 0, 1)

        if np.sum(gt_bin) == 0:
            continue

        y_true_all.extend(gt_bin.flatten())
        y_score_all.extend(pred_norm.flatten())

    y_true_all = np.array(y_true_all)
    y_score_all = np.array(y_score_all)

    fpr, tpr, _ = roc_curve(y_true_all, y_score_all)

    return fpr, tpr

# ================== 绘制 ROC 曲线 ==================
plt.figure(figsize=(8, 6))
colors = ['b', 'g', 'orange', 'purple', 'pink', 'brown', 'gray', 'yellow', 'red']

for i, (name, path) in enumerate(pred_dirs.items()):
    try:
        fpr, tpr = evaluate_ROC(path, gt_dir)
        plt.plot(
            fpr,
            tpr,
            linewidth=2.4,
            color=colors[i % len(colors)],
            label=name
        )
    except Exception as e:
        print(f"❌ {name} 绘图失败：{e}")

# ================== 图像美化 ==================
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC curve on the IRSTD-1K dataset", pad=12)

plt.xlim(0, 0.5e-3)
plt.ylim(0, 1)

plt.grid(True, linestyle="--", alpha=0.3)
plt.legend(loc="lower right", frameon=True)

plt.xticks(np.linspace(0, 0.5e-3, 6))
plt.yticks(np.linspace(0, 1, 11))

plt.tight_layout(pad=0.2)

# ================== 保存图像 ==================
plt.savefig("ROC curve on the IRSTD-1K dataset.png", dpi=300,
            bbox_inches='tight', pad_inches=0.05)
plt.savefig("ROC curve on the IRSTD-1K dataset.svg", format='svg',
            bbox_inches='tight', pad_inches=0.05)

plt.show()
