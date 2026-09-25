"""
问题描述:
  心形线 (cardioid) 极坐标方程 r = 1 - sin(θ)
  在正方形区域 [-1.5, 1.5] × [-2, 1] 内均匀采样
  判断每个点是否位于心形内部 (内部为正类, 外部为负类)

  直角坐标判据: (x² + y² + y)² < x² + y²
  这个决策边界是非线性的，感知机无法有效分类
  而具有隐藏层的 MLP 可以通过非线性激活函数逼近该边界
================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from perceptron_classifier import perceptron_classifier
from mlp_classifier import mlp_classifier, forward_pass

# 设置中文字体（Windows 用 SimHei 或 Microsoft YaHei）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# %% ========== 1. 数据生成 ==========
n_samples = 2000
x_range = [-1.5, 1.5]
y_range = [-2.0, 1.0]

# 均匀采样
X_data = np.random.rand(n_samples, 2)
X_data[:, 0] = x_range[0] + (x_range[1] - x_range[0]) * X_data[:, 0]
X_data[:, 1] = y_range[0] + (y_range[1] - y_range[0]) * X_data[:, 1]

# 心形线判据: (x² + y² + y)² < x² + y²
x = X_data[:, 0]
yc = X_data[:, 1]
y_label = ((x**2 + yc**2 + yc)**2 < x**2 + yc**2).astype(int)

print("=" * 50)
print("  心形曲线二分类实验 (Python)")
print("=" * 50)
print(f"样本总数: {n_samples}")
print(f"正类 (内部): {np.sum(y_label)} ({100 * np.mean(y_label):.1f}%)")
print(f"负类 (外部): {np.sum(1 - y_label)} ({100 * (1 - np.mean(y_label)):.1f}%)")
print("决策边界方程: (x^2 + y^2 + y)^2 = x^2 + y^2")
print("=" * 50)

# 划分训练集 / 测试集 (80% / 20%)
idx_rand = np.random.permutation(n_samples)
n_train = int(0.8 * n_samples)
idx_train = idx_rand[:n_train]
idx_test = idx_rand[n_train:]

X_train = X_data[idx_train]
y_train = y_label[idx_train]
X_test = X_data[idx_test]
y_test = y_label[idx_test]

# 标准化特征
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0)
X_std[X_std < 1e-10] = 1.0

X_train_norm = (X_train - X_mean) / X_std
X_test_norm = (X_test - X_mean) / X_std

# %% ========== 2. 感知机训练 ==========
print("\n>>> 训练感知机模型...")

wp, bp, acc_p_train, hist_p = perceptron_classifier(
    X_train_norm, y_train, lr=0.01, epochs=200
)

# 测试集准确率
y_pred_p_test = np.sign(X_test_norm @ wp + bp)
y_pred_p_test[y_pred_p_test == 0] = 1
y_pred_p_test = (y_pred_p_test == 1).astype(int)
acc_p_test = np.mean(y_pred_p_test == y_test)

print(f"感知机 — 训练准确率: {acc_p_train * 100:.2f}%,  "
      f"测试准确率: {acc_p_test * 100:.2f}%")

# %% ========== 3. MLP 训练 (BP + SGD) ==========
print("\n>>> 训练 MLP 模型 (隐藏层 30 神经元, 500 轮)...")

W1, b1, W2, b2, acc_mlp_train, loss_hist, acc_hist = mlp_classifier(
    X_train_norm, y_train, hidden=30, lr=0.1, epochs=500, batch_size=32
)

# 测试集准确率
y_pred_mlp_test = (forward_pass(X_test_norm, W1, b1, W2, b2) >= 0.5).astype(int).ravel()
acc_mlp_test = np.mean(y_pred_mlp_test == y_test)

print(f"MLP — 训练准确率: {acc_mlp_train * 100:.2f}%,  "
      f"测试准确率: {acc_mlp_test * 100:.2f}%")


# %% ========== 辅助函数 ==========

def plot_decision_boundary(ax, X, y, X_mean, X_std, predict_fn, x_lim, y_lim, title):
    """画决策边界和样本点"""
    grid_size = 200
    xg = np.linspace(x_lim[0], x_lim[1], grid_size)
    yg = np.linspace(y_lim[0], y_lim[1], grid_size)
    Xg, Yg = np.meshgrid(xg, yg)
    X_grid = np.c_[Xg.ravel(), Yg.ravel()]

    y_grid = predict_fn(X_grid)

    # 画背景决策区域
    cmap_bg = ListedColormap(['#d0d0ff', '#ffd0d0'])
    ax.contourf(Xg, Yg, y_grid.reshape(grid_size, grid_size),
                levels=[-0.5, 0.5, 1.5], colors=['#d0d0ff', '#ffd0d0'], alpha=0.6)

    # 画样本点
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, c='red',
               alpha=0.6, label='Pos (Inside)')
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, c='blue',
               alpha=0.6, label='Neg (Outside)')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.axis('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def perceptron_predict(X_raw, w, b, X_mean, X_std):
    """感知机预测（含标准化）"""
    X_norm = (X_raw - X_mean) / X_std
    s = X_norm @ w + b
    return (s >= 0).astype(int)


def mlp_predict(X_raw, W1, b1, W2, b2, X_mean, X_std):
    """MLP 预测（含标准化）"""
    X_norm = (X_raw - X_mean) / X_std
    prob = forward_pass(X_norm, W1, b1, W2, b2)
    return (prob >= 0.5).astype(int).ravel()


# %% ========== 4. 结果可视化 ==========

# 理论心形边界
theta = np.linspace(0, 2 * np.pi, 500)
r_cardioid = 1 - np.sin(theta)
x_boundary = r_cardioid * np.cos(theta)
y_boundary = r_cardioid * np.sin(theta)

# ---- 图1: 6合1 总览 ----
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle("Cardioid Binary Classification: Perceptron vs MLP", fontsize=14, fontweight='bold')

# (1) 数据集分布
ax = axes[0, 0]
ax.scatter(X_data[y_label == 1, 0], X_data[y_label == 1, 1], s=6,
           c='red', alpha=0.5, label='Pos (Inside)')
ax.scatter(X_data[y_label == 0, 0], X_data[y_label == 0, 1], s=6,
           c='blue', alpha=0.5, label='Neg (Outside)')
ax.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='Theoretical boundary')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Heart-shaped curve dataset')
ax.legend(loc='best', fontsize=7)
ax.set_xlim(x_range); ax.set_ylim(y_range)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

# (2) 感知机决策边界
ax = axes[0, 1]
plot_decision_boundary(
    ax, X_test, y_test, X_mean, X_std,
    lambda Xr: perceptron_predict(Xr, wp, bp, X_mean, X_std),
    x_range, y_range,
    f'Perceptron (Test Acc: {acc_p_test * 100:.1f}%)'
)
ax.plot(x_boundary, y_boundary, 'k--', linewidth=1.5, alpha=0.7)

# (3) MLP 决策边界
ax = axes[0, 2]
plot_decision_boundary(
    ax, X_test, y_test, X_mean, X_std,
    lambda Xr: mlp_predict(Xr, W1, b1, W2, b2, X_mean, X_std),
    x_range, y_range,
    f'MLP (Test Acc: {acc_mlp_test * 100:.1f}%)'
)
ax.plot(x_boundary, y_boundary, 'k--', linewidth=1.5, alpha=0.7)

# (4) 感知机训练曲线
ax = axes[1, 0]
ax.plot(hist_p, 'b-', linewidth=1.2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.set_title('Perceptron training curve')
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# (5) MLP 损失和准确率曲线
ax = axes[1, 1]
color1 = 'tab:red'
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (Cross-Entropy)', color=color1)
ax.plot(loss_hist, color=color1, linewidth=1, alpha=0.8)
ax.tick_params(axis='y', labelcolor=color1)
ax2 = ax.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color2)
ax2.plot(acc_hist, color=color2, linewidth=1, alpha=0.8)
ax2.set_ylim(0, 1.05)
ax.set_title('MLP training curves')
ax.grid(True, alpha=0.3)

# (6) 模型对比柱状图
ax = axes[1, 2]
bar_data = np.array([[acc_p_train * 100, acc_mlp_train * 100],
                     [acc_p_test * 100, acc_mlp_test * 100]])
x_ticks = np.arange(2)
width = 0.30
bars1 = ax.bar(x_ticks - width / 2, bar_data[:, 0], width,
               color='steelblue', label='Perceptron')
bars2 = ax.bar(x_ticks + width / 2, bar_data[:, 1], width,
               color='coral', label='MLP')
ax.set_xticks(x_ticks)
ax.set_xticklabels(['Train', 'Test'])
ax.set_ylabel('Accuracy (%)')
ax.set_title('Perceptron vs MLP')
ax.set_ylim(0, 110)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 1.5,
            f'{h:.1f}%', ha='center', fontsize=9)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 1.5,
            f'{h:.1f}%', ha='center', fontsize=9)

plt.tight_layout()
fig1.savefig('experiment_fig1_overview.png', dpi=150, bbox_inches='tight')
print("\n图1 已保存: experiment_fig1_overview.png")


# ---- 图2: 测试集预测细节对比 ----
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
fig2.suptitle("Test Set Predictions: Perceptron vs MLP", fontsize=13, fontweight='bold')

for idx_model, (name, y_pred, acc, W1_, b1_, W2_, b2_) in enumerate([
    ('Perceptron', y_pred_p_test, acc_p_test, None, None, None, None),
    ('MLP', y_pred_mlp_test, acc_mlp_test, W1, b1, W2, b2)
]):
    ax = axes2[idx_model]

    # TP: 正类预测正确(绿)
    mask_tp = (y_test == 1) & (y_pred == 1)
    # TN: 负类预测正确(灰)
    mask_tn = (y_test == 0) & (y_pred == 0)
    # FN: 正类预测错误(红叉)
    mask_fn = (y_test == 1) & (y_pred == 0)
    # FP: 负类预测错误(红叉)
    mask_fp = (y_test == 0) & (y_pred == 1)

    ax.scatter(X_test[mask_tp, 0], X_test[mask_tp, 1], s=10,
               c='green', alpha=0.6, label='TP (Correct+)')
    ax.scatter(X_test[mask_tn, 0], X_test[mask_tn, 1], s=10,
               c='gray', alpha=0.5, label='TN (Correct-)')
    ax.scatter(X_test[mask_fn, 0], X_test[mask_fn, 1], s=35,
               c='red', marker='x', linewidth=1.5, label='FN (Miss)')
    ax.scatter(X_test[mask_fp, 0], X_test[mask_fp, 1], s=35,
               c='red', marker='x', linewidth=1.5, label='FP (False+)')

    ax.plot(x_boundary, y_boundary, 'k-', linewidth=2, label='True boundary')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.set_title(f'{name} (Test Acc: {acc * 100:.1f}%)')
    ax.legend(loc='best', fontsize=7)
    ax.set_xlim(x_range); ax.set_ylim(y_range)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig2.savefig('experiment_fig2_detail.png', dpi=150, bbox_inches='tight')
print("图2 已保存: experiment_fig2_detail.png")


# %% ========== 5. 打印实验结果汇总 ==========
print("\n" + "=" * 50)
print("           实验结果汇总")
print("=" * 50)
print(f"{'模型':<25} {'训练准确率':>12} {'测试准确率':>12}")
print(f"{'感知机':<25} {acc_p_train * 100:>11.2f}% {acc_p_test * 100:>11.2f}%")
print(f"{'MLP (30隐藏层)':<25} {acc_mlp_train * 100:>11.2f}% {acc_mlp_test * 100:>11.2f}%")
print("-" * 50)
print("分析: 感知机作为线性分类器，只能产生直线决策边界，")
print("无法拟合心形曲线的非线性边界，准确率受限。")
print("MLP 通过隐藏层的非线性激活函数组合，能够逼近任意")
print("复杂的决策边界，因此分类准确率显著优于感知机。")
print("=" * 50)

plt.show()
