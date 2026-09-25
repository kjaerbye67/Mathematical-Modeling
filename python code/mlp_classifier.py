"""
多层感知机二分类器 — 单隐藏层
使用 BP 算法 + Mini-batch SGD 训练，仅依赖 numpy

参数:
    X          - 样本特征矩阵 [n_samples x n_features]
    y          - 标签向量 [n_samples], 取值 0 或 1
    hidden     - 隐藏层神经元个数 (默认 20)
    lr         - 学习率 (默认 0.1)
    epochs     - 训练轮数 (默认 500)
    batch_size - mini-batch 大小 (默认 32)
返回:
    W1, b1, W2, b2 - 网络参数
    acc       - 训练准确率
    loss_hist - 每轮平均损失
    acc_hist  - 每轮准确率
"""

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def forward_pass(X, W1, b1, W2, b2):
    """对整个数据集做前向传播，返回预测概率"""
    A1 = sigmoid(X @ W1 + b1)
    A2 = sigmoid(A1 @ W2 + b2)
    return A2


def mlp_classifier(X, y, hidden=20, lr=0.1, epochs=500, batch_size=32):
    n, d = X.shape
    y = y.reshape(-1, 1)

    # He 初始化（针对 sigmoid 缩小到一半）
    np.random.seed(42)
    W1 = np.random.randn(d, hidden) * np.sqrt(2.0 / d) * 0.5
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, 1) * np.sqrt(2.0 / hidden) * 0.5
    b2 = 0.0

    loss_hist = np.zeros(epochs)
    acc_hist = np.zeros(epochs)

    for epoch in range(epochs):
        # 随机打乱
        idx = np.random.permutation(n)
        X_shuf = X[idx]
        y_shuf = y[idx]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch SGD
        for start in range(0, n, batch_size):
            finish = min(start + batch_size, n)
            X_batch = X_shuf[start:finish]
            y_batch = y_shuf[start:finish]
            m_batch = X_batch.shape[0]

            # ---------- 前向传播 ----------
            Z1 = X_batch @ W1 + b1                    # [m x hidden]
            A1 = sigmoid(Z1)                          # [m x hidden]
            Z2 = A1 @ W2 + b2                         # [m x 1]
            A2 = sigmoid(Z2)                          # [m x 1]

            # 二分类交叉熵损失
            eps = 1e-8
            A2_clip = np.clip(A2, eps, 1 - eps)
            loss = -np.mean(y_batch * np.log(A2_clip)
                            + (1 - y_batch) * np.log(1 - A2_clip))

            # ---------- 反向传播 (BP) ----------
            dZ2 = A2 - y_batch                        # [m x 1]  交叉熵+sigmoid 合并梯度
            dW2 = (A1.T @ dZ2) / m_batch              # [hidden x 1]
            db2 = np.mean(dZ2)

            dA1 = dZ2 @ W2.T                           # [m x hidden]
            dZ1 = dA1 * (A1 * (1 - A1))               # sigmoid 导数
            dW1 = (X_batch.T @ dZ1) / m_batch         # [d x hidden]
            db1 = np.mean(dZ1, axis=0, keepdims=True)  # [1 x hidden]

            # ---------- SGD 更新 ----------
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

            epoch_loss += loss
            num_batches += 1

        loss_hist[epoch] = epoch_loss / num_batches

        # 计算全训练集准确率
        y_pred_prob = forward_pass(X, W1, b1, W2, b2)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        acc_hist[epoch] = np.mean(y_pred == y)

    # 最终准确率
    acc = acc_hist[-1]

    return W1, b1, W2, b2, acc, loss_hist, acc_hist
