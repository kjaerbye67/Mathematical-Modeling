"""
感知机二分类器
使用随机梯度下降学习线性决策边界

参数:
    X       - 样本特征矩阵 [n_samples x n_features]
    y       - 标签向量 [n_samples], 取值 0 或 1
    lr      - 学习率 (默认 0.01)
    epochs  - 最大训练轮数 (默认 100)
返回:
    w       - 权重向量 [n_features]
    b       - 偏置
    acc     - 训练准确率
    history - 每轮训练的准确率记录
"""

import numpy as np


def perceptron_classifier(X, y, lr=0.01, epochs=100):
    n, d = X.shape

    # 将标签 0/1 转换为感知机所需的 ±1 形式
    y_pm = 2 * y - 1  # 0 -> -1, 1 -> +1

    # 初始化权重
    w = np.random.randn(d) * 0.01
    b = 0.0

    history = np.zeros(epochs)

    for epoch in range(epochs):
        # 随机打乱数据
        idx = np.random.permutation(n)
        X_shuffled = X[idx]
        y_shuffled = y_pm[idx]

        errors = 0
        for i in range(n):
            x_i = X_shuffled[i]
            y_i = y_shuffled[i]

            # 前向计算
            linear_out = np.dot(w, x_i) + b
            pred = np.sign(linear_out)
            if pred == 0:
                pred = 1

            # 若分类错误则更新权重
            if pred != y_i:
                w += lr * y_i * x_i
                b += lr * y_i
                errors += 1

        # 计算本轮准确率
        y_pred_all = np.sign(X @ w + b)
        y_pred_all[y_pred_all == 0] = 1
        history[epoch] = np.mean(y_pred_all == y_pm)

        # 若全部分类正确则提前终止
        if errors == 0:
            history[epoch + 1:] = history[epoch]
            break

    # 最终准确率
    y_final = np.sign(X @ w + b)
    y_final[y_final == 0] = 1
    acc = np.mean(y_final == y_pm)

    return w, b, acc, history
