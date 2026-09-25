% ============================================================
% 多层感知机（MLP）二分类器 — 单隐藏层
% 使用 BP 算法 + SGD 训练
% 输入：
%   X        - 样本特征矩阵 [n_samples x n_features]
%   y        - 标签向量 [n_samples x 1]，取值 0 或 1
%   hidden   - 隐藏层神经元个数 (默认 20)
%   lr       - 学习率 (默认 0.1)
%   epochs   - 训练轮数 (默认 500)
%   batch_size - mini-batch 大小 (默认 32)
% 输出：
%   W1, b1, W2, b2 - 网络参数
%   acc     - 训练准确率
%   loss_hist - 每轮平均损失
%   acc_hist  - 每轮准确率
% ============================================================

function [W1, b1, W2, b2, acc, loss_hist, acc_hist] = mlp_classifier(X, y, hidden, lr, epochs, batch_size)
    if nargin < 3
        hidden = 20;
    end
    if nargin < 4
        lr = 0.1;
    end
    if nargin < 5
        epochs = 500;
    end
    if nargin < 6
        batch_size = 32;
    end

    [n, d] = size(X);

    % 标签转为列向量
    y = y(:);

    % He 初始化（适用于 ReLU 类激活，这里用 sigmoid 则缩小）
    rng(42);
    W1 = randn(d, hidden) * sqrt(2 / d) * 0.5;
    b1 = zeros(1, hidden);
    W2 = randn(hidden, 1) * sqrt(2 / hidden) * 0.5;
    b2 = 0;

    loss_hist = zeros(epochs, 1);
    acc_hist  = zeros(epochs, 1);

    for epoch = 1:epochs
        % 随机打乱
        idx = randperm(n);
        X_shuf = X(idx, :);
        y_shuf = y(idx);

        epoch_loss = 0;
        num_batches = 0;

        % Mini-batch SGD
        for start = 1:batch_size:n
            finish = min(start + batch_size - 1, n);
            X_batch = X_shuf(start:finish, :);
            y_batch = y_shuf(start:finish);
            m_batch = size(X_batch, 1);

            % ---------- 前向传播 ----------
            % 隐藏层
            Z1 = X_batch * W1 + b1;          % [m x hidden]
            A1 = sigmoid(Z1);                 % [m x hidden]

            % 输出层
            Z2 = A1 * W2 + b2;                % [m x 1]
            A2 = sigmoid(Z2);                 % [m x 1]  预测概率

            % 二分类交叉熵损失
            eps_val = 1e-8;
            A2_clip = max(min(A2, 1 - eps_val), eps_val);
            loss = -mean(y_batch .* log(A2_clip) + (1 - y_batch) .* log(1 - A2_clip));

            % ---------- 反向传播 (BP) ----------
            % 输出层梯度
            dZ2 = A2 - y_batch;               % [m x 1]  交叉熵 + sigmoid 合并梯度
            dW2 = (A1' * dZ2) / m_batch;      % [hidden x 1]
            db2 = mean(dZ2);

            % 隐藏层梯度
            dA1 = dZ2 * W2';                  % [m x hidden]
            dZ1 = dA1 .* (A1 .* (1 - A1));    % sigmoid 导数
            dW1 = (X_batch' * dZ1) / m_batch; % [d x hidden]
            db1 = mean(dZ1, 1);               % [1 x hidden]

            % ---------- SGD 更新 ----------
            W2 = W2 - lr * dW2;
            b2 = b2 - lr * db2;
            W1 = W1 - lr * dW1;
            b1 = b1 - lr * db1;

            epoch_loss = epoch_loss + loss;
            num_batches = num_batches + 1;
        end

        loss_hist(epoch) = epoch_loss / num_batches;

        % 计算全训练集准确率
        y_pred_prob = forward_pass(X, W1, b1, W2, b2);
        y_pred = y_pred_prob >= 0.5;
        acc_hist(epoch) = mean(y_pred == y);
    end

    % 最终准确率
    y_pred_final = forward_pass(X, W1, b1, W2, b2) >= 0.5;
    acc = mean(y_pred_final == y);
end

%% ---------- 子函数 ----------

function A = sigmoid(Z)
    A = 1 ./ (1 + exp(-Z));
end

function A2 = forward_pass(X, W1, b1, W2, b2)
    % 对整个数据集做前向传播，返回预测概率
    A1 = sigmoid(X * W1 + b1);
    A2 = sigmoid(A1 * W2 + b2);
end
