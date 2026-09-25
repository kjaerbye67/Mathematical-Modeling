% ============================================================
% 感知机二分类器
% 使用随机梯度下降学习线性决策边界
% 输入：
%   X     - 样本特征矩阵 [n_samples x n_features]
%   y     - 标签向量 [n_samples x 1]，取值 0 或 1
%   lr    - 学习率 (默认 0.01)
%   epochs- 最大训练轮数 (默认 100)
% 输出：
%   w     - 权重向量 [n_features x 1]
%   b     - 偏置
%   acc   - 训练准确率
%   history - 每轮训练的准确率记录
% ============================================================

function [w, b, acc, history] = perceptron_classifier(X, y, lr, epochs)
    if nargin < 3
        lr = 0.01;
    end
    if nargin < 4
        epochs = 100;
    end

    [n, d] = size(X);

    % 将标签 0/1 转换为感知机所需的 ±1 形式
    y_pm = 2 * y - 1;  % 0 -> -1, 1 -> +1

    % 初始化权重
    w = randn(d, 1) * 0.01;
    b = 0;

    history = zeros(epochs, 1);

    for epoch = 1:epochs
        % 随机打乱数据
        idx = randperm(n);
        X_shuffled = X(idx, :);
        y_shuffled = y_pm(idx);

        errors = 0;
        for i = 1:n
            x_i = X_shuffled(i, :)';
            y_i = y_shuffled(i);

            % 前向计算
            linear_out = w' * x_i + b;
            pred = sign(linear_out);
            if pred == 0
                pred = 1;  % 处理零值
            end

            % 若分类错误则更新权重
            if pred ~= y_i
                w = w + lr * y_i * x_i;
                b = b + lr * y_i;
                errors = errors + 1;
            end
        end

        % 计算本轮准确率
        y_pred_all = sign(X * w + b);
        y_pred_all(y_pred_all == 0) = 1;
        history(epoch) = mean(y_pred_all == y_pm);

        % 若全部分类正确则提前终止
        if errors == 0
            history(epoch+1:end) = history(epoch);
            break;
        end
    end

    % 最终准确率
    y_final = sign(X * w + b);
    y_final(y_final == 0) = 1;
    acc = mean(y_final == y_pm);
end
