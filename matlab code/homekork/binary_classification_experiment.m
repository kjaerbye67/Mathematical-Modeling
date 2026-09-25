% 二分类问题: 心形曲线内外点分类 (Cardioid Classification)
% 问题描述:
%   心形线 (cardioid) 极坐标方程 r = 1 - sin(θ)
%   在正方形区域 [-1.5, 1.5] × [-2, 1] 内均匀采样
%   判断每个点是否位于心形内部 (内部为正类, 外部为负类)
%
%   直角坐标判据: (x² + y² + y)² < x² + y²
%   这个决策边界是非线性的，感知机无法有效分类
%   而具有隐藏层的 MLP 可以通过非线性激活函数逼近该边界
clear; clc; close all;
rng(42);

%% ========== 1. 数据生成 ==========
n_samples = 2000;                        % 样本量
x_range = [-1.5, 1.5];
y_range = [-2.0, 1.0];

% 均匀采样
X_data = rand(n_samples, 2);
X_data(:, 1) = x_range(1) + (x_range(2) - x_range(1)) * X_data(:, 1);
X_data(:, 2) = y_range(1) + (y_range(2) - y_range(1)) * X_data(:, 2);

% 心形线判据: (x² + y² + y)² < x² + y²
x = X_data(:, 1);
y_coord = X_data(:, 2);
y_label = ((x.^2 + y_coord.^2 + y_coord).^2 < x.^2 + y_coord.^2);

fprintf('============================================\n');
fprintf('  心形曲线二分类实验\n');
fprintf('============================================\n');
fprintf('样本总数: %d\n', n_samples);
fprintf('正类 (内部): %d (%.1f%%)\n', sum(y_label), 100*mean(y_label));
fprintf('负类 (外部): %d (%.1f%%)\n', sum(~y_label), 100*mean(~y_label));
fprintf('决策边界方程: (x^2 + y^2 + y)^2 = x^2 + y^2\n');
fprintf('============================================\n\n');

% 划分训练集 / 测试集 (80% / 20%)
idx_rand = randperm(n_samples);
n_train = round(0.8 * n_samples);
idx_train = idx_rand(1:n_train);
idx_test  = idx_rand(n_train+1:end);

X_train = X_data(idx_train, :);
y_train = y_label(idx_train);
X_test  = X_data(idx_test, :);
y_test  = y_label(idx_test);

% 标准化特征 (使均值为 0，标准差为 1)
X_mean = mean(X_train);
X_std  = std(X_train);
% 防止除零
X_std(X_std < 1e-10) = 1;

X_train_norm = (X_train - X_mean) ./ X_std;
X_test_norm  = (X_test  - X_mean) ./ X_std;

%% ========== 2. 感知机训练 ==========
fprintf('>>> 训练感知机模型...\n');

[wp, bp, acc_p_train, hist_p] = perceptron_classifier(X_train_norm, y_train, 0.01, 200);

% 测试集准确率
y_pred_p_test = sign(X_test_norm * wp + bp);
y_pred_p_test(y_pred_p_test == 0) = 1;
y_pred_p_test = (y_pred_p_test == 1);  % 转回 0/1
acc_p_test = mean(y_pred_p_test == y_test);

fprintf('感知机 — 训练准确率: %.2f%%, 测试准确率: %.2f%%\n', ...
    acc_p_train*100, acc_p_test*100);

%% ========== 3. MLP 训练 (BP + SGD) ==========
fprintf('\n>>> 训练 MLP 模型 (隐藏层 30 神经元, 500 轮)...\n');

[W1, b1, W2, b2, acc_mlp_train, loss_hist, acc_hist] = ...
    mlp_classifier(X_train_norm, y_train, 30, 0.1, 500, 32);

% 测试集准确率 (用 mlp_classifier 内部的 forward_pass)
y_pred_mlp_test = forward_pass_mlp(X_test_norm, W1, b1, W2, b2) >= 0.5;
acc_mlp_test = mean(y_pred_mlp_test == y_test);

fprintf('MLP — 训练准确率: %.2f%%, 测试准确率: %.2f%%\n', ...
    acc_mlp_train*100, acc_mlp_test*100);

%% ========== 4. 结果对比与可视化 ==========

% ---- 图1: 数据集分布 ----
figure('Position', [50, 100, 1600, 900]);

subplot(2, 3, 1);
scatter(X_data(y_label==1, 1), X_data(y_label==1, 2), 15, 'r', 'filled', ...
    'MarkerFaceAlpha', 0.6);
hold on;
scatter(X_data(y_label==0, 1), X_data(y_label==0, 2), 15, 'b', 'filled', ...
    'MarkerFaceAlpha', 0.6);

% 画出理论心形边界
theta = linspace(0, 2*pi, 500)';
r_cardioid = 1 - sin(theta);
x_boundary = r_cardioid .* cos(theta);
y_boundary = r_cardioid .* sin(theta);
plot(x_boundary, y_boundary, 'k-', 'LineWidth', 2);

xlabel('x_1'); ylabel('x_2');
title('心形曲线数据集 (红:内部 蓝:外部)');
legend({'正类 (内部)', '负类 (外部)', '理论边界'}, 'Location', 'best');
grid on;
axis equal;
xlim(x_range); ylim(y_range);

% ---- 图2: 感知机决策边界 ----
subplot(2, 3, 2);
plot_decision_boundary(X_test, y_test, X_mean, X_std, ...
    @(X) perceptron_predict(X, wp, bp), x_range, y_range);
title(sprintf('感知机决策边界 (测试准确率 %.1f%%)', acc_p_test*100));

% ---- 图3: MLP 决策边界 ----
subplot(2, 3, 3);
plot_decision_boundary(X_test, y_test, X_mean, X_std, ...
    @(X) forward_pass_mlp((X - X_mean)./X_std, W1, b1, W2, b2) >= 0.5, ...
    x_range, y_range);
title(sprintf('MLP 决策边界 (测试准确率 %.1f%%)', acc_mlp_test*100));

% ---- 图4: 感知机训练曲线 ----
subplot(2, 3, 4);
plot(hist_p, 'b-', 'LineWidth', 1.2);
xlabel('训练轮数'); ylabel('准确率');
title('感知机训练曲线');
ylim([0, 1.05]);
grid on;

% ---- 图5: MLP 损失曲线 ----
subplot(2, 3, 5);
yyaxis left;
plot(loss_hist, 'r-', 'LineWidth', 1);
xlabel('训练轮数'); ylabel('损失 (交叉熵)');
title('MLP 训练曲线');
grid on;

yyaxis right;
plot(acc_hist, 'b-', 'LineWidth', 1);
ylabel('准确率');
legend({'Loss', 'Accuracy'}, 'Location', 'best');

% ---- 图6: 模型对比柱状图 ----
subplot(2, 3, 6);
bar_data = [acc_p_train, acc_mlp_train; acc_p_test, acc_mlp_test] * 100;
b = bar(bar_data);
b(1).FaceColor = [0.3, 0.6, 0.9];
b(2).FaceColor = [0.9, 0.4, 0.3];
set(gca, 'XTickLabel', {'训练集', '测试集'});
ylabel('准确率 (%)');
title('感知机 vs MLP 准确率对比');
legend({'感知机', 'MLP'}, 'Location', 'southeast');
ylim([0, 105]);
grid on;

% 在柱状图上标注数值
for i = 1:2
    for j = 1:2
        text(i + (j-1.5)*0.22, bar_data(i, j) + 1, ...
            sprintf('%.1f%%', bar_data(i, j)), ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
    end
end

sgtitle(sprintf('Cardioid Binary Classification: Perceptron vs MLP'), ...
    'FontSize', 13, 'FontWeight', 'bold');

% 保存图1
print(gcf, 'experiment_fig1_overview.png', '-dpng', '-r150');

% ---- 图2: 单独展示对比细节 ----
figure('Position', [100, 150, 1200, 450]);

% 感知机测试结果细览
subplot(1, 2, 1);
scatter(X_test(y_test==1 & y_pred_p_test==1, 1), X_test(y_test==1 & y_pred_p_test==1, 2), ...
    15, [0 0.6 0], 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
scatter(X_test(y_test==1 & y_pred_p_test==0, 1), X_test(y_test==1 & y_pred_p_test==0, 2), ...
    40, 'r', 'x', 'LineWidth', 1.5);
scatter(X_test(y_test==0 & y_pred_p_test==0, 1), X_test(y_test==0 & y_pred_p_test==0, 2), ...
    15, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.7);
scatter(X_test(y_test==0 & y_pred_p_test==1, 1), X_test(y_test==0 & y_pred_p_test==1, 2), ...
    40, 'r', 'x', 'LineWidth', 1.5);
plot(x_boundary, y_boundary, 'k-', 'LineWidth', 2);
xlabel('x_1'); ylabel('x_2');
title(sprintf('感知机预测结果 (绿色:正确  红叉:错误)  准确率: %.1f%%', acc_p_test*100));
legend({'TN', 'FN (漏检)', 'TN', 'FP (误报)', '理论边界'}, 'Location', 'best');
grid on; axis equal; xlim(x_range); ylim(y_range);

% MLP 测试结果细览
subplot(1, 2, 2);
scatter(X_test(y_test==1 & y_pred_mlp_test==1, 1), X_test(y_test==1 & y_pred_mlp_test==1, 2), ...
    15, [0 0.6 0], 'filled', 'MarkerFaceAlpha', 0.7);
hold on;
scatter(X_test(y_test==1 & y_pred_mlp_test==0, 1), X_test(y_test==1 & y_pred_mlp_test==0, 2), ...
    40, 'r', 'x', 'LineWidth', 1.5);
scatter(X_test(y_test==0 & y_pred_mlp_test==0, 1), X_test(y_test==0 & y_pred_mlp_test==0, 2), ...
    15, [0.6 0.6 0.6], 'filled', 'MarkerFaceAlpha', 0.7);
scatter(X_test(y_test==0 & y_pred_mlp_test==1, 1), X_test(y_test==0 & y_pred_mlp_test==1, 2), ...
    40, 'r', 'x', 'LineWidth', 1.5);
plot(x_boundary, y_boundary, 'k-', 'LineWidth', 2);
xlabel('x_1'); ylabel('x_2');
title(sprintf('MLP 预测结果 (绿色:正确  红叉:错误)  准确率: %.1f%%', acc_mlp_test*100));
legend({'TP', 'FN (漏检)', 'TN', 'FP (误报)', '理论边界'}, 'Location', 'best');
grid on; axis equal; xlim(x_range); ylim(y_range);

sgtitle('Test Set Prediction Results: Perceptron vs MLP');

% 保存图2
print(gcf, 'experiment_fig2_detail.png', '-dpng', '-r150');

%% ========== 5. 打印实验结果汇总 ==========
fprintf('\n============================================\n');
fprintf('           实验结果汇总\n');
fprintf('============================================\n');
fprintf('%-20s %15s %15s\n', '模型', '训练准确率', '测试准确率');
fprintf('%-20s %14.2f%% %14.2f%%\n', '感知机', acc_p_train*100, acc_p_test*100);
fprintf('%-20s %14.2f%% %14.2f%%\n', 'MLP (30隐藏层)', acc_mlp_train*100, acc_mlp_test*100);
fprintf('--------------------------------------------\n');
fprintf('分析: 感知机作为线性分类器，只能产生直线决策边界，\n');
fprintf('无法拟合心形曲线的非线性边界，准确率受限。\n');
fprintf('MLP 通过隐藏层的非线性激活函数组合，能够逼近任意\n');
fprintf('复杂的决策边界，因此分类准确率显著优于感知机。\n');
fprintf('============================================\n');

%% ========== 辅助函数 ==========

function plot_decision_boundary(X, y, X_mean, X_std, predict_fn, x_lim, y_lim)
    % 画决策边界和样本点
    grid_size = 200;
    xg = linspace(x_lim(1), x_lim(2), grid_size);
    yg = linspace(y_lim(1), y_lim(2), grid_size);
    [Xg, Yg] = meshgrid(xg, yg);
    X_grid = [Xg(:), Yg(:)];

    y_grid = predict_fn(X_grid);

    % 画背景决策区域
    imagesc(xg, yg, reshape(y_grid, grid_size, grid_size));
    colormap(gca, [0.85 0.85 1.0; 1.0 0.85 0.85]);
    hold on;

    % 画样本点
    scatter(X(y==1, 1), X(y==1, 2), 12, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    scatter(X(y==0, 1), X(y==0, 2), 12, 'b', 'filled', 'MarkerFaceAlpha', 0.6);

    xlabel('x_1'); ylabel('x_2');
    axis equal;
    xlim(x_lim); ylim(y_lim);
    set(gca, 'YDir', 'normal');
    hold off;
end

function y_pred = perceptron_predict(X, w, b)
    % 感知机预测（需要标准化）
    s = X * w + b;
    y_pred = s >= 0;
end

function A2 = forward_pass_mlp(X, W1, b1, W2, b2)
    % MLP 前向传播
    A1 = 1 ./ (1 + exp(-(X * W1 + b1)));
    A2 = 1 ./ (1 + exp(-(A1 * W2 + b2)));
end
