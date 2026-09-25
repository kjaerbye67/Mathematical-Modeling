% ============================================================
% 多元线性回归分析脚本（仅依赖 MATLAB 核心函数）
% 适用于任意数量的自变量：y = a1*x1 + a2*x2 + ... + ak*xk + b
% ============================================================

clear; clc; close all;

%% ========== 1. 输入数据 ==========
% 请在此处替换为你自己的数据

% 两个自变量（二元回归示例）
x1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]';
x2 = [2.3, 4.1, 5.8, 7.2, 9.5, 10.8, 13.0, 15.2, 17.0, 19.3]';
y  = [3.5, 6.2, 9.1, 12.0, 15.3, 18.5, 21.0, 24.2, 27.5, 30.8]';

% 将所有自变量合并为矩阵（每列一个自变量）
X_vars = [x1, x2];
var_names = {'x_1', 'x_2'};  % 自变量名称，用于显示

[n, k] = size(X_vars);  % n: 样本量, k: 自变量个数

%% ========== 2. 最小二乘回归（矩阵形式）==========
% 模型：y = X * beta，其中 X = [x1, x2, ..., xk, 1]
X_design = [X_vars, ones(n, 1)];
beta = X_design \ y;

coeffs = beta(1:k);    % 各自变量的系数 (a1, a2, ..., ak)
intercept = beta(k+1);  % 截距 b

%% ========== 3. 拟合值与残差 ==========
y_fit = X_design * beta;
residuals = y - y_fit;

%% ========== 4. 拟合优度 ==========
SS_res = sum(residuals.^2);
SS_tot = sum((y - mean(y)).^2);
R2 = 1 - SS_res / SS_tot;
R2_adj = 1 - (1 - R2) * (n - 1) / (n - k - 1);

%% ========== 5. 标准误差与统计检验 ==========
sigma2 = SS_res / (n - k - 1);            % 残差方差（无偏估计）
Cov_matrix = sigma2 * inv(X_design' * X_design);  % 系数协方差矩阵
SE = sqrt(diag(Cov_matrix));              % 各系数标准误 (k个系数 + 截距)

alpha = 0.05;
t_crit = t_inv(1 - alpha/2, n - k - 1);
t_stat = beta ./ SE;
p_value = 2 * (1 - t_cdf(abs(t_stat), n - k - 1));
CI = [beta - t_crit * SE, beta + t_crit * SE];

%% ========== 6. F 检验（整体显著性）==========
SS_reg = sum((y_fit - mean(y)).^2);
MS_reg = SS_reg / k;
MS_res = SS_res / (n - k - 1);
F_stat = MS_reg / MS_res;
p_value_F = 1 - f_cdf(F_stat, k, n - k - 1);

%% ========== 7. 打印结果 ==========
fprintf('============================================\n');
fprintf('          多元线性回归分析结果\n');
fprintf('============================================\n');

% 回归方程
fprintf('回归方程:  y = ');
for i = 1:k
    fprintf('%.4f·%s + ', coeffs(i), var_names{i});
end
fprintf('%.4f\n', intercept);

fprintf('样本量 n = %d,  自变量个数 k = %d\n', n, k);
fprintf('--------------------------------------------\n');
fprintf('R² (决定系数):         %.4f\n', R2);
fprintf('校正 R²:               %.4f\n', R2_adj);
fprintf('残差标准差 (RMSE):     %.4f\n', sqrt(sigma2));
fprintf('--------------------------------------------\n');

fprintf('%-10s %10s %10s %10s %10s\n', '系数', '估计值', '标准误', 't 值', 'p 值');
for i = 1:k
    fprintf('%-10s %10.4f %10.4f %10.4f %10.4f\n', ...
        var_names{i}, coeffs(i), SE(i), t_stat(i), p_value(i));
end
fprintf('%-10s %10.4f %10.4f %10.4f %10.4f\n', ...
    '截距', intercept, SE(k+1), t_stat(k+1), p_value(k+1));

fprintf('--------------------------------------------\n');
for i = 1:k
    fprintf('%s 95%% CI: [%.4f, %.4f]\n', var_names{i}, CI(i,1), CI(i,2));
end
fprintf('截距 95%% CI: [%.4f, %.4f]\n', CI(k+1,1), CI(k+1,2));
fprintf('--------------------------------------------\n');
fprintf('F 统计量: %.4f (df1 = %d, df2 = %d),  p 值 = %.4e\n', ...
    F_stat, k, n - k - 1, p_value_F);
fprintf('============================================\n');

%% ========== 8. 可视化 ==========
figure('Position', [100, 100, 1400, 900]);

% ---- 子图1: 实际值 vs 拟合值 ----
subplot(2, 3, 1);
scatter(y, y_fit, 50, 'b', 'filled');
hold on;
lims = [min([y; y_fit]), max([y; y_fit])];
plot(lims, lims, 'r--', 'LineWidth', 1.5);
xlabel('实际值'); ylabel('拟合值');
title(sprintf('实际值 vs 拟合值 (R^2 = %.4f)', R2));
grid on; axis equal;

% ---- 子图2: 残差 vs 拟合值 ----
subplot(2, 3, 2);
scatter(y_fit, residuals, 50, 'b', 'filled');
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('拟合值'); ylabel('残差');
title('残差 vs 拟合值');
grid on;

% ---- 子图3: 残差直方图 ----
subplot(2, 3, 3);
histogram(residuals, max(5, round(n/3)), 'FaceColor', 'b', 'EdgeColor', 'w');
xlabel('残差'); ylabel('频数');
title('残差分布直方图');
grid on;

% ---- 子图4-6: 各偏回归图（每个自变量 vs 因变量）----
for i = 1:k
    subplot(2, 3, 3 + i);
    scatter(X_vars(:, i), y, 50, 'b', 'filled', 'DisplayName', '原始数据');
    hold on;
    % 在其他变量均值处固定，画偏回归线
    x_i_range = max(X_vars(:, i)) - min(X_vars(:, i));
    x_i_plot = linspace(min(X_vars(:, i)) - 0.1*x_i_range, ...
                        max(X_vars(:, i)) + 0.1*x_i_range, 100)';
    X_mean = mean(X_vars);
    X_mean(i) = 0;
    y_partial = coeffs(i) * x_i_plot + (intercept + sum(coeffs' .* X_mean));
    plot(x_i_plot, y_partial, 'r-', 'LineWidth', 2);
    xlabel(var_names{i}); ylabel('y');
    title(sprintf('%s 偏回归图 (a_%d = %.4f)', var_names{i}, i, coeffs(i)));
    legend('Location', 'best');
    grid on;
    hold off;
end

sgtitle(sprintf('多元线性回归分析 (k = %d, n = %d)', k, n));


%% ========== 自定义函数 ==========

function p = t_cdf(t_val, df)
    % 支持向量输入：对每个元素分别积分
    p = arrayfun(@(tv) integral(@(u) t_pdf(u, df), -Inf, tv), t_val);
end

function t_val = t_inv(p, df)
    if p <= 0 || p >= 1
        error('p 必须在 (0, 1) 之间');
    end
    x0 = norminv_approx(p);
    t_val = fzero(@(t) t_cdf(t, df) - p, x0);
end

function y = t_pdf(t_val, df)
    c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2));
    y = c * (1 + t_val.^2 / df) .^ (-(df + 1) / 2);
end

function z = norminv_approx(p)
    a = [2.515517, 0.802853, 0.010328];
    b = [1.0, 1.432788, 0.189269, 0.001308];
    t_val = sqrt(-2 * log(min(p, 1-p)));
    num = a(1) + a(2)*t_val + a(3)*t_val^2;
    den = b(1) + b(2)*t_val + b(3)*t_val^2 + b(4)*t_val^3;
    z = t_val - num / den;
    if p < 0.5, z = -z; end
end

function p = f_cdf(f_val, df1, df2)
    % F 分布的 CDF（通过 Beta 函数计算）
    x = df1 * f_val / (df1 * f_val + df2);
    p = betainc(x, df1/2, df2/2);
end
