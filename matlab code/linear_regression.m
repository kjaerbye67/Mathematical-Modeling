% ============================================================
% 线性回归分析脚本（仅依赖 MATLAB 核心函数，无需额外 Toolbox）
% 使用最小二乘法拟合 y = ax + b
% ============================================================

clear; clc; close all;

%% ========== 1. 输入数据 ==========
% 请在此处替换为你自己的数据
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]';
y = [2.1, 4.0, 5.8, 8.2, 9.9, 12.3, 14.0, 16.1, 18.2, 20.0]';

n = length(x);

%% ========== 2. 最小二乘回归 ==========
% y = a*x + b
X_design = [x, ones(n, 1)];
beta = X_design \ y;          % 等同于 inv(X'X) * X'y
a = beta(1);                  % 斜率
b = beta(2);                  % 截距

%% ========== 3. 拟合值与残差 ==========
y_fit = X_design * beta;
residuals = y - y_fit;

%% ========== 4. 拟合优度 R² ==========
SS_res = sum(residuals.^2);
SS_tot = sum((y - mean(y)).^2);
R2 = 1 - SS_res / SS_tot;

%% ========== 5. 标准误差与置信区间 ==========
sigma2 = SS_res / (n - 2);                       % 残差方差
SE_slope = sqrt(sigma2 / sum((x - mean(x)).^2)); % 斜率标准误
SE_intercept = sqrt(sigma2 * (1/n + mean(x)^2 / sum((x - mean(x)).^2))); % 截距标准误

alpha = 0.05;
t_crit = t_inv(1 - alpha/2, n - 2);             % 自定义 t 逆函数
CI_slope = [a - t_crit * SE_slope, a + t_crit * SE_slope];
CI_intercept = [b - t_crit * SE_intercept, b + t_crit * SE_intercept];

%% ========== 6. 假设检验 (t 检验) ==========
t_stat_a = a / SE_slope;
t_stat_b = b / SE_intercept;
p_value_a = 2 * (1 - t_cdf(abs(t_stat_a), n - 2));
p_value_b = 2 * (1 - t_cdf(abs(t_stat_b), n - 2));

%% ========== 7. 打印结果 ==========
fprintf('============================================\n');
fprintf('           线性回归分析结果\n');
fprintf('============================================\n');
fprintf('回归方程:  y = %.4f x + %.4f\n', a, b);
fprintf('样本量 n:  %d\n', n);
fprintf('--------------------------------------------\n');
fprintf('R² (决定系数):        %.4f\n', R2);
fprintf('校正 R²:              %.4f\n', 1 - (1-R2)*(n-1)/(n-2));
fprintf('残差标准差 (RMSE):    %.4f\n', sqrt(sigma2));
fprintf('--------------------------------------------\n');
fprintf('          系数      标准误      t 值      p 值\n');
fprintf('斜率 a:  %8.4f  %8.4f  %8.4f  %8.4f\n', a, SE_slope, t_stat_a, p_value_a);
fprintf('截距 b:  %8.4f  %8.4f  %8.4f  %8.4f\n', b, SE_intercept, t_stat_b, p_value_b);
fprintf('--------------------------------------------\n');
fprintf('斜率 95%% 置信区间: [%.4f, %.4f]\n', CI_slope(1), CI_slope(2));
fprintf('截距 95%% 置信区间: [%.4f, %.4f]\n', CI_intercept(1), CI_intercept(2));
fprintf('============================================\n');

%% ========== 8. 可视化 ==========
figure('Position', [100, 100, 1200, 500]);

% ---- 子图1: 散点图 + 回归线 ----
subplot(1, 2, 1);
x_range = max(x) - min(x);
x_plot = linspace(min(x) - 0.1*x_range, max(x) + 0.1*x_range, 100)';
y_plot = a * x_plot + b;

scatter(x, y, 60, 'b', 'filled', 'DisplayName', '原始数据');
hold on;
plot(x_plot, y_plot, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('y = %.4f x + %.4f', a, b));

% 95% 预测区间
se_pred = sqrt(sigma2 * (1 + 1/n + (x_plot - mean(x)).^2 / sum((x - mean(x)).^2)));
y_pred_upper = y_plot + t_crit * se_pred;
y_pred_lower = y_plot - t_crit * se_pred;
fill([x_plot; flipud(x_plot)], [y_pred_lower; flipud(y_pred_upper)], ...
    'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '95% 预测区间');

xlabel('x');
ylabel('y');
title(sprintf('线性回归 (R^2 = %.4f)', R2));
legend('Location', 'best');
grid on;
hold off;

% ---- 子图2: 残差图 ----
subplot(1, 2, 2);
stem(x, residuals, 'b', 'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', '残差');
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('x');
ylabel('残差');
title('残差分布图');
legend('Location', 'best');
grid on;
hold off;

sgtitle('线性回归分析');


%% ========== 自定义函数：仅依赖 MATLAB 核心函数 ==========

function p = t_cdf(t_val, df)
    % Student's t 分布的累积分布函数
    % 通过数值积分计算，仅使用 integral / gamma / gamma
    f = @(u) t_pdf(u, df);
    p = integral(f, -Inf, t_val, 'ArrayValued', true);
end

function t_val = t_inv(p, df)
    % Student's t 分布的逆累积分布函数
    % 使用 fzero 求根，不依赖 Statistics Toolbox
    if p <= 0 || p >= 1
        error('p 必须在 (0, 1) 之间');
    end
    % 初始猜测：用正态近似
    x0 = norminv_approx(p);
    t_val = fzero(@(t) t_cdf(t, df) - p, x0);
end

function y = t_pdf(t_val, df)
    % Student's t 分布的概率密度函数
    c = gamma((df + 1) / 2) / (sqrt(df * pi) * gamma(df / 2));
    y = c * (1 + t_val.^2 / df) .^ (-(df + 1) / 2);
end

function z = norminv_approx(p)
    % 标准正态分布逆函数的近似（Abramowitz and Stegun）
    % 用作 t_inv 中 fzero 的初始值
    a = [2.515517, 0.802853, 0.010328];
    b = [1.0, 1.432788, 0.189269, 0.001308];
    t_val = sqrt(-2 * log(min(p, 1-p)));
    num = a(1) + a(2)*t_val + a(3)*t_val^2;
    den = b(1) + b(2)*t_val + b(3)*t_val^2 + b(4)*t_val^3;
    z = t_val - num / den;
    if p < 0.5
        z = -z;
    end
end
