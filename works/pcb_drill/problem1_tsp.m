%% B题 问题1：PCB钻孔路径优化（TSP）
% 钻头从原点 O(0,0) 出发，遍历所有 n 个钻孔点后返回原点
% 移动速度 v = 100 mm/s，距离为欧氏距离
% 策略：n≤200 用遗传算法+2-opt，n≤500 用模拟退火+2-opt，n>500 用多次随机NN+2-opt

clear; clc; close all;

% 数据目录：优先使用脚本所在目录，其次使用绝对路径
script_dir = fileparts(mfilename('fullpath'));
if isempty(script_dir)
    script_dir = pwd;
end
datadir = fullfile(script_dir);
datasets = [50, 198, 442, 1173];
v = 100;  % mm/s

results = cell(length(datasets), 1);

for d = 1:length(datasets)
    n = datasets(d);
    fprintf('========== n = %d ==========\n', n);

    % 读取坐标
    filename = fullfile(datadir, sprintf('pcb%d.txt', n));
    data = readtable(filename);
    pts = [data.x, data.y];  % n×2

    % 加入原点
    coords = [0, 0; pts];
    N = n + 1;
    origin_idx = 1;

    % 距离矩阵
    diffX = coords(:,1) - coords(:,1)';
    diffY = coords(:,2) - coords(:,2)';
    D = sqrt(diffX.^2 + diffY.^2);

    % 选择算法
    tic;
    if n <= 500
        [route, best_dist] = ga_tsp(D, N, origin_idx, n);
    else
        [route, best_dist] = multi_nn_2opt(D, N, origin_idx);
    end
    elapsed = toc;

    % 确保从原点出发
    route = shift_to_origin(route, origin_idx);

    % 计算总距离
    total_dist = route_distance(route, D);

    fprintf('  最短路径: %.2f mm\n', total_dist);
    fprintf('  移动时间: %.2f s (%.3f min)\n', total_dist/v, total_dist/v/60);
    fprintf('  计算耗时: %.2f s\n', elapsed);

    results{d} = struct('n', n, 'route', route, 'dist', total_dist, 'time', total_dist/v);

    % 绘图
    figure('Position', [50+d*20, 50+d*20, 800, 600]);
    plot_route(coords, route, n, total_dist);
    sgtitle(sprintf('PCB钻孔路径 n=%d  总长=%.1f mm  时间=%.2f s', n, total_dist, total_dist/v));
end

% 汇总
fprintf('\n========== 结果汇总 ==========\n');
fprintf('%-6s  %-12s  %-10s  %-10s\n', 'n', '路径长度(mm)', '时间(s)', '时间(min)');
for d = 1:length(datasets)
    r = results{d};
    fprintf('%-6d  %-12.2f  %-10.2f  %-10.3f\n', r.n, r.dist, r.time, r.time/60);
end

%% ========== 遗传算法 ==========
function [best_route, best_dist] = ga_tsp(D, N, origin_idx, n)
    pop_size = min(200, max(80, n * 2));
    n_generations = min(800, max(300, n * 3));
    elite_size = max(2, round(pop_size * 0.05));
    p_cross = 0.85;
    p_mut = 0.15;
    stall_limit = max(50, n_generations / 5);

    % 初始化种群
    pop = zeros(pop_size, N);
    pop(1,:) = nearest_neighbor(D, N, origin_idx);
    for i = 2:pop_size
        pop(i,:) = randperm(N);
    end
    fitness = zeros(pop_size, 1);
    for i = 1:pop_size
        fitness(i) = route_distance(pop(i,:), D);
    end

    % 初始2-opt优化最好的几个
    for i = 1:min(5, pop_size)
        pop(i,:) = two_opt(pop(i,:), D);
        fitness(i) = route_distance(pop(i,:), D);
    end

    [best_dist, best_idx] = min(fitness);
    best_route = pop(best_idx, :);
    stall_count = 0;

    for gen = 1:n_generations
        new_pop = zeros(pop_size, N);

        % 精英保留
        [~, sort_idx] = sort(fitness);
        for i = 1:elite_size
            new_pop(i,:) = pop(sort_idx(i),:);
        end

        % 选择、交叉、变异
        for i = elite_size+1:2:pop_size
            p1 = tournament_select(pop, fitness, 3);
            p2 = tournament_select(pop, fitness, 3);

            if rand < p_cross
                [c1, c2] = pmx_crossover(p1, p2);
            else
                c1 = p1; c2 = p2;
            end

            if rand < p_mut
                c1 = swap_mutate(c1);
            end
            if rand < p_mut
                c2 = swap_mutate(c2);
            end

            new_pop(i,:) = c1;
            if i+1 <= pop_size
                new_pop(i+1,:) = c2;
            end
        end

        % 计算新适应度
        for i = elite_size+1:pop_size
            fitness(i) = route_distance(new_pop(i,:), D);
        end

        pop = new_pop;
        [gen_best, gen_best_idx] = min(fitness);

        % 每隔K代对最优个体做2-opt
        if mod(gen, 10) == 0
            pop(gen_best_idx,:) = two_opt(pop(gen_best_idx,:), D);
            fitness(gen_best_idx) = route_distance(pop(gen_best_idx,:), D);
            gen_best = fitness(gen_best_idx);
        end

        % 检查改进
        if gen_best < best_dist
            best_dist = gen_best;
            best_route = pop(gen_best_idx, :);
            stall_count = 0;
        else
            stall_count = stall_count + 1;
        end

        if mod(gen, 50) == 0
            fprintf('  GA gen=%d best=%.2f\n', gen, best_dist);
        end

        if stall_count >= stall_limit
            fprintf('  GA 收敛于 gen=%d\n', gen);
            break;
        end
    end

    % 最终2-opt精炼
    best_route = two_opt(best_route, D);
    best_dist = route_distance(best_route, D);
end

%% ========== 模拟退火 ==========
function [best_route, best_dist] = sa_tsp(D, N, origin_idx, n)
    % 初始解：NN + 2-opt
    route = nearest_neighbor(D, N, origin_idx);
    route = two_opt(route, D);
    curr_dist = route_distance(route, D);

    best_route = route;
    best_dist = curr_dist;

    % 参数
    T0 = max(10, n * 0.5);
    T = T0;
    alpha = 0.997;
    T_min = 0.01;
    max_iter_per_T = max(100, n * 2);
    max_total_iter = min(200000, max(50000, n * 100));

    iter = 0;
    while T > T_min && iter < max_total_iter
        for i = 1:max_iter_per_T
            % 2-opt邻域
            new_route = two_opt_move(route);
            new_dist = route_distance(new_route, D);
            delta = new_dist - curr_dist;

            if delta < 0 || rand < exp(-delta / T)
                route = new_route;
                curr_dist = new_dist;
                if curr_dist < best_dist
                    best_route = route;
                    best_dist = curr_dist;
                end
            end

            iter = iter + 1;
            if iter >= max_total_iter, break; end
        end

        T = T * alpha;

        if mod(iter, max_iter_per_T * 10) < max_iter_per_T
            fprintf('  SA T=%.3f best=%.2f\n', T, best_dist);
        end
    end

    best_route = two_opt(best_route, D);
    best_dist = route_distance(best_route, D);
end

%% ========== 多次NN + 2-opt（大规模） ==========
function [best_route, best_dist] = multi_nn_2opt(D, N, origin_idx)
    n_restarts = 10;

    route = nearest_neighbor(D, N, origin_idx);
    route = two_opt(route, D);
    best_dist = route_distance(route, D);
    best_route = route;

    fprintf('  NN+2opt 第 1 次: %.2f\n', best_dist);

    for r = 2:n_restarts
        % 随机起始点
        rand_start = randi([1, N]);
        cand = nearest_neighbor(D, N, rand_start);
        cand = two_opt(cand, D);
        d = route_distance(cand, D);
        fprintf('  NN+2opt 第 %d 次: %.2f\n', r, d);
        if d < best_dist
            best_dist = d;
            best_route = cand;
        end
    end

    % 最终SA微调（步数较少）
    fprintf('  最终SA微调...\n');
    route = best_route;
    curr_dist = best_dist;
    T = 5; alpha = 0.99; T_min = 0.01;
    while T > T_min
        for i = 1:N*2
            new_route = two_opt_move(route);
            new_dist = route_distance(new_route, D);
            delta = new_dist - curr_dist;
            if delta < 0 || rand < exp(-delta / T)
                route = new_route;
                curr_dist = new_dist;
                if curr_dist < best_dist
                    best_route = route;
                    best_dist = curr_dist;
                end
            end
        end
        T = T * alpha;
    end
end

%% ========== 最近邻构造 ==========
function route = nearest_neighbor(D, N, start_node)
    visited = false(1, N);
    route = zeros(1, N);
    current = start_node;
    visited(current) = true;
    route(1) = current;

    for i = 2:N
        unvisited = find(~visited);
        [~, idx] = min(D(current, unvisited));
        current = unvisited(idx);
        visited(current) = true;
        route(i) = current;
    end
end

%% ========== 2-opt 局部搜索 ==========
function route = two_opt(route, D)
    N = length(route);
    improved = true;
    while improved
        improved = false;
        for i = 1:N-2
            for j = i+2:N
                % 避免首尾相接的情况
                if i == 1 && j == N, continue; end
                i_next = i + 1;
                if i_next > N, i_next = 1; end
                j_next = j + 1;
                if j_next > N, j_next = 1; end

                old_cost = D(route(i), route(i_next)) + D(route(j), route(j_next));
                new_cost = D(route(i), route(j)) + D(route(i_next), route(j_next));
                if new_cost < old_cost - 1e-10
                    route(i+1:j) = route(j:-1:i+1);
                    improved = true;
                end
            end
        end
    end
end

%% ========== 单步2-opt移动 ==========
function route = two_opt_move(route)
    N = length(route);
    i = randi(N-1);
    j = randi(N);
    while abs(j - i) < 2 || (i == 1 && j == N)
        j = randi(N);
    end
    if i > j
        tmp = i; i = j; j = tmp;
    end
    route(i+1:j) = route(j:-1:i+1);
end

%% ========== 计算回路距离 ==========
function d = route_distance(route, D)
    N = length(route);
    d = 0;
    for i = 1:N-1
        d = d + D(route(i), route(i+1));
    end
    d = d + D(route(N), route(1));
end

%% ========== 锦标赛选择 ==========
function winner = tournament_select(pop, fitness, k)
    [psize, ~] = size(pop);
    idx = randperm(psize, k);
    [~, best] = min(fitness(idx));
    winner = pop(idx(best), :);
end

%% ========== PMX交叉 ==========
function [c1, c2] = pmx_crossover(p1, p2)
    N = length(p1);
    a = randi(N-1); b = randi(N);
    while b <= a
        b = randi(N);
    end

    c1 = p1; c2 = p2;

    % 子代1
    seg1 = p1(a:b);
    seg2 = p2(a:b);
    c1(a:b) = seg2;
    % 修复冲突
    for k = [1:a-1, b+1:N]
        while ismember(c1(k), seg2)
            pos = find(seg2 == c1(k), 1);
            c1(k) = seg1(pos);
        end
    end

    % 子代2
    c2(a:b) = seg1;
    for k = [1:a-1, b+1:N]
        while ismember(c2(k), seg1)
            pos = find(seg1 == c2(k), 1);
            c2(k) = seg2(pos);
        end
    end
end

%% ========== 交换变异 ==========
function route = swap_mutate(route)
    N = length(route);
    i = randi(N);
    j = randi(N);
    while j == i, j = randi(N); end
    tmp = route(i);
    route(i) = route(j);
    route(j) = tmp;
end

%% ========== 确保路线从原点出发 ==========
function route = shift_to_origin(route, origin_idx)
    pos = find(route == origin_idx, 1);
    if pos > 1
        route = [route(pos:end), route(1:pos-1)];
    end
end

%% ========== 可视化 ==========
function plot_route(coords, route, n, total_dist)
    % 路径线
    subplot(1,2,1);
    x = coords(route, 1);
    y = coords(route, 2);
    x = [x; x(1)];  % 回到起点
    y = [y; y(1)];
    plot(x, y, 'b-', 'LineWidth', 1);
    hold on;

    % 原点标记
    plot(coords(1,1), coords(1,2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    % 钻孔点
    scatter(coords(2:end,1), coords(2:end,2), 12, 'k', 'filled', 'MarkerFaceAlpha', 0.5);

    xlabel('X (mm)'); ylabel('Y (mm)');
    title(sprintf('钻孔路径 n=%d', n));
    legend('路径', '原点O', '钻孔点', 'Location', 'best');
    axis equal tight; grid on;

    % 距离收敛图（如果GA或SA有记录）
    subplot(1,2,2);
    % 绘制路径长度分解
    route_pts = coords(route, :);
    segs = sqrt(sum(diff([route_pts; route_pts(1,:)]).^2, 2));
    bar(segs(1:min(50,length(segs))));
    xlabel('路径段序号'); ylabel('段长度 (mm)');
    title(sprintf('前50段路径长度分布\n总长=%.1f mm, 平均段长=%.1f mm', total_dist, mean(segs)));
    grid on;
end
