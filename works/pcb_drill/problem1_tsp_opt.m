%% B题 问题1：PCB钻孔路径优化（TSP）—— 优化版
% 改进点:
%   1. 候选列表：2-opt只查K=20最近邻, O(n^2)→O(nK)
%   2. OX交叉：替代PMX, 更好地保留边的顺序
%   3. Double-bridge扰动：跳出局部最优
%   4. ILS框架：多轮"扰动→优化→接受"迭代
%   5. 自适应策略：n≤500用GA+ILS混合, n>500用ILS

clear; clc; close all;

datadir = fileparts(mfilename('fullpath'));
if isempty(datadir), datadir = pwd; end
datasets = [50, 198, 442, 1173];
v = 100;

K = 20;  % 候选列表大小

results = cell(length(datasets), 1);

for d = 1:length(datasets)
    n = datasets(d);
    fprintf('========== n = %d ==========\n', n);

    filename = fullfile(datadir, sprintf('pcb%d.txt', n));
    data = readtable(filename);
    pts = [data.x, data.y];
    coords = [0, 0; pts];
    N = n + 1;
    origin_idx = 1;

    % 距离矩阵
    diffX = coords(:,1) - coords(:,1)';
    diffY = coords(:,2) - coords(:,2)';
    D = sqrt(diffX.^2 + diffY.^2);

    % 构建候选列表
    fprintf('  构建候选列表 (K=%d)...\n', K);
    [cand_list, ~] = build_candidate_list(D, K);

    tic;
    if n <= 500
        [route, best_dist] = ga_ils_tsp(D, cand_list, N, origin_idx, n);
    else
        [route, best_dist] = ils_tsp(D, cand_list, N, origin_idx, n);
    end
    elapsed = toc;

    route = shift_to_origin(route, origin_idx);
    total_dist = route_distance(route, D);

    fprintf('  最短路径: %.2f mm\n', total_dist);
    fprintf('  移动时间: %.2f s (%.3f min)\n', total_dist/v, total_dist/v/60);
    fprintf('  计算耗时: %.2f s\n', elapsed);

    results{d} = struct('n', n, 'route', route, 'dist', total_dist, 'time', total_dist/v);

    figure('Position', [50+d*20, 50+d*20, 800, 600]);
    plot_route(coords, route, n, total_dist);
    sgtitle(sprintf('PCB钻孔路径 n=%d  总长=%.1f mm  时间=%.2f s', n, total_dist, total_dist/v));
end

fprintf('\n========== 结果汇总 ==========\n');
fprintf('%-6s  %-12s  %-10s  %-10s\n', 'n', '路径长度(mm)', '时间(s)', '时间(min)');
for d = 1:length(datasets)
    r = results{d};
    fprintf('%-6d  %-12.2f  %-10.2f  %-10.3f\n', r.n, r.dist, r.time, r.time/60);
end

%% ==================== 候选列表 ====================
function [cand_list, cand_dists] = build_candidate_list(D, K)
    N = size(D, 1);
    K = min(K, N - 1);
    cand_list = zeros(N, K);
    cand_dists = zeros(N, K);
    for i = 1:N
        d = D(i, :);
        d(i) = inf;
        [sorted, idx] = sort(d);
        cand_list(i, :) = idx(1:K);
        cand_dists(i, :) = sorted(1:K);
    end
end

%% ==================== ILS框架 ====================
function [best_route, best_dist] = ils_tsp(D, cand_list, N, origin_idx, n)
    % 迭代局部搜索
    max_ils = min(40, max(15, round(1500 / n)));
    max_no_improve = 8;

    % 初始解：贪心插入 + 全量2-opt
    route = greedy_insertion(D, N, origin_idx);
    route = two_opt(route, D);
    best_route = route;
    best_dist = route_distance(route, D);

    fprintf('  初始解: %.2f\n', best_dist);

    no_improve = 0;
    for iter = 1:max_ils
        % Double-bridge 扰动
        perturbed = double_bridge(route);

        % 全量2-opt局部搜索
        local_opt = two_opt(perturbed, D);
        local_dist = route_distance(local_opt, D);

        % 接受准则：只接受更优解
        if local_dist < best_dist - 1e-8
            fprintf('  ILS iter=%d: %.2f -> %.2f (改进 %.2f%%)\n', ...
                iter, best_dist, local_dist, (best_dist-local_dist)/best_dist*100);
            best_dist = local_dist;
            best_route = local_opt;
            route = local_opt;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
            if mod(iter, 5) == 0
                fprintf('  ILS iter=%d: 无改进 (best=%.2f)\n', iter, best_dist);
            end
        end

        if no_improve >= max_no_improve
            % 随机重启动（全量2-opt）
            route = greedy_insertion(D, N, randi(N));
            route = two_opt(route, D);
            no_improve = 0;
        end
    end
end

%% ==================== GA + ILS混合 ====================
function [best_route, best_dist] = ga_ils_tsp(D, cand_list, N, origin_idx, n)
    % GA 探索 + ILS 精炼
    pop_size = min(180, max(80, n * 2));
    n_generations = min(500, max(200, n * 2));
    elite_size = max(2, round(pop_size * 0.05));
    p_cross = 0.9;

    % 初始化：多种子 + 全量2-opt
    pop = zeros(pop_size, N);
    fitness = zeros(pop_size, 1);
    % 种子1: NN + 2opt
    pop(1,:) = nearest_neighbor(D, N, origin_idx);
    pop(1,:) = two_opt(pop(1,:), D);
    fitness(1) = route_distance(pop(1,:), D);
    % 种子2: 贪心插入 + 2opt
    pop(2,:) = greedy_insertion(D, N, origin_idx);
    pop(2,:) = two_opt(pop(2,:), D);
    fitness(2) = route_distance(pop(2,:), D);
    % 其余: 随机起始贪心插入 + 2opt
    for i = 3:pop_size
        pop(i,:) = greedy_insertion(D, N, randi(N));
        pop(i,:) = two_opt(pop(i,:), D);
        fitness(i) = route_distance(pop(i,:), D);
    end

    [best_dist, best_idx] = min(fitness);
    best_route = pop(best_idx, :);
    stall_count = 0;
    stall_limit = max(30, n_generations / 10);
    mutation_rate = 0.02;

    fprintf('  GA初始 best=%.2f\n', best_dist);

    for gen = 1:n_generations
        new_pop = zeros(pop_size, N);

        % 精英保留
        [~, sort_idx] = sort(fitness);
        for i = 1:elite_size
            new_pop(i,:) = pop(sort_idx(i),:);
        end

        % OX交叉 + 自适应变异
        for i = elite_size+1:2:pop_size
            p1 = tournament_select(pop, fitness, 2);
            p2 = tournament_select(pop, fitness, 2);

            if rand < p_cross
                [c1, c2] = ox_crossover(p1, p2);
            else
                c1 = p1; c2 = p2;
            end

            % 对不优良个体做更多变异
            if rand < mutation_rate * (1 + stall_count / stall_limit)
                c1 = double_bridge(c1);
            end
            if rand < mutation_rate * (1 + stall_count / stall_limit)
                c2 = double_bridge(c2);
            end

            new_pop(i,:) = c1;
            if i+1 <= pop_size
                new_pop(i+1,:) = c2;
            end
        end

        for i = elite_size+1:pop_size
            fitness(i) = route_distance(new_pop(i,:), D);
        end

        pop = new_pop;
        [gen_best, gen_best_idx] = min(fitness);

        % 对种群最优做全量2-opt
        if mod(gen, 5) == 0
            pop(gen_best_idx,:) = two_opt(pop(gen_best_idx,:), D);
            fitness(gen_best_idx) = route_distance(pop(gen_best_idx,:), D);
            gen_best = fitness(gen_best_idx);
        end

        if gen_best < best_dist
            best_dist = gen_best;
            best_route = pop(gen_best_idx, :);
            stall_count = 0;
            if mod(gen, 10) == 0
                fprintf('  GA gen=%d best=%.2f\n', gen, best_dist);
            end
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

    % 最终ILS精炼（从GA最优解出发）
    fprintf('  最终ILS精炼...\n');
    [best_route, best_dist] = ils_refine(best_route, D, cand_list, N);
end

%% ==================== ILS精炼（从已有解出发） ====================
function [best_route, best_dist] = ils_refine(route, D, cand_list, N)
    max_ils = 20;
    max_no_improve = 5;
    best_route = route;
    best_dist = route_distance(route, D);
    no_improve = 0;

    fprintf('  从 %.2f 开始精炼...\n', best_dist);

    for iter = 1:max_ils
        perturbed = double_bridge(route);
        local_opt = two_opt(perturbed, D);
        local_dist = route_distance(local_opt, D);

        if local_dist < best_dist - 1e-8
            fprintf('  ILS iter=%d: %.2f -> %.2f (改进 %.2f%%)\n', ...
                iter, best_dist, local_dist, (best_dist-local_dist)/best_dist*100);
            best_dist = local_dist;
            best_route = local_opt;
            route = local_opt;
            no_improve = 0;
        else
            no_improve = no_improve + 1;
        end

        if no_improve >= max_no_improve
            break;
        end
    end
end

%% ==================== 贪心插入构造 ====================
function route = greedy_insertion(D, N, start_node)
    % 最近插入法：每次将距当前回路最近的未访问节点插入最佳位置
    unvisited = true(1, N);
    route = zeros(1, N);
    route(1) = start_node;
    unvisited(start_node) = false;
    m = 1;

    while m < N
        % 找距当前回路最近的未访问节点
        min_dist = inf;
        best_k = 0;
        for k = 1:N
            if unvisited(k)
                d = min(D(k, route(1:m)));
                if d < min_dist
                    min_dist = d;
                    best_k = k;
                end
            end
        end

        % 在回路中找最佳插入位置
        best_gain = inf;
        best_pos = 0;
        for pos = 1:m
            a = route(pos);
            b = route(mod(pos, m) + 1);
            gain = D(a, best_k) + D(best_k, b) - D(a, b);
            if gain < best_gain
                best_gain = gain;
                best_pos = pos;
            end
        end

        % 在 best_pos 后插入 best_k
        route(best_pos + 2:m + 1) = route(best_pos + 1:m);
        route(best_pos + 1) = best_k;
        unvisited(best_k) = false;
        m = m + 1;
    end
end

%% ==================== 全量2-opt（用于初始化和精炼） ====================
function route = two_opt(route, D)
    N = length(route);
    pos = zeros(1, N);
    for i = 1:N, pos(route(i)) = i; end
    improved = true;
    while improved
        improved = false;
        for i = 1:N-2
            i_node = route(i);
            i_next_node = route(i + 1);
            for j = i+2:N
                if i == 1 && j == N, continue; end
                j_next = j + 1;
                if j_next > N, j_next = 1; end
                j_node = route(j);
                j_next_node = route(j_next);

                old_cost = D(i_node, i_next_node) + D(j_node, j_next_node);
                new_cost = D(i_node, j_node) + D(i_next_node, j_next_node);
                if new_cost < old_cost - 1e-10
                    route(i+1:j) = route(j:-1:i+1);
                    for k = i+1:j, pos(route(k)) = k; end
                    improved = true;
                    break;
                end
            end
            if improved, break; end
        end
    end
end

%% ==================== 快速2-opt（候选列表） ====================
function route = fast_2opt(route, D, cand_list)
    N = length(route);
    % 位置索引
    pos = zeros(1, N);
    for i = 1:N
        pos(route(i)) = i;
    end

    improved = true;
    while improved
        improved = false;
        for i = 1:N-2
            i_node = route(i);
            i_next_node = route(i + 1);
            candidates = cand_list(i_node, :);

            for c = 1:length(candidates)
                j_node = candidates(c);
                j = pos(j_node);
                if j <= i + 1 || (i == 1 && j == N), continue; end

                j_next = j + 1;
                if j_next > N, j_next = 1; end
                j_next_node = route(j_next);

                old_cost = D(i_node, i_next_node) + D(j_node, j_next_node);
                new_cost = D(i_node, j_node) + D(i_next_node, j_next_node);
                if new_cost < old_cost - 1e-10
                    % 翻转 [i+1, j]
                    route(i+1:j) = route(j:-1:i+1);
                    % 更新位置
                    for k = i+1:j
                        pos(route(k)) = k;
                    end
                    improved = true;
                    break;
                end
            end
            if improved, break; end
        end
    end
end

%% ==================== Double-bridge 扰动 ====================
function route = double_bridge(route)
    % 随机切4段，重新拼接：A-B-C-D → A-D-C-B
    % 无法被2-opt逆转为原路径，有效跳出局部最优
    N = length(route);

    % 随机选3个切点
    cuts = sort(randperm(N-2, 3) + 1);
    c1 = cuts(1); c2 = cuts(2); c3 = cuts(3);

    A = route(1:c1-1);
    B = route(c1:c2-1);
    C = route(c2:c3-1);
    D = route(c3:end);

    route = [A, D, C, B];
end

%% ==================== OX交叉 ====================
function [c1, c2] = ox_crossover(p1, p2)
    N = length(p1);
    a = randi(N-1); b = randi(N);
    while b <= a, b = randi(N); end

    % 子代1：从p2继承[a,b]段, 其余按p1顺序补全
    c1 = ox_build(p1, p2, a, b);
    % 子代2：从p1继承[a,b]段, 其余按p2顺序补全
    c2 = ox_build(p2, p1, a, b);
end

function child = ox_build(preserve, donate, a, b)
    N = length(preserve);
    child = zeros(1, N);
    child(a:b) = donate(a:b);
    used = false(1, N);
    used(donate(a:b)) = true;

    idx = b + 1;
    for i = 1:N
        k = mod(b + i - 1, N) + 1;
        if idx > N, idx = 1; end
        if idx >= a && idx <= b
            idx = b + 1;
            if idx > N, idx = 1; end
        end
        v = preserve(k);
        if ~used(v)
            child(idx) = v;
            used(v) = true;
            idx = idx + 1;
        end
    end
end

%% ==================== 辅助函数 ====================
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

function d = route_distance(route, D)
    N = length(route);
    d = sum(D(sub2ind(size(D), route(1:N-1), route(2:N))));
    d = d + D(route(N), route(1));
end

function winner = tournament_select(pop, fitness, k)
    [psize, ~] = size(pop);
    idx = randperm(psize, k);
    [~, best] = min(fitness(idx));
    winner = pop(idx(best), :);
end

function route = shift_to_origin(route, origin_idx)
    pos = find(route == origin_idx, 1);
    if pos > 1
        route = [route(pos:end), route(1:pos-1)];
    end
end

%% ==================== 可视化 ====================
function plot_route(coords, route, n, total_dist)
    subplot(1,2,1);
    x = coords(route, 1); y = coords(route, 2);
    x = [x; x(1)]; y = [y; y(1)];
    plot(x, y, 'b-', 'LineWidth', 1); hold on;
    plot(coords(1,1), coords(1,2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    scatter(coords(2:end,1), coords(2:end,2), 12, 'k', 'filled', 'MarkerFaceAlpha', 0.5);
    xlabel('X (mm)'); ylabel('Y (mm)');
    title(sprintf('钻孔路径 n=%d', n));
    legend('路径', '原点O', '钻孔点', 'Location', 'best');
    axis equal tight; grid on;

    subplot(1,2,2);
    route_pts = coords(route, :);
    segs = sqrt(sum(diff([route_pts; route_pts(1,:)]).^2, 2));
    bar(segs(1:min(50,length(segs))));
    xlabel('路径段序号'); ylabel('段长度 (mm)');
    title(sprintf('前50段路径长度分布\n总长=%.1f mm, 平均段长=%.1f mm', total_dist, mean(segs)));
    grid on;
end
