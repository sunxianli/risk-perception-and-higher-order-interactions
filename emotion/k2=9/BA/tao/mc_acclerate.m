clc; clear;
%%加速版本
%% —— 载入数据 ——
data_down = load('RSC_BA.mat');
data_up= load('RSC9.mat');
adj_A = sparse(data_up.net.adj_A);        % 用稀疏矩阵
adj_B = sparse(data_down.net.adj_B);
neighbors_A = data_up.net.neighbors_A;    %# 仍保留，情绪统计用不到它也行
triangles_A = data_up.net.triangles_A;
N = length(adj_A);
%% —— 参数 ——
stp    = 30;
MC_rep = 20;
termi  = 40;

lamda1 = 0.15;
delta  = 0.6;
lamda_deta =1.6;
mu     = 0.8;
gamma  = 0.3;
beta_R =0.4;
% omega1 = 0.8; 

% 节点度/三角形度
k_i = full(sum(adj_A, 2));  k_i(k_i==0) = 1;
k_i_delta_raw = accumarray(triangles_A(:), 1, [N,1]);      % 原始计数
avg_k_delta   = mean(k_i_delta_raw);
lamda2 = (delta * lamda_deta) / max(avg_k_delta, eps);
k_i_delta_div = k_i_delta_raw;  k_i_delta_div(k_i_delta_div==0) = 1;

% 阈值
rng(42);
EPS_SCALE = 0.6;                 % <—— 0<scale<=1；越小越容易过阈
epsilon   = EPS_SCALE * rand(1, N);
% yeta      = 0.05;
% tau0      = 0.6;
% threshold = tau0 + yeta .* epsilon;
paramter=[0.3,0.6,0.8];
num=length(paramter);
% —— 预处理：构建 (i, u, v) 列表，用于 2-simplex 统计 ——
[I_list, U_list, V_list] = build_pair_list(triangles_A, N);

tau0_list   = paramter(:).';              % 长度 = num
yeta_list   = linspace(1/termi, 1, termi);% 长度 = termi
omega1_list = linspace(1/termi, 1, termi);% 长度 = termi

% ---- 结果容器（只存 MC 均值与方差）----
AG_mean = zeros(num, termi, termi);       % [tau0 × yeta × omega1]
AG_std  = zeros(num, termi, termi);

% 建议：epsilon 固定一次，保证不同参数组合可比

for chance = 1:num
    tau0 = tau0_list(chance);
    for ye = 1:termi
        yeta = yeta_list(ye);
        % epsilon = rand(1, N);
        % 每个 (tau0, yeta) 的阈值向量（固定在本三重循环内）
        threshold = tau0 + yeta .* epsilon;

        for l = 1:termi
            omega1 = omega1_list(l);
            % omega2 = 1 - omega1;

            % 本组合下的 MC 收集器（每次重复输出一个标量 AG）
            ag_rep = zeros(MC_rep, 1);

            % —— 并行重复（rep 维度）——
            parfor rep = 1:MC_rep
                % ====== 初始化（按你原来的来）======
                rng(1000 + rep);              % 跨参数点复用同一批子种子（提升可比）

                csmd_G = 0.1; csmd_A = 0.2;
                x = rand(1, N) < csmd_G;      % G
                m = rand(1, N) < csmd_A;      % A
                m(x) = true; x(~m) = false;

                Total_Emo = zeros(1, N);
        for t = 1:stp
            old_Emo = Total_Emo;

            % ===== 情绪统计：#A邻居、#AA对 =====
            % #A邻居：m * adj_A
            Num = double(m) * adj_A;                 % 1 x N
            % #AA对：对 (i,u,v) 列表做一次向量化统计
            AA = (m(U_list) & m(V_list));           % 1/0 列向量
            Num_delta = accumarray(I_list, double(AA), [N,1], @sum, 0).';  % 1 x N

            % 情绪累计
            Total_Emo = omega1*old_Emo + (1-omega1) .* ...
                        (0.5*(Num ./ k_i.') + 0.5*(Num_delta ./ k_i_delta_div.'));

            % ===== 上层 U→A（阈值、1-simplex、2-simplex） & A→U =====
            n = m;    % t+1 的上层
            U_mask = ~m;                 % 当前为 U 的节点
            % 1) 阈值触发
            adopt_by_thresh = U_mask & (old_Emo >= threshold);
            n(adopt_by_thresh) = true;

            % 2) 1-simplex 危害聚合（仍为U者）
            still_U = ~n;
            p1 = 1 - (1 - lamda1) .^ Num;      % 1 x N
            fire = (rand(1,N) < p1) & still_U;
            n(fire) = true;

            % 3) 2-simplex 危害聚合（仍为U者）
            still_U = ~n;
            p2 = 1 - (1 - lamda2) .^ Num_delta;
            fire2 = (rand(1,N) < p2) & still_U;
            n(fire2) = true;

            % 4) A→U 恢复
            rec = (rand(1,N) < delta) & m;
            n(rec) = false;

            % ===== 下层 S→G / G→S （危害聚合）=====
            y = x;    % t+1 的下层
            % 概率 p_beta(i)
            p_beta = beta_R * (1 + (gamma - 1) * double(m)); % m=1 → βγ, m=0 → β
            p_beta = min(1, max(0, p_beta));

            % G 邻居数量
            Gcnt = double(x) * adj_B;            % 1 x N
            % S→G 总概率
            pG = 1 - (1 - p_beta) .^ Gcnt;

            S_mask = ~x;
            infect = (rand(1,N) < pG) & S_mask;
            y(infect) = true;
            n(infect) = true;                    % UG→AG

            % G→S
            back = (rand(1,N) < mu) & x;
            y(back) = false;

            % 若仍为 G，则视为 AG（与你原逻辑一致）
            keepG = x & ~back;
            n(keepG) = true;

            % 收尾
            m = n;
            x = y;
        end%t
           ag_rep(rep) = mean(x);
            end

            % —— 写入本组合的 MC 统计量 ——
            AG_mean(chance, ye, l) = mean(ag_rep);
            AG_std(chance,  ye, l) = std(ag_rep, 0, 1);
        end
    end
end
% % 平均并画图
% ---------- 改造后的画图部分：Y 轴使用 lambda2 ----------


figure; hold on
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15);
set(gca,'TickLabelInterpreter','latex');

omega1_vals = linspace(1/termi, 1, termi);     % x 轴：lambda1
yeta_vals = linspace(1/termi, 1,termi);     % 用于计算 lambda2
tau0_vals =paramter;           % z 轴层：lambda_delta（逐层悬浮）
[MA, LAM2] = meshgrid(omega1_vals,yeta_vals);

z_gap = 0.02;  % 控制 λ1 层之间的间距，让它们有“浮空”效果

for idx = 1:length(tau0_vals)
    Z_layer =tau0_vals(idx);  % 当前层的 λ1 值，作为 Z 坐标
    Z = squeeze(AG_mean(idx, :, :));  % (λ2, ma)

    % 构建与 MA/LAM2 对应大小的 Z 面，值恒定为当前 λ1，表示“悬浮”
    Zplane = Z_layer * ones(size(MA));

    surf(MA, LAM2, Zplane, Z, 'EdgeColor', 'none', ...
         'FaceAlpha', 0.95, 'DisplayName', ['\tau0= ', num2str(Z_layer)])
end
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15);
set(gca, 'TickLabelInterpreter', 'latex');
  
xlabel('$\omega_1$','FontSize',15)
ylabel('$\eta_\tau$','FontSize',15)
zlabel('$\tau_{0}$ ','FontSize',15)
cb = colorbar;                             % 生成颜色条
cb.TickLabelInterpreter = 'latex';         % 刻度同样用 LaTeX（可选）
title(cb, '$\rho^{AG}$', 'Interpreter','latex', 'FontSize', 15);  % 标题放在色条上方

colormap turbo
view(45, 20)
grid on
save('AG_mean.mat','AG_mean');
saveas(gcf, 'mmc.fig');



%% ===== 本地函数 =====
function [I_list, U_list, V_list] = build_pair_list(triangles_A, N)
% 为每个节点 i 建立 (i,u,v) 列表，用于统计“与 i 同处三角形的另外两点”
    M = size(triangles_A,1);
    % 总长度 = 所有节点参与三角形次数之和 = 3*M
    I_list = zeros(3*M,1,'uint32');
    U_list = zeros(3*M,1,'uint32');
    V_list = zeros(3*M,1,'uint32');
    idx = 0;
    for m = 1:M
        tri = triangles_A(m,:); a = tri(1); b = tri(2); c = tri(3);
        I_list(idx+1:idx+3) = uint32([a b c]);
        U_list(idx+1:idx+3) = uint32([b a a]);
        V_list(idx+1:idx+3) = uint32([c c b]);
        idx = idx + 3;
    end
    if idx < numel(I_list)   % 理论上不会发生，防守式裁剪
        I_list = I_list(1:idx); U_list = U_list(1:idx); V_list = V_list(1:idx);
    end
end
