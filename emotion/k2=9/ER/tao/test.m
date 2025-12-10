clc; clear;

%% —— 载入数据 —— 
data = load('RSC_ER.mat');
adj_A = sparse(data.net.adj_A);
adj_B = sparse(data.net.adj_B);
triangles_A = data.net.triangles_A;
N = length(adj_A);

%% —— 参数（保持 stp/MC 不变，只调其它旋钮）——
stp      = 40;      % 不加算力
MC_rep   = 20;      % 不加算力
termi    = 40;

% ↑↑ 放大上层影响 / 减弱下层喧宾（关键三处）
lamda1      = 0.35;   % 原 0.15 —— 提升 1-simplex
delta       = 0.6;
lamda_deta  = 3.2;    % 原 1.6 —— 使 lamda2 更大（见下）
mu          = 0.8;
gamma       = 0.3;    % 规则限制，不改
beta_R      = 0.2;    % 原 0.4 —— 降低下层基底 S->G

% 节点度/三角形度
k_i = full(sum(adj_A, 2));  k_i(k_i==0) = 1;
k_i_delta_raw = accumarray(triangles_A(:), 1, [N,1]);
avg_k_delta   = mean(k_i_delta_raw);
lamda2        = (delta * lamda_deta) / max(avg_k_delta, eps);   % 2-simplex 传染率
k_i_delta_div = k_i_delta_raw;  k_i_delta_div(k_i_delta_div==0) = 1;

% —— 预处理：构建 (i, u, v) 列表，用于 2-simplex 统计 ——
[I_list, U_list, V_list] = build_pair_list(triangles_A, N);

% —— 参数网格（不改范围） ——
paramter     = [0.3, 0.6, 0.8];
tau0_list    = paramter(:).';
yeta_list    = linspace(1/termi, 1, termi);
omega1_list  = linspace(1/termi, 1, termi);

% —— 结果容器（tau0 × yeta × omega1）——
AG_mean = zeros(numel(tau0_list), termi, termi);
AG_std  = zeros(size(AG_mean));

% —— 轻降阈值：缩放 epsilon，便于过阈 —— 
rng(42);
EPS_SCALE = 0.6;                 % <—— 0<scale<=1；越小越容易过阈
epsilon   = EPS_SCALE * rand(1, N);

%% —— 主循环 —— 
for a = 1:numel(tau0_list)
    tau0 = tau0_list(a);
    for b = 1:termi
        yeta = yeta_list(b);
        threshold = tau0 + yeta .* epsilon;   % 缩放后的阈值

        for c = 1:termi
            omega1 = omega1_list(c);
            ag_rep = zeros(MC_rep, 1);

            parfor rep = 1:MC_rep
                rng(1000 + rep);              % 跨参数点复用同一批子种子（提升可比）

                % ====== 初始化（与原逻辑一致）======
                csmd_G = 0.1; csmd_A = 0.2;
                x = rand(1, N) < csmd_G;      % 下层 G
                m = rand(1, N) < csmd_A;      % 上层 A
                m(x) = true; x(~m) = false;   % 强耦合初始化

                Total_Emo = zeros(1, N);

                for t = 1:stp
                    old_Emo = Total_Emo;

                    % ===== 情绪统计：#A 邻居、#AA 对 =====
                    Num = double(m) * adj_A;           % 1 x N
                    AA  = (m(U_list) & m(V_list));     
                    Num_delta = accumarray(I_list, double(AA), [N,1], @sum, 0).';

                    % ===== 情绪累计 =====
                    Total_Emo = omega1*old_Emo + (1-omega1) .* ...
                                (0.5*(Num ./ k_i.') + 0.5*(Num_delta ./ k_i_delta_div.'));

                    % ===== 上层 U→A（阈值、1/2-simplex） & A→U =====
                    n = m;
                    U_mask = ~m;

                    % 阈值触发（受 EPS_SCALE 影响）
                    adopt_by_thresh = U_mask & (old_Emo >= threshold);
                    n(adopt_by_thresh) = true;

                    % 1-simplex（放大）
                    still_U = ~n;
                    p1 = 1 - (1 - lamda1) .^ Num;
                    fire = (rand(1,N) < p1) & still_U; 
                    n(fire) = true;

                    % 2-simplex（放大后的 lamda2）
                    still_U = ~n;
                    p2 = 1 - (1 - lamda2) .^ Num_delta;
                    fire2 = (rand(1,N) < p2) & still_U; 
                    n(fire2) = true;

                    % A→U
                    rec = (rand(1,N) < delta) & m; 
                    n(rec) = false;

                    % ===== 下层 S→G / G→S（保留强耦合；但 beta_R 降低）=====
                    y = x;
                    p_beta = beta_R * (1 + (gamma - 1) * double(m)); % m=1→βγ, m=0→β
                    p_beta = min(1, max(0, p_beta));
                    Gcnt = double(x) * adj_B;
                    pG   = 1 - (1 - p_beta) .^ Gcnt;

                    S_mask = ~x;
                    infect = (rand(1,N) < pG) & S_mask;
                    y(infect) = true;
                    n(infect) = true;           % UG→AG
                    back = (rand(1,N) < mu) & x;
                    y(back) = false;
                    keepG = x & ~back;
                    n(keepG) = true;

                    % 更新
                    m = n; x = y;
                end % t

                % —— 输出真正的 ρ^{AG}（如要看 G，占比，改成 mean(x)）——
                ag_rep(rep) = mean(m & x);
            end

            % MC 统计
            AG_mean(a, b, c) = mean(ag_rep);
            AG_std (a, b, c) = std(ag_rep, 0, 1);
        end
    end
end

%% —— 绘图（仅显示级插值，不改数据）——
figure; hold on
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15,'TickLabelInterpreter','latex');

omega1_vals = omega1_list;
yeta_vals   = yeta_list;
tau0_vals   = tau0_list;
[MA, LAM2]  = meshgrid(omega1_vals, yeta_vals);

for idx = 1:length(tau0_vals)
    Z_layer = tau0_vals(idx);
    Zplane  = Z_layer * ones(size(MA));
    Ccolor  = squeeze(AG_mean(idx, :, :));  % 颜色 = ρ^{AG}

    s = surf(MA, LAM2, Zplane, Ccolor);
    set(s,'EdgeColor','none','FaceColor','interp','FaceAlpha',0.95, ...
          'DisplayName', ['\tau_0= ', num2str(Z_layer)]);
end

xlabel('$\omega_1$','FontSize',15)
ylabel('$\eta_\tau$','FontSize',15)
zlabel('$\tau_0$','FontSize',15)
cb = colorbar; cb.TickLabelInterpreter = 'latex';
title(cb, '$\rho^{AG}$','Interpreter','latex','FontSize',15);
colormap turbo; view(45, 20); grid on

save('AG_mean.mat','AG_mean','AG_std');

%% ===== 本地函数 =====
function [I_list, U_list, V_list] = build_pair_list(triangles_A, N)
% 为每个节点 i 建立 (i,u,v) 列表，用于统计“与 i 同处三角形的另外两点”
    M = size(triangles_A,1);
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
    if idx < numel(I_list)
        I_list = I_list(1:idx); U_list = U_list(1:idx); V_list = V_list(1:idx);
    end
end
