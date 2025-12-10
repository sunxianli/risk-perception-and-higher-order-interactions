clc; clear;
%%加速版本
%% —— 载入数据 ——
data = load('RSC_ER.mat');
adj_A = sparse(data.net.adj_A);        % 用稀疏矩阵
adj_B = sparse(data.net.adj_B);
neighbors_A = data.net.neighbors_A;    %# 仍保留，情绪统计用不到它也行
triangles_A = data.net.triangles_A;
N = length(adj_A);

%% —— 参数 ——
stp    = 40;
MC_rep = 20;
termi  = 40;

lamda1 = 0.15;
delta  = 0.6;
lamda_deta = 1.6;
mu     = 0.8;
gamma  = 0.3;
 beta_R=0.4;
% omega1 = 0.8; 

% 节点度/三角形度
k_i = full(sum(adj_A, 2));  k_i(k_i==0) = 1;
k_i_delta_raw = accumarray(triangles_A(:), 1, [N,1]);      % 原始计数
avg_k_delta   = mean(k_i_delta_raw);
lamda2 = (delta * lamda_deta) / max(avg_k_delta, eps);
k_i_delta_div = k_i_delta_raw;  k_i_delta_div(k_i_delta_div==0) = 1;

% 阈值
rng(1);                       % 固定随机种子便于复现实验
epsilon   = rand(1, N);
yeta      = 0.05;
tau0      = 0.6;
threshold = tau0 + yeta .* epsilon;

% —— 预处理：构建 (i, u, v) 列表，用于 2-simplex 统计 ——
[I_list, U_list, V_list] = build_pair_list(triangles_A, N);
% I_list/U_list/V_list 均为列向量，长度 sum(k_i_delta_raw)

%% —— 结果容器 ——
BETA_AG = zeros(MC_rep, termi);
EMO_T_SURF = zeros(stp, termi, MC_rep);   % [t × omega1 × rep]

parfor rep = 1:MC_rep
    local_AG = zeros(1, termi);
    local_emo = zeros(stp, termi);      % 本 rep 的 (t, omega1) 面
    for l = 1:termi
        omega1 = l / termi;
        omega2=1-omega1;
        % 初始状态（逻辑型更快）
        csmd_G = 0.1; csmd_A = 0.2;
        x = rand(1, N) < csmd_G;      % G
        m = rand(1, N) < csmd_A;      % A
        m(x) = true;                  % AG 覆盖
        x(~m) = false;                % 无 A 则无 G

        Total_Emo = zeros(1, N);
        emo_mean_t = zeros(stp, 1);    % 记录每个 t 的平均情绪
        for t = 1:stp
            old_Emo = Total_Emo;

            % ===== 情绪统计：#A邻居、#AA对 =====
            % #A邻居：m * adj_A
            Num = double(m) * adj_A;                 % 1 x N
            % #AA对：对 (i,u,v) 列表做一次向量化统计
            AA = (m(U_list) & m(V_list));           % 1/0 列向量
            Num_delta = accumarray(I_list, double(AA), [N,1], @sum, 0).';  % 1 x N

            % 情绪累计
            Total_Emo = omega1*old_Emo + omega2.* ...
                        (0.5*(Num ./ k_i.') + 0.5*(Num_delta ./ k_i_delta_div.'));

            % ===== 关键：记录每个 t 的平均 =====
            emo_mean_t(t) = mean(Total_Emo);
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
        end

        BETA_A = mean(m);
        local_AG(l) = mean(x);
        local_emo(:, l) = emo_mean_t;   % 把该 ω1 下的时间序列塞进列 l
    end

  BETA_AG(rep, :) = local_AG;
   EMO_T_SURF(:, :, rep) = local_emo;  % 写回本 rep 的整张面
end
% ρ_AG 是 1×termi
% —— 对 MC 取均值（也可算 std 画误差带/区间）——
EMO_T_mean = mean(EMO_T_SURF, 3);       % [stp × termi]
% EMO_T_std  = std(EMO_T_SURF, 0, 3);   % 需要的话

t_axis      = 1:stp;
omega1_axis = linspace(1/termi, 1, termi);

% 3D 曲面：X=t, Y=ω1, Z=平均 Total_Emo
[Tgrid, W1grid] = meshgrid(t_axis, omega1_axis);  % 大小 [termi × stp]
Z = EMO_T_mean.';                                 % 转置成 [termi × stp]

figure; hold on; box on; grid on;
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15,'TickLabelInterpreter','latex');

hs = surf(Tgrid, W1grid, Z, Z);     % 用 Z 着色
set(hs,'EdgeColor','none','FaceColor','interp','CDataMapping','scaled');
colormap(turbo); cb=colorbar; title(cb,'$\overline{Total\_Emo}$','Interpreter','latex');

xlabel('$t$'); ylabel('$\omega_1$'); zlabel('$\overline{Total\_Emo}$');
view(-35, 40); axis tight;

save('EMO_T_mean.mat','EMO_T_mean') 
saveas(gcf, 'mc.fig')


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
