%% —— 参数设置，MMCA 数值解，beta 关于 rho 的变化（矢量化加速版）——
clc; clear;

% —— 引入网络数据（要求 RSC_ER.mat 内包含 net.adj_A / net.adj_B / net.triangles_A）——
data = load('RSC_ER.mat');
adj_A = data.net.adj_A;             % 信息层邻接矩阵 (N×N)
adj_B = data.net.adj_B;             % 接触层邻接矩阵 (N×N)
triangles_A = data.net.triangles_A; % 信息层三角形列表 (T×3, 每行是一个三角形的三个节点索引)

% —— 稀疏化以提速 —— 
adj_A = sparse(adj_A);
adj_B = sparse(adj_B);

% —— 基本统计量 —— 
N = size(adj_A, 1);

% 扫描参数
lamda_delta = [0, 1.2, 2.4];  % 你要对比的 λ_Δ 集合
num = length(lamda_delta);

mu     = 0.4;
beta_R = 0.4;
gamma  = 0.7;
stp    = 20;   % 时间长度
ter    = 20;   % λ 与 δ 的取值个数（都扫 1/ter : 1）

% —— 情绪累计参数 —— 
omega1 = 0.8;

% —— 度与三角形参与度 —— 
k_i = sum(adj_A, 2);           % 每个节点的一阶度 (N×1)
k_i(k_i == 0) = 1;             % 防止除以 0

% 每个节点参与的三角形个数（按出现次数计数）
if isempty(triangles_A)
    k_i_delta = zeros(N,1);
else
    k_i_delta = accumarray(triangles_A(:), 1, [N, 1], @sum, 0);
end
avg_k_delta = mean(k_i_delta);
k_i_delta(k_i_delta == 0) = 1; % 防止情绪累计分母为 0

% —— 阈值部分定义（改名 thr，避免与工具箱函数重名）——
rng(1);                         % 固定随机数种子，结果可复现（可选）
epsilon = rand(1, N);
yeta    = 0.05;
tau0    = 0.6;
thr     = tau0 + yeta .* epsilon;   % 1×N

% —— 结果容器（注意是 3D：lamda_delta × delta × lambda1）——
rhoAG = zeros(num, ter, ter);

% —— 如有并行工具箱，尝试启动线程池（可选）——
if ~isempty(ver('parallel'))
    pp = gcp('nocreate');
    if isempty(pp)
        try, parpool('local'); catch, end
    end
end

% ========================= 主循环：扫 lamda_delta、delta、lambda1 =========================
parfor chance = 1:num
    % 为每个 lamda_delta 独立开一个局部缓冲，避免 parfor 切片冲突
    rhoAG_local = zeros(ter, ter);
    lamda_delta_val = lamda_delta(chance);

    for xun = 1:ter
        delta = xun / ter;
        % λ* = δ * λ_Δ / ⟨k^Δ⟩
        lamda2 = (delta * lamda_delta_val) / max(avg_k_delta, eps);

        for la = 1:ter
            lamda1 = la / ter;

            % —— 初值 —— 
            PUS = 0.8 * ones(1, N);
            PAS = 0.1 * ones(1, N);
            PAG = 0.1 * ones(1, N);

            % 每个 (δ, λ1) 单独的情绪累计
            Total_Emo = zeros(stp, N);

            % ===================== 时间演化 =====================
            for t = 1:stp-1
                % —— 当前觉察概率（用于信息层传播 & 情绪累计）——
                PA = PAG + PAS;          % 1×N

                % ================= r2: 边上传播的连乘 =================
                % r2(i) = Π_{j∈N_A(i)} (1 - λ1 * PA(j)) = exp( A * log(...) )
                log1 = log( max(1 - lamda1 .* PA(:), realmin) );    % N×1
                r2   = exp( adj_A * log1 ).';                       % 1×N

                % ================= r3: 三角形(2-simplex)连乘 =================
                if isempty(triangles_A)
                    r3 = ones(1, N);   % 没有三角形时，r3 恒为 1
                    Num_delta_i = zeros(N,1);
                else
                    Ptri = PA(triangles_A);     % T×3
                    f1 = 1 - lamda2 .* Ptri(:,2) .* Ptri(:,3);   % 贡献给 triangles_A(:,1)
                    f2 = 1 - lamda2 .* Ptri(:,1) .* Ptri(:,3);   % 贡献给 triangles_A(:,2)
                    f3 = 1 - lamda2 .* Ptri(:,1) .* Ptri(:,2);   % 贡献给 triangles_A(:,3)

                    % 用对数相加避免下溢：prod_i f = exp(sum(log f))
                    sumlog1 = accumarray(triangles_A(:,1), log(max(f1, realmin)), [N,1], @sum, 0);
                    sumlog2 = accumarray(triangles_A(:,2), log(max(f2, realmin)), [N,1], @sum, 0);
                    sumlog3 = accumarray(triangles_A(:,3), log(max(f3, realmin)), [N,1], @sum, 0);
                    r3_vec  = exp(sumlog1 + sumlog2 + sumlog3);         % N×1
                    r3      = r3_vec.';                                  % 1×N

                    % 三角形内“两人觉察乘积”的和（用于情绪累计）
                    s1 = accumarray(triangles_A(:,1), Ptri(:,2).*Ptri(:,3), [N,1], @sum, 0);
                    s2 = accumarray(triangles_A(:,2), Ptri(:,1).*Ptri(:,3), [N,1], @sum, 0);
                    s3 = accumarray(triangles_A(:,3), Ptri(:,1).*Ptri(:,2), [N,1], @sum, 0);
                    Num_delta_i = s1 + s2 + s3;                          % N×1
                end

                % 一阶邻居“觉察概率和”（用于情绪累计）
                Num_i_col = adj_A * PA.';                                % N×1

                % ================= 接触层的连乘（感染成功的 complement） =================
                % qU(i) = Π_{j∈N_B(i)} (1 - β_R * PAG(j))
                % qA(i) = Π_{j∈N_B(i)} (1 - β_R * γ   * PAG(j))
                logu = log( max(1 - beta_R * PAG(:),         realmin) );
                loga = log( max(1 - beta_R * gamma * PAG(:), realmin) );
                qU   = exp(adj_B * logu).';                              % 1×N
                qA   = exp(adj_B * loga).';                              % 1×N

                % ================= 情绪累计 & 触发门槛 =================
                Total_Emo(t+1,:) = omega1 * Total_Emo(t,:) + ...
                    (1-omega1) * ( 0.5 * (Num_i_col ./ k_i).' + 0.5 * (Num_delta_i ./ k_i_delta).' );

                r1      = double(Total_Emo(t, :) < thr);                 % 1×N
                r_total = r1 .* r2 .* r3;                                % “全都不触发”的合取

                % ================= 状态更新 =================
                PUS_UPDATE = PUS .* r_total .* qU + PAS .* delta .* qU + PAG .* mu .* delta;
                PAS_UPDATE = PUS .* (1 - r_total) .* qA + PAS .* (1 - delta) .* qA + PAG .* (1 - delta) .* mu;
                PAG_UPDATE = PUS .* r_total .* (1 - qU) + PUS .* (1 - r_total) .* (1 - qA) + ...
                             PAS .* (1 - delta) .* (1 - qA) + PAS .* delta .* (1 - qU) + ...
                             PAG .* (1 - mu);

                % 写回
                PUS = PUS_UPDATE;
                PAS = PAS_UPDATE;
                PAG = PAG_UPDATE;
            end

            % —— 记录稳态密度（时间结束后的最终一帧）——
            rhoAG_local(xun, la) = mean(PAG);
        end % lamda1
    end % delta

    % 回填到 parfor 外的 3D 数组
    rhoAG(chance, :, :) = rhoAG_local;
end % lamda_delta

% ========================= 作图 =========================
lamda_vals        = linspace(1/ter, 1, ter);   % λ1
delta_vals        = linspace(1/ter, 1, ter);   % δ
lamda_delta_vals  = lamda_delta;               % λ_Δ

[MA, LAM2] = meshgrid(lamda_vals, delta_vals);

figure; hold on
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15);
set(gca,'TickLabelInterpreter','latex');

for idx = 1:length(lamda_delta_vals)
    Z_layer = lamda_delta_vals(idx);             % 当前层的 λ_Δ 值，作为 Z 坐标（悬浮层）
    Z = squeeze(rhoAG(idx, :, :));               % 大小: δ×λ1

    % 构建与 MA/LAM2 同尺寸的 Z 面，值恒定为当前 λ_Δ
    Zplane = Z_layer * ones(size(MA));

    % 以 Z（rhoAG）作为颜色，Zplane 作为高度，实现“彩色悬浮层”
    surf(MA, LAM2, Zplane, Z, ...
        'EdgeColor','none', 'FaceAlpha',0.95, ...
        'DisplayName', sprintf('\\lambda_{\\Delta} = %.3g', Z_layer));
end

xlabel('$\lambda$','FontSize',15)
ylabel('$\delta$','FontSize',15)
zlabel('$\lambda_{\Delta}$','FontSize',15)
colorbar
colormap turbo
view(45, 20)
grid on
legend show

save('rhoAG.mat','rhoAG');
savefig('mmc.fig');
