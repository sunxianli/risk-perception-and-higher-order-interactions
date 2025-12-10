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
lamda_delta=[0.8,1.6,2.4];
num=length(lamda_delta);
mu=0.8;
beta_R=0.4;
gamma=0.3;
stp=20;%时间长度
ter=20;%变量的取值个数
%情绪累计
omega1 = 0.8;

k_i = sum(adj_A, 2);  % 每个节点的度（1-simplex）
k_i(k_i == 0) = 1;  % 防止除以 0
k_i_delta = accumarray(triangles_A(:), 1, [N, 1]);%把 triangles_A 所有行的所有值展开成一个长向量 → 所有在三角形中出现的节点,对这些节点做“统计频率” → 每个节点参与了多少个三角形
avg_k_delta = mean(k_i_delta);
% 最终按公式 λ* = δ λ_δ / ⟨k^Δ⟩

 k_i_delta(k_i_delta == 0) = 1; %如果是单个节点则会出现为0，那么在分母上就会没有意义
%阈值部分定义
epsilon=rand(1, N);
yeta=0.05;
tau0=0.6;
threshold=tau0+yeta.*epsilon;
%最终的密度
rhoAG=zeros(num,ter,ter);


% —— 如有并行工具箱，尝试启动线程池（可选）——
if ~isempty(ver('parallel'))
    pp = gcp('nocreate');
    if isempty(pp)
        try, parpool('local'); catch, end
    end
end

% ========================= 主循环：扫 beta_R =========================
parfor chance=1:num;
    for xun = 1:ter
       delta=xun/ter;
       lamda2=(delta*lamda_delta(chance))/avg_k_delta;
      
    for la=1:ter
      lamda1=la/ter

    % —— 每个 beta_R 的初值（沿用你的设定）——
    PUS = 0.8 * ones(1, N);
    PAS = 0.1 * ones(1, N);
    PAG = 0.1 * ones(1, N);

    % 每个 beta_R 单独的情绪累计
    Total_Emo = zeros(stp, N);

    % ===================== 时间演化 =====================
    for t = 1:stp-1
        % —— 当前觉察概率（用于信息层传播 & 情绪累计）——
        PA = PAG + PAS;          % 1×N

        % ================= r2: 边上传播的连乘 =================
        % r2(i) = Π_{j∈N_A(i)} (1 - λ1 * PA(j))
        % 用 log-和 trick：prod = exp(sum(log(...)))
        log1 = log( max(1 - lamda1 .* PA(:), realmin) );    % N×1（数值稳定）
        r2   = exp( adj_A * log1 ).';                       % 1×N

        % ================= r3: 三角形(2-simplex)连乘 =================
        % 对每个三角形一次性计算对三个顶点的“因子”，再按节点累积（sum of logs -> exp）
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

        % ================= 情绪累计的统计量 =================
        % 一阶邻居“觉察概率和”
        Num_i_col = adj_A * PA.';                            % N×1
        % 三角形内“两人觉察乘积”的和
        s1 = accumarray(triangles_A(:,1), Ptri(:,2).*Ptri(:,3), [N,1], @sum, 0);
        s2 = accumarray(triangles_A(:,2), Ptri(:,1).*Ptri(:,3), [N,1], @sum, 0);
        s3 = accumarray(triangles_A(:,3), Ptri(:,1).*Ptri(:,2), [N,1], @sum, 0);
        Num_delta_i = s1 + s2 + s3;                          % N×1

        % ================= 接触层的连乘（感染成功的 complement） =================
        % qU(i) = Π_{j∈N_B(i)} (1 - β_R * PAG(j))
        % qA(i) = Π_{j∈N_B(i)} (1 - β_R * γ   * PAG(j))
        logu = log( max(1 - beta_R * PAG(:),           realmin) );
        loga = log( max(1 - beta_R * gamma * PAG(:),   realmin) );
        qU   = exp(adj_B * logu).';                          % 1×N
        qA   = exp(adj_B * loga).';                          % 1×N

        % ================= 情绪累计 & 触发门槛 =================
        Total_Emo(t+1,:) = omega1 * Total_Emo(t,:) + ...
            (1-omega1) * ( 0.5 * (Num_i_col ./ k_i).' + 0.5 * (Num_delta_i ./ k_i_delta).' );

        r1 = double(Total_Emo(t, :) < threshold);            % 1×N
        r_total = r1 .* r2 .* r3;                            % 你的“全都不触发”合取

        % ================= 状态更新（沿用你的三式） =================
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
  rhoAG(chance,xun,la) = mean(PAG);
    end%lamda 
    end%delta
end%lamdadelta

lamda_vals=linspace(1/ter,1,ter);
delta_vals=linspace(1/ter,1,ter);
lamda_delta_vals =lamda_delta;  % [0.3, 0.6, 0.9]

[MA, LAM2] = meshgrid(lamda_vals,delta_vals);

% 每个 λ1 对应一个水平图层（固定在不同 z 高度）
figure;
hold on

z_gap = 0.02;  % 控制 λ1 层之间的间距，让它们有“浮空”效果

for idx = 1:length(lamda_delta_vals)
    Z_layer =lamda_delta_vals(idx);  % 当前层的 λ1 值，作为 Z 坐标
    Z = squeeze(rhoAG(idx, :, :));  % (λ2, ma)

    % 构建与 MA/LAM2 对应大小的 Z 面，值恒定为当前 λ1，表示“悬浮”
    Zplane = Z_layer * ones(size(MA));

    surf(MA, LAM2, Zplane, Z, 'EdgeColor', 'none', ...
         'FaceAlpha', 0.95, 'DisplayName', ['\lamda_delta = ', num2str(Z_layer)])
end
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15);
set(gca, 'TickLabelInterpreter', 'latex');
  
xlabel('$\lambda$','FontSize',15)
ylabel('$\delta$','FontSize',15)
zlabel('$\lambda_{\delta}$ ','FontSize',15)
cb = colorbar;                             % 生成颜色条
cb.TickLabelInterpreter = 'latex';         % 刻度同样用 LaTeX（可选）
title(cb, '$\rho^{AG}$', 'Interpreter','latex', 'FontSize', 15);  % 标题放在色条上方

colormap turbo
view(45, 20)
grid on

save('rhoAG.mat','rhoAG') 
saveas(gcf, 'mmc.fig')
