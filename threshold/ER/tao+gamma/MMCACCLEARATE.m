%% —— 参数设置，MMCA 数值解，beta 关于 rho 的变化（矢量化加速版）——
clc; clear;
%之前也有优化的，但这个版本最快
% —— 引入网络数据（要求 RSC_BA.mat 内包含 net.adj_A / net.adj_B / net.triangles_A）——
data = load('RSC_ER.mat');
adj_A = spones(sparse(data.net.adj_A));   % 稀疏 & 01 化
adj_B = spones(sparse(data.net.adj_B));   % 稀疏 & 01 化
triangles_A = data.net.triangles_A;

% —— 基本统计量 —— 
N     = size(adj_A, 1);
delta = 0.6;
mu    = 0.8;
% gamma = 0.7;
stp   = 20;      % 时间长度
ter   = 40;      % 变量的取值个数
lamda1=0.15;
% omega1 = 0.8;

% 节点度（信息层 1-simplex）
k_i = full(sum(adj_A,2));
k_i(k_i==0) = 1;

% 节点参与三角形个数（信息层 2-simplex）
idx1 = triangles_A(:,1);
idx2 = triangles_A(:,2);
idx3 = triangles_A(:,3);
k_i_delta = accumarray(triangles_A(:), 1, [N,1]);
avg_k_delta = mean(k_i_delta);
k_i_delta(k_i_delta==0) = 1;  % 防除零
lamda_deta=1.6;
lamda2=(delta*lamda_deta)/avg_k_delta;

% 阈值
epsilon   = rand(1, N);
yeta      = 0.05;
tau0      = 0.6;
threshold = tau0 + yeta.*epsilon;

% 结果矩阵
bata_c = zeros(ter, ter);

% 备用：B 的转置（稀疏）
adj_B_T = adj_B.';

% ===== 主循环：扫 λ_δ（外层并行）=====
parfor xun = 1:ter
       gamma=xun/ter;
    row_vals = zeros(1, ter);  % 本行结果缓存，减少 parfor 写冲突
    for la = 1:ter
        omega1= la/ter;
      
        % 初值
        PAS = 0.1 * ones(1, N);   % 觉察概率
        emo = zeros(1, N);        % 情绪累计（滚动向量）

        % —— 时间演化 —— 
        for t = 1:stp-1
            PA = PAS;  % 1×N

            % r2: 边上传播连乘  (log-sum trick)
            log1 = log( max(1 - lamda1 .* PA(:), realmin) );  % N×1
            r2   = exp( adj_A * log1 ).';                     % 1×N

            % 三角形内两两乘积
            % 为了减少重复索引，先取出对应概率
            P_i1 = PA(idx1);  P_i2 = PA(idx2);  P_i3 = PA(idx3);

            P12 = P_i1 .* P_i2;
            P13 = P_i1 .* P_i3;
            P23 = P_i2 .* P_i3;

            % r3: 三角形连乘（log-sum trick + accumarray）
            sumlog = zeros(N,1);
            sumlog = sumlog + accumarray(idx1, log(max(1 - lamda2.*P23, realmin)), [N,1]);
            sumlog = sumlog + accumarray(idx2, log(max(1 - lamda2.*P13, realmin)), [N,1]);
            sumlog = sumlog + accumarray(idx3, log(max(1 - lamda2.*P12, realmin)), [N,1]);
            r3 = exp(sumlog).';   % 1×N

            % 情绪累计项：一阶邻居和 & 三角形“两人觉察乘积”的和
            Num_i_col  = adj_A * PA.';  % N×1
            s_delta    = zeros(N,1);
            s_delta    = s_delta + accumarray(idx1, P23, [N,1]);
            s_delta    = s_delta + accumarray(idx2, P13, [N,1]);
            s_delta    = s_delta + accumarray(idx3, P12, [N,1]);

            emo = omega1 * emo + ...
                  (1-omega1) * ( 0.5*(Num_i_col./k_i) + 0.5*(s_delta./k_i_delta) ).';  % 1×N

            % 门槛与总“未触发”概率（合取）
            r1       = double(emo < threshold);   % 1×N
            r_total  = r1 .* r2 .* r3;

            % 状态更新（沿用你的两项式）
            PAS = (1 - PAS) .* (1 - r_total) + PAS .* (1 - delta);
        end

        % ===== 构造 H 并求谱半径（最大特征值）=====
        % H(i,j) = (1 - (1-gamma)*PAS(i)) * adj_B(j,i)
        % 等价于：H = diag(s) * adj_B'，对 adj_B' 做“逐行缩放”
        svec = 1 - (1 - gamma) * PAS(:);     % N×1
        H    = bsxfun(@times, adj_B_T, svec); % 稀疏行缩放

        % 稀疏特征值（Perron 根），非对称选 'lm'
        tez = real(eigs(H, 1, 'lm'));

        % 防数值问题
        if ~isfinite(tez) || tez <= 0
            tez = eps;
        end

        row_vals(la) = mu / tez;
    end

    bata_c(xun, :) = row_vals;
end

% ===== 绘图（与原版一致）=====
omega1_vals =(1:ter)/ter;       % 列方向（与 la 对应）
gamma_vals =(1:ter)/ter;       % 行方向（与 xun 对应）
[X, Y]=meshgrid(omega1_vals,gamma_vals);

figure;
surf(X, Y, bata_c);
shading interp; colormap('jet'); colorbar;
set(gca,'TickLabelInterpreter','latex','FontSize',15);
set(gcf,'DefaultTextInterpreter','latex');
xlabel('$\omega_1$'); ylabel('$\gamma$'); zlabel('$\beta_c$');
view(-30,45);

save('bata_c.mat','bata_c');
saveas(gcf, 'bata_c.fig');
