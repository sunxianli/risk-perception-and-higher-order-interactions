%% —— 参数设置，MMCA 数值解，beta 关于 rho 的变化（矢量化加速版）——
clc; clear;

% —— 引入网络数据（要求 RSC_ER.mat 内包含 net.adj_A / net.adj_B / net.triangles_A）——
data = load('RSC_BA.mat');
adj_A = data.net.adj_A;             % 信息层邻接矩阵 (N×N)
adj_B = data.net.adj_B;             % 接触层邻接矩阵 (N×N)
triangles_A = data.net.triangles_A; % 信息层三角形列表 (T×3, 每行是一个三角形的三个节点索引)

% —— 稀疏化以提速 —— 
adj_A = sparse(adj_A);
adj_B = sparse(adj_B);

% —— 基本统计量 —— 
N = size(adj_A, 1);
delta = 0.6;
mu=0.4;
% beta_R=0.4;
gamma=0.7;
stp=20;%时间长度
ter=40;%变量的取值个数
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
bata_c=zeros(ter,ter);

% ========================= 主循环：扫 beta_R =========================
parfor xun = 1:ter
     lamda_deta=0.2*xun;
     lamda2=(delta*lamda_deta)/avg_k_delta;
     H=zeros(N,N);
    for la= 1:ter
        lamda1=la/ter;

    PAS = 0.1 * ones(1, N);
  

    % 每个 beta_R 单独的情绪累计
    Total_Emo = zeros(stp, N);

    % ===================== 时间演化 =====================
    for t = 1:stp-1
        % —— 当前觉察概率（用于信息层传播 & 情绪累计）——
        PA =  PAS;          % 1×N

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

   
        % ================= 情绪累计 & 触发门槛 =================
        Total_Emo(t+1,:) = omega1 * Total_Emo(t,:) + ...
            (1-omega1) * ( 0.5 * (Num_i_col ./ k_i).' + 0.5 * (Num_delta_i ./ k_i_delta).' );

        r1 = double(Total_Emo(t, :) < threshold);            % 1×N
        r_total = r1 .* r2 .* r3;                            % 你的“全都不触发”合取

        % ================= 状态更新（沿用你的三式） =================
       
        % PAS_UPDATE = PUS .* (1 - r_total) .* qA + PAS .* (1 - delta) .* qA + PAG .* (1 - delta) .* mu;
         PAS_UPDATE=(1-PAS).*(1-r_total)+PAS.*(1-delta);
        % 写回
       
        PAS = PAS_UPDATE;
        
    end

 %求阈值
        for i =1:N
            for j=1:N;
                H(i,j)=(1-(1-gamma)*PAS(i))*adj_B(j,i);
            end
        end
        [m1,n1]=eig(H);
        tez=max(diag(n1));
        bata_c(xun,la)= mu/tez;
    end%lamda 
    end%delta
%lamdadelta

% hold on;
% box on;
% grid off;
% set(gca,'TickLabelInterpreter','latex');
% set(gca,'Fontsize',15);
% mu=linspace(0.05, 1, 20);
% lamda_deta = 0.2*linspace(0.05, 1, 20);
% [mu_grid, lamda_grid] = meshgrid(mu, lamda_deta);
% surf(mu,lamda_deta,bata_c);
% shading interp;
% colormap('jet');
% set(gcf,'DefaultTextInterpreter','latex');
% xlabel('$\mu$');
% ylabel('$\lamda_\deta$');
% zlabel('\beta_c');
% 
% colorbar;
% view(-30, 45);  % 调整视角
% 
% 
% save('bata_c.mat','bata_c') 
% saveas(gcf, 'bata_c.fig')
% ===== 绘图（匹配 bata_c 的 40×40 维度）=====
lambda1_vals     = (1:ter)/ter;       % 列方向（与 la 对应）
lambdaDelta_vals = 0.2*(1:ter);       % 行方向（与 xun 对应）
[X, Y] = meshgrid(lambda1_vals, lambdaDelta_vals);

figure;
surf(X, Y, bata_c);
shading interp;
colormap('jet');
colorbar;

set(gca,'TickLabelInterpreter','latex','FontSize',15);
set(gcf,'DefaultTextInterpreter','latex');
xlabel('$\lambda_1$');
ylabel('$\lambda_\delta$');
zlabel('$\beta_c$');
view(-30,45);

save('bata_c.mat','bata_c') 
saveas(gcf, 'bata_c.fig')
