%参数设置，MMCA数值解，mu关于rho的变化，
%引入活跃度矩阵
data = load('RSC_BA.mat');
adj_A = data.net.adj_A;
adj_B = data.net.adj_B;
neighbors_A = data.net.neighbors_A;
triangles_A = data.net.triangles_A;
neighbors_B = data.net.neighbors_B;
%定义网络中参数的值
N = length(adj_A);
lamda1 = 0.15;
lamda_deta = 1.6;
delta = 0.6;
% mu=0.4;
beta_R=0.6;
parmters=[0.3,0.6,0.9];
choice=length(parmters);
% gamma=0.7;
stp=20;%时间长度
ter=40;%变量的取值个数
%情绪累计
omega1 = 0.8;

k_i = sum(adj_A, 2);  % 每个节点的度（1-simplex）
k_i(k_i == 0) = 1;  % 防止除以 0
k_i_delta = accumarray(triangles_A(:), 1, [N, 1]);%把 triangles_A 所有行的所有值展开成一个长向量 → 所有在三角形中出现的节点,对这些节点做“统计频率” → 每个节点参与了多少个三角形
avg_k_delta = mean(k_i_delta);
% 最终按公式 λ* = δ λ_δ / ⟨k^Δ⟩
lamda2=(delta*lamda_deta)/avg_k_delta;
k_i_delta(k_i_delta == 0) = 1; %如果是单个节点则会出现为0，那么在分母上就会没有意义

%阈值部分定义
epsilon=rand(1, N);
yeta=0.05;
tau0=0.6;
%最终的密度
rhoAG=zeros(1,ter);
rhoAS=zeros(1,ter);
rhoUS=zeros(1,ter); 
parfor ga=1:choice
    gamma=parmters(ga);
for xun = 1:ter
    mu=xun/ter;
 % —— 每个 beta_R 迭代的「本地」状态（先赋值，再用）——
    PUS = 0.8*ones(1,N);
    PAS = 0.1*ones(1,N);
    PAG = 0.1*ones(1,N);
     
    % 每个迭代单独的情绪累计（避免 parfor 共享写）
    Total_Emo = zeros(stp, N);
% %计算马氏链
for t=1:stp-1
    % 初始化 Num_i 和 Num^m_i
    Num_i = zeros(N,1);       % 1-simplex 中知晓邻居数
    Num_delta_i = zeros(N,1);     % 2-simplex 中满足传播条件的个数
    PA=PAG+PAS;%每个时刻A节点是概率   
  
    %% ---- r_i| 和 r_iΔ ----
    r1=ones(1, N);
    r2=ones(1, N);
    r3=ones(1, N);
    qU=ones(1,N);
    qA=ones(1,N);
    r_total=ones(1, N);

    %% 计算r2
    for i =1:N
        neiA = neighbors_A{i};
            if ~isempty(neiA)
                % 连乘形式
                r2(i)   = prod(1 - PA(neiA) * lamda1);
                Num_i(i)= sum(PA(neiA));% 感知状态邻居的概率密度
            end
        %计算r3
        for t_idx = 1:size(triangles_A,1)
            tri = triangles_A(t_idx,:);
              if any(tri==i)
                 others=tri(tri~=i);
                 u=others(1); 
                 v=others(2);
                 r3(i)=r3(i)*(1-PA(u)*PA(v)*lamda2);
                 Num_delta_i(i)=Num_delta_i(i)+PA(u)*PA(v);
              end
        end

        %计算qu,qA
        neiB = neighbors_B{i};
        if ~isempty(neiB)
           qU(i) = prod(1-PAG(neiB)*beta_R);
           qA(i) = prod(1-PAG(neiB)*beta_R*gamma);
        end
      end
       
       %累计情绪的表达式
       Total_Emo(t+1,:)=omega1*Total_Emo(t,:)+(1-omega1).*(0.5*(Num_i./k_i)'+0.5*(Num_delta_i./k_i_delta)');
 
       %阈值计算
    threshold=tau0+yeta.*epsilon;
    r1 = double(Total_Emo(t, :)<threshold); 
    % r_total=omega2.*r1+(1 - omega2).*omega3.*r2+(1 - omega2).*(1 - omega3).*r3;
    r_total=r1.*r2.*r3;
    %开始进行节点的迭代,这里不涉及i，而是用整个矩阵计算   
    PUS_UPDATE=PUS.*r_total.*qU+PAS.*delta.*qU+PAG.*mu.*delta;
    PAS_UPDATE=PUS.*(1-r_total).*qA+PAS.*(1-delta).*qA+PAG.*(1-delta).*mu;
    PAG_UPDATE=PUS.*r_total.*(1-qU)+PUS.*(1-r_total).*(1-qA)+PAS.*(1-delta).*(1-qA)+PAS.*delta.*(1-qU)+...
              PAG.*(1-mu);
    %更新节点的状态
    PUS=PUS_UPDATE;
    PAG=PAG_UPDATE;
    PAS= PAS_UPDATE;

end
      
    % rhoUS(xun) = mean(PUS);
    % rhoAS(xun) = mean(PAS);
    rhoAG(ga,xun) = mean(PAG);
end
end
mu = (1:ter)/ter;
figure;
% 色盲友好配色（Okabe–Ito）
c.blue      = [0 114 178]/255;  % 蓝
c.vermilion = [213 94   0]/255; % 朱红
c.green     = [0 158 115]/255;  % 蓝绿
hold on;
box on;
grid off;
set(gca,'Fontsize',15);
set(gca, 'TickLabelInterpreter', 'latex');

plot(mu,rhoAG(1,:), '-o', 'Color', c.vermilion, 'MarkerSize', 4,'MarkerFaceColor', c.vermilion);
plot(mu, rhoAG(2,:), '-^', 'Color', c.blue,'MarkerSize', 4,      'MarkerFaceColor', c.blue);
plot(mu, rhoAG(3,:), '-v', 'Color', c.green,'MarkerSize', 4,     'MarkerFaceColor', c.green);

set(gcf,'DefaultTextInterpreter','latex');
xlabel('$\mu$','FontSize',15);ylabel('$\rho^{AG}$','FontSize',15);   
h=legend({'$\gamma=0.3(MC)$', '$\gamma=0.6(MC)$', '$\gamma=0.9(MC)$'}, 'Interpreter', 'latex', 'FontSize', 15);
set(h,'Interpreter','latex','FontSize',15)%,

% save('rhoUS.mat','rhoUS')
% save('rhoAS.mat','rhoAS')
save('rhoAG.mat','rhoAG') 
saveas(gcf, 'mmc.fig')

