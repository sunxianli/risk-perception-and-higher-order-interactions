%参数设置，MMCA数值解，lamda，lamda_delta,delta
%引入活跃度矩阵
data = load('RSC_ER.mat');
adj_A = data.net.adj_A;
adj_B = data.net.adj_B;
neighbors_A = data.net.neighbors_A;
triangles_A = data.net.triangles_A;
neighbors_B = data.net.neighbors_B;
%定义网络中参数的值
N = length(adj_A);
% lamda1 = 0.15;
% lamda_deta = 1.6;
% delta = 0.6;
lamda_deta=[0,1.2,2.4];
num=length(lamda_deta);
mu=0.4;
beta_R=0.4;
gamma=0.7;
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
%最终的密度
rhoAG=zeros(num,ter,ter);

parfor chance=1:num;
   
    for xun = 1:ter
       delta=xun/ter;
       lamda2=(delta*lamda_deta(chance))/avg_k_delta;
      
    for la=1:ter
      lamda=la/ter
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
    rhoAG(chance,xun,la) = mean(PAG);
    end%lamda 
    end%delta
end%lamdadelta

lamda_vals=linspace(1/ter,1,ter);
delta_vals=linspace(1/ter,1,ter);
lamda_delta_vals =lamda1_delta;  % [0.3, 0.6, 0.9]

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
         'FaceAlpha', 0.95, 'DisplayName', ['\lambda_delta = ', num2str(Z_layer)])
end
set(gcf,'DefaultTextInterpreter','latex');
set(gca,'Fontsize',15);
set(gca, 'TickLabelInterpreter', 'latex');
  
xlabel('$\lambda$','FontSize',15)
ylabel('$\delta$','FontSize',15)
zlabel('$\lambda_{delta}$ ','FontSize',15)
colorbar
colormap turbo
view(45, 20)
grid on
legend show

save('rhoAG.mat','rhoAG') 
saveas(gcf, 'mmc.fig')

