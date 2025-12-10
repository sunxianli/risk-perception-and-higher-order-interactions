function csv_to_net_unweighted(data_dir, out_mat)
% 把 upper_S1_edges / upper_S2_triangles / lower_edges / node_map
% 转成不带权的 net 结构并保存到 out_mat（默认 'multiplex_unweighted.mat'）
%
% 生成的 net 结构字段：
%   net.N
%   net.adj_A        % 上层的 1-simplex 邻接矩阵（0/1）
%   net.neighbors_A  % 上层的邻接表（cell）
%   net.triangles_A  % 上层的 2-simplex（三角列表，每行 i<j<k）
%   net.adj_B        % 下层的邻接矩阵（0/1）
%   net.neighbors_B  % 下层的邻接表（cell）
%
% 用法示例：
%   csv_to_net_unweighted('D:\senate_out', 'senate_multiplex_unweighted.mat')

if nargin < 1 || isempty(data_dir), data_dir = pwd; end
if nargin < 2 || isempty(out_mat),  out_mat  = fullfile(data_dir,'multiplex_unweighted.mat'); end

% ---- 1) 读取四个文件（支持 .csv 或 .xlsx） ----
tMap = read_any(data_dir, 'node_map');
tS1  = read_any(data_dir, 'upper_S1_edges');
tS2  = read_any(data_dir, 'upper_S2_triangles');
tLB  = read_any(data_dir, 'lower_edges');

% 统一列名到小写，方便索引
tMap.Properties.VariableNames = lower(tMap.Properties.VariableNames);
tS1.Properties.VariableNames  = lower(tS1.Properties.VariableNames);
tS2.Properties.VariableNames  = lower(tS2.Properties.VariableNames);
tLB.Properties.VariableNames  = lower(tLB.Properties.VariableNames);

% ---- 2) 基本清洗（只取整数 ID，忽略权重；去自环与重复） ----
S1 = unique(sort([tS1.u, tS1.v], 2), 'rows');
S1 = S1(S1(:,1)~=S1(:,2), :);

LB = unique(sort([tLB.u, tLB.v], 2), 'rows');
LB = LB(LB(:,1)~=LB(:,2), :);

if ~isempty(tS2)
    T = unique(sort([tS2.i, tS2.j, tS2.k], 2), 'rows');
    T = T( (T(:,1)~=T(:,2)) & (T(:,1)~=T(:,3)) & (T(:,2)~=T(:,3)), :);
else
    T = zeros(0,3);
end

% ---- 3) N 的确定（优先用 node_map；否则取边/三角的最大 ID） ----
if ismember('new_id', tMap.Properties.VariableNames)
    N = max(tMap.new_id);
else
    maxID = 0;
    if ~isempty(S1), maxID = max(maxID, max(S1(:))); end
    if ~isempty(LB), maxID = max(maxID, max(LB(:))); end
    if ~isempty(T),  maxID = max(maxID,  max(T(:)));  end
    N = maxID;
end
assert(N>0, '无法确定节点数 N，请检查输入文件。');

% ---- 4) 构建邻接矩阵（不考虑权重 → 0/1） ----
adj_A = zeros(N,N);
if ~isempty(S1)
    idx = sub2ind([N,N], [S1(:,1); S1(:,2)], [S1(:,2); S1(:,1)]);
    adj_A(idx) = 1;
end
adj_A(1:N+1:end) = 0;           % 清零对角

adj_B = zeros(N,N);
if ~isempty(LB)
    idx = sub2ind([N,N], [LB(:,1); LB(:,2)], [LB(:,2); LB(:,1)]);
    adj_B(idx) = 1;
end
adj_B(1:N+1:end) = 0;

% ---- 5) 上层三角的有效性（可选：确保三角的三条边都在上层边里） ----
if ~isempty(T)
    ok = false(size(T,1),1);
    for r = 1:size(T,1)
        i = T(r,1); j = T(r,2); k = T(r,3);
        ok(r) = adj_A(i,j) && adj_A(i,k) && adj_A(j,k);
    end
    triangles_A = T(ok, :);
else
    triangles_A = zeros(0,3);
end

% ---- 6) 邻接表（cell） ----
neighbors_A = arrayfun(@(i) find(adj_A(i,:)>0).', 1:N, 'UniformOutput', false).';
neighbors_B = arrayfun(@(i) find(adj_B(i,:)>0).', 1:N, 'UniformOutput', false).';

% ---- 7) 打包并保存 ----
net = struct();
net.N = N;
net.adj_A = adj_A;
net.neighbors_A = neighbors_A;
net.triangles_A = triangles_A;
net.adj_B = adj_B;
net.neighbors_B = neighbors_B;

save(out_mat, 'net', '-v7');    % 你的代码只要 load(out_mat) 然后用变量 net 即可
fprintf('[OK] 已保存：%s\n', out_mat);
fprintf('    N=%d, |E_A|=%d, |tri_A|=%d, |E_B|=%d\n', N, size(S1,1), size(triangles_A,1), size(LB,1));
end

% ============== 辅助：读任意 csv/xlsx ==============
function T = read_any(dirpath, basename)
% 优先读 CSV；没有就读 XLSX；两者都没有则报错
cand = { fullfile(dirpath, [basename,'.csv']), fullfile(dirpath, [basename,'.xlsx']) };
ok = cellfun(@(p) exist(p,'file')==2, cand);
if ~any(ok)
    error('未找到 %s.[csv|xlsx]（目录：%s）', basename, dirpath);
end
p = cand{find(ok,1,'first')};
T = readtable(p);
end
