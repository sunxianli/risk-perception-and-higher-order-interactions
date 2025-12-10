function build_senate_multiplex_from_uploaded
% 严格双层：上层 = Senate-committees (1/2-simplex)，下层 = Senate-bills (2-section)
% 兼容上传文本的多种格式：有/无ID、"LAST, FIRST [State]"、全大写姓、分隔符（空格/逗号/冒号/竖线/Tab）等。

clc;

%% 1) 路径（把 data_dir 改成放置四个 .txt 的目录；默认=脚本目录）
data_dir = fileparts(mfilename('fullpath'));

top_hyper_file = fullfile(data_dir,'hyperedges-house-committees.txt');
top_names_file = fullfile(data_dir,'node-names-house-committees.txt');
bot_hyper_file = fullfile(data_dir,'hyperedges-house-bills.txt');
bot_names_file = fullfile(data_dir,'node-names-house-bills.txt');

assert(exist(top_hyper_file,'file')==2, ['找不到文件: ', top_hyper_file]);
assert(exist(top_names_file,'file')==2, ['找不到文件: ', top_names_file]);
assert(exist(bot_hyper_file,'file')==2, ['找不到文件: ', bot_hyper_file]);
assert(exist(bot_names_file,'file')==2, ['找不到文件: ', bot_names_file]);

%% 2) 参数（可按需修改）
out_dir = data_dir;
PAIR_WEIGHT_MODE = 'invdeg';      % {'count','invdeg','invchoose2'}
TRI_WEIGHT_MODE  = 'invchoose3';  % {'count','invchoose3'}
ONLY_TRIANGLES_IN_UPPER = false;  % 若上层只用三角，设为 true

%% 3) 读取姓名表（鲁棒解析 + 规范化）
T_top = read_names_flexible(top_names_file);   % table: id, raw, first, last, canon
T_bot = read_names_flexible(bot_names_file);

fprintf('[INFO] 名单规模：committees=%d, bills=%d\n', height(T_top), height(T_bot));

% ID -> 规范化全名 的映射
id2canon_top = containers.Map('KeyType','int32','ValueType','char');
for i=1:height(T_top), id2canon_top(int32(T_top.id(i))) = T_top.canon{i}; end
id2canon_bot = containers.Map('KeyType','int32','ValueType','char');
for i=1:height(T_bot), id2canon_bot(int32(T_bot.id(i))) = T_bot.canon{i}; end

%% 4) 读取超边并转成“姓名超边”
Htop_ids = read_hyperedges_flexible(top_hyper_file);
Hbot_ids = read_hyperedges_flexible(bot_hyper_file);

Htop_name_full = ids_to_names(Htop_ids, id2canon_top);  % 上层：直接用 committees 的规范全名
Hbot_name_raw  = ids_to_names(Hbot_ids, id2canon_bot);  % 下层：先用 bills 的规范（可能只有姓或逗号格式）

% ——对齐两层姓名——
% 先用全名精确匹配；再用“唯一姓氏”做补救映射，把 bills 的名字替换成 committees 的规范全名
[Htop_name, Hbot_name, V] = align_two_layers(Htop_name_full, Hbot_name_raw, T_top, T_bot);

fprintf('[INFO] 对齐后共有 |V|=%d 个共同节点\n', numel(V));
assert(~isempty(V), '两层节点交集为空：请确认四个文件同属参议院，并放在同一目录。');

% 将两层的姓名超边重映射到 1..N
name2newid = containers.Map('KeyType','char','ValueType','int32');
for i=1:numel(V), name2newid(V{i}) = int32(i); end
Htop = remap_names_to_ids(Htop_name, name2newid);
Hbot = remap_names_to_ids(Hbot_name, name2newid);

%% 5) 上层：1-/2-simplex
edgeMap = containers.Map('KeyType','char','ValueType','double'); % 上层边
triMap  = containers.Map('KeyType','char','ValueType','double'); % 上层三角
for t=1:numel(Htop)
    e = Htop{t}; k = numel(e);
    if ~ONLY_TRIANGLES_IN_UPPER && k>=2
        edgeMap = add_pairs(edgeMap, e, pair_weight(k, PAIR_WEIGHT_MODE));
    end
    if k>=3
        triMap  = add_tris (triMap,  e, tri_weight(k, TRI_WEIGHT_MODE));
    end
end

%% 6) 下层：bills 的 2-section
botEdgeMap = containers.Map('KeyType','char','ValueType','double');
for t=1:numel(Hbot)
    e = Hbot{t}; k = numel(e);
    if k>=2
        botEdgeMap = add_pairs(botEdgeMap, e, pair_weight(k, PAIR_WEIGHT_MODE));
    end
end

%% 7) 导出
% 节点映射（规范化全名）
fid = fopen(fullfile(out_dir,'node_map.csv'),'w'); fprintf(fid,"new_id,name\n");
for i=1:numel(V), fprintf(fid,"%d,%s\n", i, V{i}); end; fclose(fid);

% 上层 1-simplex
if ~ONLY_TRIANGLES_IN_UPPER
    [U,Vv,W] = map_to_arrays2(edgeMap);
    writetable(table(U,Vv,W,'VariableNames',{'u','v','weight'}), fullfile(out_dir,'upper_S1_edges.csv'));
else
    writetable(table([],[],[],'VariableNames',{'u','v','weight'}), fullfile(out_dir,'upper_S1_edges.csv'));
end

% 上层 2-simplex
[I,J,K,W2] = map_to_arrays3(triMap);
writetable(table(I,J,K,W2,'VariableNames',{'i','j','k','weight'}), fullfile(out_dir,'upper_S2_triangles.csv'));

% 下层 边
[Lu,Lv,Lw] = map_to_arrays2(botEdgeMap);
writetable(table(Lu,Lv,Lw,'VariableNames',{'u','v','weight'}), fullfile(out_dir,'lower_edges.csv'));

fprintf('[完成] |V|=%d, 上层边=%d, 上层三角=%d, 下层边=%d\n', numel(V), numel(U), numel(I), numel(Lu));
end % ===== 主函数 =====


%% ===================== 子函数（解析 / 对齐 / 构建） =====================

function T = read_names_flexible(fname)
% 返回 table: id, raw, first, last, canon（"first last" 小写）
lines = read_all_lines(fname);
ids = zeros(0,1,'int32'); raws = strings(0,1);

for i=1:numel(lines)
    s = strtrim(lines{i});
    if isempty(s), continue; end
    if startsWith(s,'#') || startsWith(s,'//') || startsWith(s,'%'), continue; end
    % 1) 形如 "id<sep>name"
    tok = regexp(s, '^\s*(-?\d+)\s*[,;:\|\t ]+\s*(.*\S)\s*$', 'tokens', 'once');
    if isempty(tok)
        % 2) 形如 "id  name"
        tok = regexp(s, '^\s*(-?\d+)\s+(.*\S)\s*$', 'tokens', 'once');
    end
    if ~isempty(tok)
        ids(end+1,1) = int32(str2double(tok{1}));
        raws(end+1,1) = string(tok{2});
        continue;
    end
    % 3) 只有名字（无 id）→ 赋顺序 id
    ids(end+1,1)  = int32(numel(ids)+1);
    raws(end+1,1) = string(s);
end

first = strings(size(raws)); last = strings(size(raws)); canon = strings(size(raws));
for i=1:numel(raws)
    [first(i), last(i)] = split_name(raws(i));
    if first(i) ~= ""
        canon(i) = lower(strtrim(first(i) + " " + last(i)));
    else
        canon(i) = lower(strtrim(last(i)));      % 只有姓时的退化形式
    end
end
T = table(ids, raws, first, last, canon, 'VariableNames', {'id','raw','first','last','canon'});
end

function [first,last] = split_name(s)
% 把 "LAST, FIRST M. [STATE]" 或 "First M Last Jr." 统一为 (first,last)
t = char(s); t = regexprep(t,'\[[^\]]*\]',' ');  % 去掉 [State]
t = regexprep(t,'[^\w, -]',' ');                 % 清理标点
t = regexprep(t,'\b(jr|sr|ii|iii|iv|v)\b',' ');  % 去后缀
t = regexprep(t,'\s+',' '); t = strtrim(t);
if isempty(t), first=""; last=""; return; end
if contains(t, ',')             % "last, first ..."
    p = strsplit(lower(t), ',');
    last  = strtrim(p{1});
    rest  = strtrim(p{2});
    toks  = strsplit(rest, ' ');
    first = strtrim(toks{1});
else                            % "first ... last" 或全大写一个词
    toks  = strsplit(lower(t),' ');
    if numel(toks)>=2
        first = strtrim(toks{1});
        last  = strtrim(toks{end});
    else
        first = "";             % 只有一个词：当作姓
        last  = strtrim(toks{1});
    end
end
end

function H = read_hyperedges_flexible(fname)
% 每行提取所有整数作为该超边（长度>=2 保留）
lines = read_all_lines(fname);
H = {};
for i=1:numel(lines)
    nums = regexp(lines{i}, '-?\d+', 'match');
    if ~isempty(nums)
        ids = unique(int32(str2double(nums)));
        if numel(ids) >= 2
            H{end+1} = ids(:)'; %#ok<AGROW>
        end
    end
end
end

function lines = read_all_lines(fname)
try
    L = readlines(fname); lines = cellstr(L);
catch
    fid = fopen(fname,'r'); assert(fid>0, ['无法打开 ',fname]);
    c = onCleanup(@() fclose(fid)); %#ok<NASGU>
    lines = {};
    t = fgetl(fid);
    while ischar(t)
        lines{end+1} = t; %#ok<AGROW>
        t = fgetl(fid);
    end
end
end

function Hn = ids_to_names(Hids, id2canon)
% 用 ID->规范名 映射把超边转成“姓名超边”
Hn = cell(size(Hids));
for m=1:numel(Hids)
    ids = Hids{m};
    names = {};
    for k=1:numel(ids)
        ik = ids(k);
        if isKey(id2canon, ik)
            names{end+1} = id2canon(ik); %#ok<AGROW>
        end
    end
    if ~isempty(names), names = unique(names); end
    if numel(names) >= 2, Hn{m} = sort(names); else, Hn{m} = {}; end
end
Hn = Hn(~cellfun(@isempty, Hn));
end

function [Htop_name, Hbot_name, V] = align_two_layers(Htop_name_full, Hbot_name_raw, T_top, T_bot)
% 先全名匹配；再用“唯一姓氏”把 bills 名替换为 committees 的规范全名
% ——全名集合
S_top = unique(cat(2,Htop_name_full{:}));
S_bot = unique(cat(2,Hbot_name_raw{:}));

% 1) 全名直接匹配
full_inter = intersect(S_top, S_bot, 'stable');
directSet  = containers.Map(full_inter, full_inter); % 直接映射

% 2) 唯一姓氏映射：两边都只出现一次的姓
% top: 姓 -> 该人的规范全名（仅当此姓在 top 中唯一）
[uniqLastTop, last2canonTop] = unique_last_to_canon(T_top);
[uniqLastBot, ~]            = unique_last_to_canon(T_bot);
uniqLast = intersect(uniqLastTop, uniqLastBot);

last2top = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(uniqLast)
    key = uniqLast{i};
    last2top(key) = last2canonTop(key);
end

% ——将 bills 的名字统一替换成 committees 的规范全名（优先全名，其次唯一姓）
Htop_name = Htop_name_full;
Hbot_name = cell(size(Hbot_name_raw));
for m=1:numel(Hbot_name_raw)
    e  = Hbot_name_raw{m};
    ee = {};
    for k=1:numel(e)
        nm = e{k};
        if isKey(directSet, nm)
            ee{end+1} = nm; %#ok<AGROW>
        else
            % 取该 bills 名字的姓（从 T_bot 表里查）
            [~,~,last] = name_lookup(nm, T_bot);
            if ~isempty(last) && isKey(last2top, last)
                ee{end+1} = last2top(last); %#ok<AGROW>
            end
        end
    end
    if numel(ee) >= 2, Hbot_name{m} = unique(sort(ee)); else, Hbot_name{m} = {}; end
end
Hbot_name = Hbot_name(~cellfun(@isempty, Hbot_name));

% 公共节点集合（统一为 committees 的规范名）
V = intersect(unique(cat(2,Htop_name{:})), unique(cat(2,Hbot_name{:})), 'stable');
end

function [uniqLast, last2canon] = unique_last_to_canon(T)
% 返回在 T 中只出现一次的姓集合，以及 姓->规范全名 的映射
lasts = lower(string(T.last));
[ul,~,idx] = unique(lasts);
cnt = accumarray(idx, 1);
mask = cnt==1;
uniqLast = cellstr(ul(mask));

last2canon = containers.Map('KeyType','char','ValueType','char');
for i=1:numel(uniqLast)
    key = uniqLast{i};
    row = find(lasts==string(key), 1, 'first');
    last2canon(key) = T.canon{row};
end
end

function [row,first,last] = name_lookup(canonName, T)
% 在表 T（具有 canon/first/last）中查找规范名对应的行；若找不到，尝试把只有姓的规范名视为 last
row = find(strcmpi(T.canon, canonName), 1, 'first');
if ~isempty(row)
    first = lower(string(T.first(row)));
    last  = lower(string(T.last(row)));
    return;
end
% 退化：若 canonName 只有一个词，就把它当作 last
parts = strsplit(lower(canonName), ' ');
parts = parts(~cellfun(@isempty,parts));
if numel(parts)==1
    last = string(parts{1}); first = "";
else
    first = ""; last = "";
end
end

function Hr = remap_names_to_ids(Hnames, name2newid)
Hr = cell(size(Hnames));
for m=1:numel(Hnames)
    e = Hnames{m};
    keep = false(numel(e),1);
    v = zeros(numel(e),1,'int32');
    for k=1:numel(e)
        nm = e{k};
        if isKey(name2newid, nm)
            keep(k)=true; v(k)=name2newid(nm);
        end
    end
    v = unique(v(keep));
    if numel(v)>=2, Hr{m} = sort(v(:)'); else, Hr{m} = []; end
end
Hr = Hr(~cellfun(@isempty, Hr));
end

function edgeMap = add_pairs(edgeMap, vec, wpair)
pairs = nchoosek(vec, 2);
for r=1:size(pairs,1)
    u = pairs(r,1); v = pairs(r,2);
    if u>v, tmp=u; u=v; v=tmp; end %#ok<NASGU>
    key = sprintf('%d-%d', u, v);
    if isKey(edgeMap,key), edgeMap(key)=edgeMap(key)+wpair;
    else, edgeMap(key)=wpair; end
end
end

function triMap = add_tris(triMap, vec, wtri)
tris = nchoosek(vec, 3);
for r=1:size(tris,1)
    a=tris(r,1); b=tris(r,2); c=tris(r,3);
    key = sprintf('%d-%d-%d', a,b,c);
    if isKey(triMap,key), triMap(key)=triMap(key)+wtri;
    else, triMap(key)=wtri; end
end
end

function w = pair_weight(k, mode)
switch lower(mode)
    case 'count',      w = 1.0;
    case 'invdeg',     w = 1.0/(k-1);
    case 'invchoose2', w = 1.0/nchoosek(k,2);
    otherwise, error('PAIR_WEIGHT_MODE 只能是 {count, invdeg, invchoose2}');
end
end

function w = tri_weight(k, mode)
switch lower(mode)
    case 'count',       w = 1.0;
    case 'invchoose3',  w = 1.0/nchoosek(k,3);
    otherwise, error('TRI_WEIGHT_MODE 只能是 {count, invchoose3}');
end
end

function [U,V,W] = map_to_arrays2(edgeMap)
keys2 = keys(edgeMap); vals2 = values(edgeMap);
n = numel(keys2); U=zeros(n,1); V=zeros(n,1); W=zeros(n,1);
for i=1:n
    uv = sscanf(keys2{i}, '%d-%d');
    U(i)=uv(1); V(i)=uv(2); W(i)=vals2{i};
end
end

function [I,J,K,W] = map_to_arrays3(triMap)
keys3 = keys(triMap); vals3 = values(triMap);
n = numel(keys3); I=zeros(n,1); J=zeros(n,1); K=zeros(n,1); W=zeros(n,1);
for i=1:n
    ijk = sscanf(keys3{i}, '%d-%d-%d');
    I(i)=ijk(1); J(i)=ijk(2); K(i)=ijk(3); W(i)=vals3{i};
end
end
