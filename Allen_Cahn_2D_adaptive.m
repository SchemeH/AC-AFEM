function Allen_Cahn_2D_adaptive(Tol_Space)
% 全局参数设置
global epsilon;
epsilon = 0.01;

% 域和时间参数设置
domain = [-0.5, 0.5, -0.5, 0.5]; % [xb, xe, yb, ye]
T0 = 0; Te = 1; 
NK = 10; % 时间步数
dt = (Te - T0) / NK;
tstop = T0:dt:Te;
Tol_Space = 3*1e-1;

% 初始网格设置
N = 16; % 初始网格尺寸
h = min(diff(domain(1:2))/N, diff(domain(3:4))/N); % 统一网格尺寸
maxdof = 256*256; % 最大自由度限制

theta = 0.7;    % 细化阈值
ctheta = 0.4;   % 粗化阈值

% 打印参数信息
fprintf('\n--- Parameters ---\n');
fprintf('Space tolerance: %e\n', Tol_Space);
fprintf('Max DOF: %d\n', maxdof);
fprintf('Refine/Coarsen thresholds: %.2f/%.2f\n', theta, ctheta);
fprintf('Epsilon: %.4f\n', epsilon);
fprintf('Initial h: %.4f, dt: %.4f\n', h, dt);
fprintf('Time steps: %d\n', NK);

% ====================== 初始化网格和解 ======================
[node, elem] = squaremesh(domain, h);
bdFlag = setboundary(node, elem, 'Neumann');
pde = AC2d; % 初始化PDE问题

% 初始网格自适应
fprintf('\nInitial mesh adaptation...\n');
n_refine = 0;
while true
    u = pde.u0(node);
    eta = estimaterecovery(node, elem, u);
    TotalError = relativeerror(node, elem, sqrt(sum(eta.^2)), u);
    Dof = size(node, 1);
    
    fprintf('Refine step %d: DOF=%d, Error=%.4e\n', n_refine, Dof, TotalError);
    
    % 终止条件：达到容差或超过最大自由度
    if TotalError < Tol_Space || Dof > maxdof
        break;
    end
    
    % 标记并细化单元
    markedElem = mark(elem, eta, theta, 'MAX'); 
    if isempty(markedElem)
        fprintf('No elements to refine. Stopping initial refinement.\n');
        break;
    end
    
    [node, elem, bdFlag, HB] = bisect(node, elem, markedElem, bdFlag);
    n_refine = n_refine + 1;
end

figure(1);
showmesh(node, elem);
title('Initial Mesh (t=%.1f)');
xlabel(sprintf('DOF: %d', size(node, 1)));

% ====================== 时间步进准备 ======================
u = pde.u0(node); % 初始解
% u = 1.8*rand(size(node,1),1)-0.9;
fprintf('\nStarting time integration...\n');
fprintf('Initial DOF: %d\n', size(node, 1));

% 预分配记录数组
dofs = zeros(1, NK+1); 
uinf = zeros(1, NK+1);
total_mass = zeros(1, NK+1);
egy = zeros(1, NK+1);
cut_u = zeros(1, NK+1);

% 记录初始值
dofs(1) = size(node, 1);
uinf(1) = max(abs(u)); 
[~, total_mass(1)] = compute_element_mass_fem(node, elem, u);
egy(1) = get_energy(node, elem, u, epsilon^2);
cut_u(1) = 0;
% ====================== 主时间循环 ======================
t_start = tic;
for i = 1:NK
    tn = i * dt;
    
    % --- 时间步进 ---
    u = KS_onestep_ETD(elem, node, dt, u);
    % u = max(min(un, 1), -1); % 确保解在[-1,1]范围内
    % cut_u(i+1) = max(max(abs(u-un)));
    % --- 空间自适应 ---
    adapt_params = struct(...
        'TolSpace', Tol_Space, ...
        'MaxDof', maxdof, ...
        'Theta', theta, ...
        'CTheta', ctheta);
    
    [node, elem, un, ~, bdFlag] = spaceAdaptation(...
        node, elem, u, bdFlag, [], adapt_params);
    u = min(1,max(-1,un));
    cut_u(i+1) = max(max(abs(u-un)));
    % --- 记录数据 ---
    dofs(i+1) = size(node, 1);
    uinf(i+1) = max(abs(u));
    [~, total_mass(i+1)] = compute_element_mass_fem(node, elem, u);
    egy(i+1) = get_energy(node, elem, u, epsilon^2);
    
    % --- 进度报告 ---
    if mod(i, 10) == 0 || i == NK
        fprintf('Step %d/%d: t=%.2f, DOF=%d, |u|_inf=%.4f\n',...
                i, NK, tn, dofs(i+1), uinf(i+1));
    end
end
comp_time = toc(t_start);

% ====================== 结果输出 ======================
fprintf('\n--- Results ---\n');
fprintf('Computation time: %.2f seconds\n', comp_time);
fprintf('Min/Max u: [%.4f, %.4f]\n', min(u), max(u));
fprintf('Energy: initial=%.4e, final=%.4e\n', egy(1), egy(end));
fprintf('Mass change: %.4e\n', total_mass(end) - total_mass(1));

% ====================== 可视化优化 ======================
% 只在最后时间步创建图形
% 1. 网格可视化
figure(2);
showmesh(node, elem);
title(sprintf('Final Mesh (t=%.1f)', tn));
xlabel(sprintf('DOF: %d', size(node, 1)));

% 2. 解的可视化
figure(3);
trisurf(elem, node(:,1), node(:,2), u);
shading interp;
colormap jet;
colorbar;
title(sprintf('Solution (t=%.1f)', tn));
xlabel('X'); ylabel('Y');
view(3);

% 3. 最大模随时间变化
figure(4);
plot(tstop, uinf, 'b-o', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Time'); ylabel('||u||_{\infty}');
grid on;


% % 4. 质量变化
% figure(5);
% plot(tstop, total_mass - total_mass(1), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 4);
% xlabel('Time'); ylabel('Mass Change');
% grid on;


% 5. 能量变化
figure(6);
plot(tstop, egy, 'g-d', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Time'); ylabel('Energy');
grid on;


% 6. 自由度变化
figure(7);
plot(tstop, dofs, 'm-^', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Time'); ylabel('DOF');
grid on;


figure(8);
plot(tstop, cut_u, 'm-^', 'LineWidth', 1.5, 'MarkerSize', 4);
xlabel('Time'); ylabel('Cut-off values');
grid on;
end


function [mass_per_element, total_mass] = compute_element_mass_fem(nodes, elements, u_values)
num_elements = size(elements, 1);
mass_per_element = zeros(num_elements, 1);

for e = 1:num_elements
    % 1. Obtain the node index and coordinates of the current unit.
    node_ids = elements(e, :);
    pts = nodes(node_ids, :);
    
    % 2. Calculate the area of the unit.
    vec1 = pts(2,:) - pts(1,:);
    vec2 = pts(3,:) - pts(1,:);
    area = 0.5 * abs(vec1(1)*vec2(2) - vec1(2)*vec2(1));
    
    % 3. Calculate the mass by integrating the shape function.
    u1 = u_values(node_ids(1));
    u2 = u_values(node_ids(2));
    u3 = u_values(node_ids(3));
    
    % integration: ∫u dΩ = ∑u_i ∫N_i dΩ = (u1 + u2 + u3) * (Area/3)
    mass_per_element(e) = (u1 + u2 + u3) * area / 3;
end

total_mass = sum(mass_per_element);
end

function egy = get_energy(node, elem, u, alpha)
egy = 0;
gaussPoints = [1/6, 1/6, 2/3, 1/3;   % first Guass point
               1/6, 2/3, 1/6, 1/3;   % second Guass point
               2/3, 1/6, 1/6, 1/3];  % third point

for i = 1:size(elem, 1)
    % 1. Obtain the node index and coordinates of the current unit.
    idx = elem(i, :);
    n1 = idx(1); n2 = idx(2); n3 = idx(3);
    p1 = node(n1, :);
    p2 = node(n2, :);
    p3 = node(n3, :);
    
    % 2. Calculate the area of the unit.
    v1 = p2 - p1;
    v2 = p3 - p1;
    area = 0.5 * abs(v1(1)*v2(2) - v1(2)*v2(1));
    
    % 3. Cauculate the gradient term
    A = [v1(1), v1(2);  % Jacobian matrix
         v2(1), v2(2)];
    b_v = [u(n2) - u(n1);  % The difference of v
          u(n3) - u(n1)];
    grad_v = A \ b_v;      % ∇v = [grad_x; grad_y]
    gradient_energy =  1/2*alpha * (grad_v(1)^2 + grad_v(2)^2)*area;
    
   % 4. Cauculate the nonlinear term
    nonlinear_energy = 0;
    for k = 1:size(gaussPoints, 1)
        lambda1 = gaussPoints(k, 1);
        lambda2 = gaussPoints(k, 2);
        lambda3 = gaussPoints(k, 3);
        weight = gaussPoints(k, 4);
        
        % Interpolation calculation of the values of u and v at the integration points using shape functions
        u_val = lambda1*u(n1) + lambda2*u(n2) + lambda3*u(n3);
        
        % Handle the case where u_val is non-positive (add a very small value epsilon to avoid log(0))
        epsilon = 1e-12;
        safe_u = max(u_val, epsilon);

        integrand = 1/4*(1-safe_u.^2).^2;
        nonlinear_energy = nonlinear_energy + integrand * weight * area;
    end
    egy = egy + gradient_energy + nonlinear_energy;
end
end

%% 空间自适应子程序
function [node, elem, u, HB, bdFlag] = spaceAdaptation(...
    node, elem, u, bdFlag, HB, varargin)
% 参数解析
p = inputParser;
addParameter(p, 'TolSpace', 1e-1);     % 空间误差容限
addParameter(p, 'MaxDof', 1e4);        % 最大自由度限制
addParameter(p, 'Theta', 0.7);         % 细化阈值
addParameter(p, 'CTheta', 0.3);        % 粗化阈值
addParameter(p, 'MaxCoarsenSteps', 50); % 最大粗化次数
parse(p, varargin{:});

% 初始化局部变量
coarsen_count = 0;
max_coarsen_steps = p.Results.MaxCoarsenSteps;

% 自适应主循环
while true
    % 误差估计
    eta = estimaterecovery(node, elem, u);
    TotalError = relativeerror(node, elem, sqrt(sum(eta.^2)), u);
    Dof = size(node, 1);
    
    % 输出当前状态
    fprintf('\nAdaptation Step: Dof = %d, Error = %.4e (Target = %.4e)', ...
            Dof, TotalError, p.Results.TolSpace);
    fprintf(', Coarsen count = %d/%d\n', coarsen_count, max_coarsen_steps);
    
    % 终止条件判断
    termination_condition = ...
        abs(TotalError - p.Results.TolSpace) < 1e-8 || ... % 达到目标误差
        Dof > p.Results.MaxDof || ...                      % 超过最大自由度
        coarsen_count >= max_coarsen_steps;                 % 达到最大粗化次数
    
    if termination_condition
        fprintf('Terminating: ');
        if abs(TotalError - p.Results.TolSpace) < 1e-8
            fprintf('Achieved target error tolerance.\n');
        elseif Dof > p.Results.MaxDof
            fprintf('Exceeded maximum degrees of freedom.\n');
        else
            fprintf('Reached maximum coarsening steps.\n');
        end
        break;
    end
    
    % 自适应决策
    if TotalError < p.Results.TolSpace
        % 持续粗化直到误差 ≥ TolSpace
        while TotalError < p.Results.TolSpace
            fprintf('Coarsening: Error < TolSpace (%.4e < %.4e)\n', ...
                    TotalError, p.Results.TolSpace);
            
            coarsenElem = mark(elem, eta, p.Results.CTheta, 'COARSEN');
            
            if isempty(coarsenElem)
                fprintf('No elements to coarsen. Breaking coarsening loop.\n');
                break;
            end
            
            [node, elem, bdFlag, indexMap] = coarsen(node, elem, coarsenElem, bdFlag);
            u = nodeinterpolate(u, indexMap);
            coarsen_count = coarsen_count + 1;
            
            % 更新误差估计
            eta = estimaterecovery(node, elem, u);
            TotalError = relativeerror(node, elem, sqrt(sum(eta.^2)), u);
            fprintf('  After coarsen: Dof = %d, Error = %.4e\n', size(node,1), TotalError);
            
            % 检查粗化终止条件
            if coarsen_count >= max_coarsen_steps
                fprintf('Reached maximum coarsening steps. Breaking coarsening loop.\n');
                break;
            end
        end
        
    else
        % 持续细化直到误差 ≤ TolSpace
        while TotalError > p.Results.TolSpace
            fprintf('Refining: Error > TolSpace (%.4e > %.4e)\n', ...
                    TotalError, p.Results.TolSpace);
            
            markedElem = mark(elem, eta, p.Results.Theta, 'MAX');
            
            if isempty(markedElem)
                fprintf('No elements to refine. Breaking refinement loop.\n');
                break;
            end
            
            [node, elem, bdFlag, HB] = bisect(node, elem, markedElem, bdFlag);
            u = nodeinterpolate(u, HB);
            
            % 更新误差估计
            eta = estimaterecovery(node, elem, u);
            TotalError = relativeerror(node, elem, sqrt(sum(eta.^2)), u);
            fprintf('  After refine: Dof = %d, Error = %.4e\n', size(node,1), TotalError);
            
            % 检查细化终止条件
            if size(node,1) > p.Results.MaxDof
                fprintf('Exceeded maximum degrees of freedom. Breaking refinement loop.\n');
                break;
            end
        end
    end
end

% 最终状态报告
fprintf('\nFinal Adaptation State:');
fprintf('\n  Degrees of Freedom: %d', size(node, 1));
fprintf('\n  Space Error: %.4e (Target: %.4e)', TotalError, p.Results.TolSpace);
fprintf('\n  Coarsening Steps: %d/%d\n', coarsen_count, max_coarsen_steps);
end

function [eta,Du] = estimaterecovery(node,elem,u)
%% ESTIMATERECOVERY recovery type error estimator.
%  
% eta = estimaterecovery(node,elem,u) computes an error estimator eta by
% recovery second derivative of the gradient of a finite element function u. 
%
% [eta, Du] = estimaterecovery(node,elem,u) also returns the recovered
% derivative Du which is in P1 finite element space.
%
% By interpolation error estimate $|u-uI|_{1,2}\leq C|u|_{2,1}$. Therefore
% we recovery an approximation of second derivatives of u and compute L1
% norm. We use the weighted averaging recovery scheme with area weight.
%
% See also recovery, estimaterecovery3, Lshape, crack
% 
% Copyright (C)  Long Chen. See COPYRIGHT.txt for details.

[Dlambda,area] = gradbasis(node,elem);
Du = gradu(node,elem,u,Dlambda);
Du = recovery(node,elem,Du,area); % 梯度重构
DDu(:,1:2) = gradu(node,elem,Du(:,1),Dlambda);
DDu(:,3:4) = gradu(node,elem,Du(:,2),Dlambda);
eta = area.*sum(abs(DDu),2);
%% TODO add diffusion coefficient
end

function [node,elem] = squaremesh(square,h)
%% SQUAREMESH uniform mesh of a square
%
% [node,elem] = squaremesh([x0,x1,y0,y1],h) generates a uniform mesh of the
% rectangle [x0,x1]*[y0,y1] with mesh size h.
%
% Example
%
%   [node,elem] = squaremesh([0,1,0,1],0.2);
%   showmesh(node,elem);
%   findnode(node);
%   findelem(node,elem);
%
% See also: squarequadmesh, cubehexmesh
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

%% Generate nodes
x0 = square(1); x1 = square(2); 
y0 = square(3); y1 = square(4);
[x,y] = meshgrid(x0:h:x1,y0:h:y1);
node = [x(:),y(:)];

%% Generate elements
ni = size(x,1); % number of rows
N = size(node,1);
t2nidxMap = 1:N-ni;
topNode = ni:ni:N-ni;
t2nidxMap(topNode) = [];
k = (t2nidxMap)';
elem = [k+ni k+ni+1 k; k+1 k k+ni+1];
% 4 k+1 --- k+ni+1 3  
%    |        |
% 1  k  ---  k+ni  2
end

function bdFlag = setboundary(node,elem,varargin)
%% SETBOUNDARY set type of boundary edges.
%
%  Type of bundary conditions:
%   - 1. Dirichlet
%   - 2. Neumann
%   - 3. Robin
%   - 3. ABC for wave type equation
%
%  bdFlag = SETBOUNDARY(node,elem,'Dirichlet') set all boundary edges to
%  Dirichlet type. 
%
%  bdFlag = SETBOUNDARY(node,elem,'Neumann') set all boundary edges to
%  Neumann type. 
%
%  bdFlag = SETBOUNDARY(node,elem,'Robin') set all boundary edges to
%  Robin type. 
%
%  bdFlag = SETBOUNDARY(node,elem,'Dirichlet','(x==1) | (x==-1)') set
%  Dirichlet boundary condition on x=1 and x=-1. Other edges are
%  homongenous Neumann boundary condition.
%
%  bdFlag = SETBOUNDARY(node,elem,'Dirichlet','(x==1) | ...
%  (x==-1)','Neumann','(y==1) | (y==-1)') set
%  Dirichlet boundary condition on x=1 or x=-1 and Neumann boundary
%  condition on y=1 or y=-1.
%
%  bdFlag = SETBOUNDARY(node,elem,'Dirichlet','(x==1) | ...
%  (x==-1)','Neumann','y==1', 'Robin',' y==-1') set
%  Dirichlet boundary condition on x=1 or x=-1 and Neumann boundary
%  condition on y=1, and Robin boundary condition on y=-1.
%
%  bdFlag = SETBOUNDARY(node,elem,'Dirichlet','all','Neumann','y==1') set
%  Neumann boundary condition on y=1 and others are Dirichlet boundary condition.
%
% Example
%   
%      node = [0,0; 1,0; 1,1; 0,1];
%      elem = [2,3,1; 4,1,3];
%      bdFlag = setboundary(node,elem,'Dirichlet','all','Neumann','y==1');
%      [node,elem,bdFlag] = uniformbisect(node,elem,bdFlag);
%      [node,elem,bdFlag] = uniformbisect(node,elem,bdFlag);
%      showmesh(node,elem);
%      allEdge = [elem(:,[2,3]); elem(:,[3,1]); elem(:,[1,2])];
%      Dirichlet = allEdge((bdFlag(:) == 1),:);
%      Neumann = allEdge((bdFlag(:) == 2) | (bdFlag(:) == 3),:);
%      findedge(node,Dirichlet,[],'noindex','LineWidth',4,'Color','r');
%      findedge(node,Neumann,[],'noindex','LineWidth',4,'Color','b');
%
% See also setboundary3
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

%% Find boundary edges
nv = size(elem,2);
if nv == 3 % triangles 
    allEdge = uint32(sort([elem(:,[2,3]); elem(:,[3,1]); elem(:,[1,2])],2));
elseif nv == 4 % quadrilateral
    allEdge = uint32(sort([elem(:,[1,2]); elem(:,[2,3]); elem(:,[3,4]); elem(:,[4 1])],2));    
end
ne = nv; % number of edges in one element
Neall = length(allEdge);
[edge, i2, j] = myunique(allEdge);
NT = size(elem,1);
i1(j(Neall:-1:1)) = Neall:-1:1; 
i1 = i1';
bdFlag = zeros(Neall,1,'uint8');
bdEdgeidx = i1(i1==i2);

%% Set up boundary edges
nVarargin = size(varargin,2);
if (nVarargin==1)
    bdType = findbdtype(varargin{1});
    bdFlag(bdEdgeidx) = bdType;
end
if (nVarargin>=2)
    for i=1:nVarargin/2
        bdType = findbdtype(varargin{2*i-1});
        expr = varargin{2*i};
        if strcmp(expr,'all')
            bdFlag(bdEdgeidx) = bdType;
        else
           x = (node(allEdge(bdEdgeidx,1),1) + node(allEdge(bdEdgeidx,2),1))/2; %#ok<NASGU>
           y = (node(allEdge(bdEdgeidx,1),2) + node(allEdge(bdEdgeidx,2),2))/2; %#ok<NASGU>
           idx = eval(expr);
           bdFlag(bdEdgeidx(idx)) = bdType;
        end
    end
end
bdFlag = reshape(bdFlag,NT,ne);
end
%%
function bdType = findbdtype(bdstr)
    switch bdstr
        case 'Dirichlet'
            bdType = 1;
        case 'Neumann'
            bdType = 2;
        case 'Robin'
            bdType = 3;
        case 'ABC' % absorbing boundary condition for wave-type equations
            bdType = 3;
    end
end

function pde = AC2d
pde = struct('u0', @u0);

    function s = u0(p)
        x = p(:, 1);
        y = p(:, 2);
        s = cos(2*pi*x).*cos(2*pi*y);
        % % 创建逻辑索引：判断点是否在圆盘内 (x² + y² ≤ 0.04)
        % inside_disk = (x.^2 + y.^2) <= 0.04;  % 0.2² = 0.04
        % 
        % % 初始化结果数组（全为-1）
        % s = -ones(size(x));
        % 
        % % 将圆盘内的值设为1
        % s(inside_disk) = 1;
    end
end

function [Dlambda,area,elemSign] = gradbasis(node,elem)
%% GRADBASIS gradient of barycentric basis. 
%
% [Dlambda,area,elemSign] = GRADBASIS(node,elem) compute gradient of
% barycentric basis and areas of triangles. The array area is NT by 1 and
% area(t) is the volume of the t-th tetrahedron. Dlambda(1:NT, 1:3, 1:3)
% the first index is the label of tetrahedron, the second is the x-y
% coordinate, and the last one is the label of three indices of a triangle.
% For example, Dlambda(t,:,1) is the gradient of lambda of the 1st index of
% the t-th triangle. The elemSign array taking values 1 or -1 records the
% sign of the signed area.
%
% See also gradbasis3, gradu, gradu3
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

NT = size(elem,1);
% $\nabla \phi_i = rotation(l_i)/(2|\tau|)$
ve1 = node(elem(:,3),:)-node(elem(:,2),:);
ve2 = node(elem(:,1),:)-node(elem(:,3),:);
ve3 = node(elem(:,2),:)-node(elem(:,1),:);
area = 0.5*(-ve3(:,1).*ve2(:,2) + ve3(:,2).*ve2(:,1));
Dlambda(1:NT,:,3) = [-ve3(:,2)./(2*area), ve3(:,1)./(2*area)];
Dlambda(1:NT,:,1) = [-ve1(:,2)./(2*area), ve1(:,1)./(2*area)];
Dlambda(1:NT,:,2) = [-ve2(:,2)./(2*area), ve2(:,1)./(2*area)];

% When the triangle is not positive orientated, we reverse the sign of the
% area. The sign of Dlambda is always right since signed area is used in
% the computation.
idx = (area<0); 
area(idx,:) = -area(idx,:);
elemSign = ones(NT,1);
elemSign(idx) = -1;
end

function [Du,area,Dlambda] = gradu(node,elem,u,Dlambda)
%% GRADU gradient of a finite element function.
%
% Du = GRADU(node,elem,u) compute the gradient of a finite element function
% u on a mesh representing by (node,elem).
% 
% [Du,area,Dlambda] = GRADU(node,elem,u) also outputs area and Dlambda
% which is the gradient of P1 basis. 
%
% Du = GRADU(node,elem,u,Dlambda) compute the gradient with Dlambda. It
% will save time when Dlambda is available. See recovery.m
%
% See also gradu3, gradbasis
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

if ~exist('Dlambda','var')
    [Dlambda,area] = gradbasis(node,elem);
end
dudx =  u(elem(:,1)).*Dlambda(:,1,1) + u(elem(:,2)).*Dlambda(:,1,2) ...
      + u(elem(:,3)).*Dlambda(:,1,3);
dudy =  u(elem(:,1)).*Dlambda(:,2,1) + u(elem(:,2)).*Dlambda(:,2,2) ...
      + u(elem(:,3)).*Dlambda(:,2,3);         
Du = [dudx, dudy];
end

function RDu = recovery(node,elem,Du,area)
%% RECOVERY recovery a piecewise constant function to a piecewise linear one.
%
% RDu = recovery(node,elem,Du,area) compute a P1 approximation u using area
% weighted average of the piecewise constant gradient Du usually given by
% [Du,area] = gradu(node,elem,u).
%
% See also gradu, estimaterecovery, recovery3
%
% Copyright (C)  Long Chen. See COPYRIGHT.txt for details.

N = size(node,1);
dudxArea = area.*Du(:,1);
dudyArea = area.*Du(:,2);
patchArea = accumarray(elem(:),[area;area;area], [N 1]); 
dudxArea = accumarray(elem(:),[dudxArea;dudxArea;dudxArea],[N 1]);
dudyArea = accumarray(elem(:),[dudyArea;dudyArea;dudyArea],[N 1]);
dudx = dudxArea./patchArea;
dudy = dudyArea./patchArea;
RDu = [dudx, dudy];
end

function markedElem = mark(elem,eta,theta,method)
% MARK mark element.
%
% markedElem = mark(elem,eta,theta) mark a subset of elements by Dorfler
% marking strategy. It returns an array of indices of marked elements
% markedElem such that sum(eta(markedElem)^2) > theta*sum(eta^2).
%
% markedElem = mark(elem,eta,theta,'max') choose markedElem such that
% eta(markedElem) > theta*max(eta).
%
% markedElem = mark(elem,eta,theta,'COARSEN') choose markedElem such that
% eta(markedElem) < theta*max(eta).
%
% Copyright (C) 2008 Long Chen. See COPYRIGHT.txt for details.

NT = size(elem,1); isMark = false(NT,1);
if ~exist('method','var'), method = 'L2'; end  % default marking is L2 based
switch upper(method)
    case 'MAX'
        isMark(eta>theta*max(eta))=1;
    case 'COARSEN'
        isMark(eta<theta*max(eta))=1;
    case 'L2'
        [sortedEta,idx] = sort(eta.^2,'descend'); 
        x = cumsum(sortedEta);
        isMark(idx(x < theta* x(NT))) = 1;
        isMark(idx(1)) = 1;
end
markedElem = uint32(find(isMark==true));
end

function [node,elem,bdFlag,HB,tree] = bisect(node,elem,markedElem,bdFlag)
%% BISECT bisect a 2-D triangulation.
% 
% [node,elem] = BISECT(node,elem,markedElem) refine the current
% triangulation by bisecting marked elements and minimal neighboring
% elements to get a conforming and shape regular triangulation. Newest
% vertex bisection is implemented. markedElem is a vector containing the
% indices of elements to be bisected. It could be a logical vector of
% length size(elem,1). 
% 
% [node,elem,bdFlag] = BISECT(node,elem,markedElem,bdFlag) returns the
% updated bdFlag after the bisection. It will be used for PDEs with mixed
% boundary conditions.
% 
% [node,elem,bdFlag,HB,tree] = BISECT(node,elem,markedElem,bdFlag)
% returns HB and tree arrays.
% 
% - HB(:,1:3) is a hierarchical basis structure for nodes, where
%   HB(:,1) is the global index of new added nodes, and HB(:,2:3) the 
%   global indices of two parent nodes of new added nodes. HB is usful
%   for the interpolation between two grids; see also nodeinterpolate.
% 
% - tree(:,1:3) stores the binary tree of the coarsening. tree(:,1) is the
%   index of parent element in coarsened mesh and tree(:,2:3) are two
%   children indices in original mesh. tree is useful for the interpolation
%   of elementwise function; see also eleminterpolate.
%
% Example
%
%      node = [0,0; 1,0; 1,1; 0,1];
%      elem = [2,3,1; 4,1,3];
%      figure(1); subplot(1,3,1); showmesh(node,elem);
%      [node,elem] = bisect(node,elem,'all');
%      figure(1); subplot(1,3,2); showmesh(node,elem);
%      bdFlag = setboundary(node,elem,'Dirichlet','all','Neumann','y==1');
%      [node,elem,bdFlag] = bisect(node,elem,[1 4],bdFlag);
%      figure(1); subplot(1,3,3); showmesh(node,elem);
%
% See also bisect3, coarsen, coarsen3, nodeinterpolate, eleminterpolate.
%
% Reference page in Help browser
% <a href="matlab:ifem meshdoc">ifem meshdoc</a>
% <a href="matlab:ifem bisectdoc">ifem bisectdoc</a> 

% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

%% Set up
HB = []; tree = []; 
if ~exist('bdFlag','var'), bdFlag = []; end
if ~exist('markedElem','var'), markedElem = (1:size(elem,1))'; end
if isempty(markedElem), return; end
if strcmp(markedElem,'all'), markedElem = (1:size(elem,1))'; end
if islogical(markedElem), markedElem = find(markedElem); end

%% Construct auxiliary data structure
T = auxstructure(elem);
neighbor = T.neighbor; elem2edge = T.elem2edge; edge = T.edge;
clear T;
%[neighbor,elem2edge,edge] = auxstructurec(int32(elem));
N = size(node,1); NT = size(elem,1); NE = size(edge,1);

%% Add new nodes
isCutEdge = false(NE,1);
while sum(markedElem)>0
    isCutEdge(elem2edge(markedElem,1)) = true;
    refineNeighbor = neighbor(markedElem,1);
    markedElem = refineNeighbor(~isCutEdge(elem2edge(refineNeighbor,1)));
end
edge2newNode = zeros(NE,1,'uint32');
edge2newNode(isCutEdge) = N+1:N+sum(isCutEdge);
HB = zeros(sum(isCutEdge),3,'uint32');
HB(:,1) = edge2newNode(isCutEdge);
HB(:,[2 3]) = edge(isCutEdge,[1 2]);
node(HB(:,1),:) = (node(HB(:,2),:) + node(HB(:,3),:))/2;

%% Refine marked elements
Nb = 0; tree = zeros(3*NT,3,'uint32');
for k = 1:2
    t = find(edge2newNode(elem2edge(:,1))>0);
    newNT = length(t);
    if (newNT == 0), break; end
    L = t; R = NT+1:NT+newNT;
    p1 = elem(t,1); p2 = elem(t,2); p3 = elem(t,3);
    p4 = edge2newNode(elem2edge(t,1));
    elem(L,:) = [p4, p1, p2];
    elem(R,:) = [p4, p3, p1];
	if nargin==4 && ~isempty(bdFlag) % Refine boundary edges
   		bdFlag(R,[1 3]) = bdFlag(t,[2 1]);
   		bdFlag(L,[1 2]) = bdFlag(t,[3 1]);
        bdFlag(L,3) = 0;
    else
        bdFlag = [];
	end
    tree(Nb+1:Nb+newNT,1) = L;
    tree(Nb+1:Nb+newNT,2) = L;
    tree(Nb+1:Nb+newNT,3) = R;
    elem2edge(L,1) = elem2edge(t,3);
    elem2edge(R,1) = elem2edge(t,2);
    NT = NT + newNT; Nb = Nb + newNT;
end
tree = tree(1:Nb,:);
end

function T = auxstructure(elem)
%% AUXSTRUCTURE auxiliary structure for a 2-D triangulation.
%
%  T = AUXSTRUCTURE(elem) constucts the indices map between elements, edges 
%  and nodes, and the boundary information. T is a structure. 
%
%  T.neighbor(1:NT,1:3): the indices map of neighbor information of elements, 
%  where neighbor(t,i) is the global index of the element oppoiste to the 
%  i-th vertex of the t-th element. 
%
%  T.elem2edge(1:NT,1:3): the indices map from elements to edges, elem2edge(t,i) 
%  is the edge opposite to the i-th vertex of the t-th element.
%
%  T.edge(1:NE,1:2): all edges, where edge(e,i) is the global index of the 
%  i-th vertex of the e-th edge, and edge(e,1) < edge(e,2) 
%
%  T.bdEdge(1:Nbd,1:2): boundary edges with positive oritentation, where
%  bdEdge(e,i) is the global index of the i-th vertex of the e-th edge for
%  i=1,2. The positive oritentation means that the interior of the domain
%  is on the left moving from bdEdge(e,1) to bdEdge(e,2). Note that this
%  requires elem is positive ordered, i.e., the signed area of each
%  triangle is positive. If not, use elem = fixorder(node,elem) to fix the
%  order.
%
%  T.edge2elem(1:NE,1:4): the indices map from edge to element, where 
%  edge2elem(e,1:2) are the global indexes of two elements sharing the e-th
%  edge, and edge2elem(e,3:4) are the local indices of e to edge2elem(e,1:2).
%
%  To save space all the data type in T is uint32. When use them as a input
%  of sparse(i,j,s,m,n), please change them into double type.
% 
%  See also auxstructure3.
% 
% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

totalEdge = uint32(sort([elem(:,[2,3]); elem(:,[3,1]); elem(:,[1,2])],2));
[edge,i2,j] = myunique(totalEdge);
NT = size(elem,1);
elem2edge = uint32(reshape(j,NT,3));
i1(j(3*NT:-1:1)) = 3*NT:-1:1; 
i1 = i1';
k1 = ceil(i1/NT); 
k2 = ceil(i2/NT); 
t1 = i1 - NT*(k1-1);
t2 = i2 - NT*(k2-1);
ix = (i1 ~= i2); 
neighbor = uint32(accumarray([[t1(ix),k1(ix)];[t2,k2]],[t2(ix);t1],[NT 3]));
edge2elem = uint32([t1,t2,k1,k2]);
bdElem = t1(t1 == t2);
bdk1 = k1(t1 == t2);
bdEdge = [elem(bdElem(bdk1==1),[2 3]); elem(bdElem(bdk1==2),[3 1]);...
          elem(bdElem(bdk1==3),[1 2])];
bdEdge2elem = [bdElem(bdk1==1);bdElem(bdk1==2);bdElem(bdk1==3)];
T = struct('neighbor',neighbor,'elem2edge',elem2edge,'edge',edge,...
           'edge2elem',edge2elem,'bdElem',bdElem,'bdEdge',bdEdge,...
           'bdEdge2elem', bdEdge2elem);
end

function u = KS_onestep_ETD(elem, node, dt, uold)
        epsilon = 0.01; poten = 1; 
        if poten == 1 
            kappa = 2;
        else
            kappa = 8.02;
        end
        
        [A,M,~] = assemblematrix(node,elem,1); %质量集中
        N = size(node, 1);
        M =  spdiags(M,0,N,N);
        %% 组装左端矩阵
        K1 = assemblematrix2D(node, elem, mobility(uold));
         
        AA1 = -M\(epsilon^2*(A-(K1)))-kappa*speye(size(A));

        FFu = (1-uold.^2).*(uold - uold.^3) + kappa*uold;

         % ETD1 求解u
         u = phipm(dt,AA1,[uold,FFu]);
end

function [A,M,area] = assemblematrix(node,elem,lumpflag,K)
%% ASSEMBLEMATRIX matrix for diffusion and reaction
%
% [A,M] = ASSEMBLEMATRIX(node,elem) return the stiffness matrix and the
% mass matrix.
%
% [A,M] = ASSEMBLEMATRIX(node,elem,1) return the stiffness matrix and the
% lumped mass matrix. Note that in the output M is a vector not a matrix. A
% sparse diagonal matrix using M as diaongal can be obtained by
% spdiags(M,0,N,N);
%
% A = ASSEMBLEMATRIX(node,elem,[],K) returns stiffness matrix with
% piecewise constant coefficient K.
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

%% Parameters
N = size(node,1);
A = sparse(N,N);
M = sparse(N,N);
if ~exist('K','var'), K = []; end
if ~exist('lumpflag','var'), lumpflag = 0; end
if (nargout > 1)
    if ~exist('lumpflag','var') || isempty(lumpflag)
        lumpflag = 0;
    end    
end

%% 3-D case
if (size(node,2) == 3) && (size(elem,2) == 4) % 3-D 
    [A,M,area] = assemblematrix3(node,elem,lumpflag);
    return
end

%% Compute vedge, edge as a vector, and area of each element
ve(:,:,1) = node(elem(:,3),:)-node(elem(:,2),:);
ve(:,:,2) = node(elem(:,1),:)-node(elem(:,3),:);
ve(:,:,3) = node(elem(:,2),:)-node(elem(:,1),:);
area = 0.5*abs(-ve(:,1,3).*ve(:,2,2)+ve(:,2,3).*ve(:,1,2));

%% Assemble stiffness matrix
for i = 1:3
    for j = 1:3
        Aij = (ve(:,1,i).*ve(:,1,j)+ve(:,2,i).*ve(:,2,j))./(4*area);
        if ~isempty(K), Aij = K.*Aij; end
        A = A + sparse(elem(:,i),elem(:,j),Aij,N,N);
        if ~lumpflag 
           Mij = area*((i==j)+1)/12;
           M = M + sparse(elem(:,i),elem(:,j),Mij,N,N);
        end
    end
end

%% Assemble the mass matrix by the mass lumping
if lumpflag
    M = accumarray([elem(:,1);elem(:,2);elem(:,3)],[area;area;area]/3,[N,1]);
end
end

function K = assemblematrix2D(node, elem, vh)
%% vh是定义在二维网格上的数值解，(N,1)数组，N为节点数
N = size(node, 1);
NT = size(elem, 1);

% 计算基函数梯度和面积（二维用面积代替体积）
[Dphi, area] = gradbasis(node,elem); % 改为二维梯度计算函数（NT,2,3）
dudx = vh(elem(:,1)).*Dphi(:,1,1) + vh(elem(:,2)).*Dphi(:,1,2) + vh(elem(:,3)).*Dphi(:,1,3);
dudy = vh(elem(:,1)).*Dphi(:,2,1) + vh(elem(:,2)).*Dphi(:,2,2) + vh(elem(:,3)).*Dphi(:,2,3); 
solnDu = [dudx, dudy]; % 改为二维梯度（NT,2）

% 获取二维积分点和权重（例如4阶积分）
[lambda, weight] = quadpts(4); % 改为二维积分函数
t1 = sum( (weight.') .* lambda, 1); % (1,3) 每个基函数的积分权重
basis_integrals = area.*t1; % (NT,3) 每个单元基函数的积分

%% 组装K矩阵：K(i,j) = ∫φ_j(∇vh·∇φ_i)dx
K = sparse(N, N);
for i = 1:3 % 三角形三个顶点
    for j = 1:3
        ii = double(elem(:,i)); % 当前单元第i个节点的全局索引
        jj = double(elem(:,j));
        % 二维梯度点乘：∇vh·∇φ_i = dudx*Dphi_x + dudy*Dphi_y
        Kij = (solnDu(:,1).*Dphi(:,1,i) + solnDu(:,2).*Dphi(:,2,i)) .* basis_integrals(:,j);
        K = K + sparse(ii, jj, Kij, N, N);
    end
end
end


function [lambda,weight] = quadpts(order)
%% QUADPTS quadrature points in 2-D.
%
% [lambda,weight] = quadpts(order) return quadrature points with given
% order (up to 9) in the barycentric coordinates.
%
% The output lambda is a matrix of size nQ by 3, where nQ is the number of
% quadrature points. lambda(i,:) is three barycentric coordinate of the
% i-th quadrature point and lambda(:,j) is the j-th barycentric coordinate
% of all quadrature points. The x-y coordinate of the p-th quadrature point
% can be computed as 
%
%     pxy = lambda(p,1)*node(elem(:,1),:) ...
%         + lambda(p,2)*node(elem(:,2),:) ... 
%         + lambda(p,3)*node(elem(:,3),:);
%
% The weight of p-th quadrature point is given by weight(p). See
% verifyquadpts for the usage of qudrature rules to compute integrals over
% triangles.
% 
% References: 
%
% * David Dunavant. High degree efficient symmetrical Gaussian
%    quadrature rules for the triangle. International journal for numerical
%    methods in engineering. 21(6):1129--1148, 1985. 
% * John Burkardt. DUNAVANT Quadrature Rules for the Triangle.
%    http://people.sc.fsu.edu/~burkardt/m_src/dunavant/dunavant.html
% 
% See also quadpts1, quadpts3, verifyquadpts
%
% Order 6 - 9 is added by Huayi Wei, modify by Jie Zhou
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details. 

if order>9
    order = 9;
end
switch order
    case 1     % Order 1, nQuad 1
        lambda = [1/3, 1/3, 1/3];
        weight = 1;
    case 2     % Order 2, nQuad 3
        lambda = [2/3, 1/6, 1/6; ...
                  1/6, 2/3, 1/6; ...
                  1/6, 1/6, 2/3];
        weight = [1/3, 1/3, 1/3];
    case 3     % Order 3, nQuad 4
        lambda = [1/3, 1/3, 1/3; ...
                  0.6, 0.2, 0.2; ...
                  0.2, 0.6, 0.2; ...
                  0.2, 0.2, 0.6];
        weight = [-27/48, 25/48, 25/48, 25/48];
    case 4     % Order 4, nQuad 6
        lambda = [0.108103018168070, 0.445948490915965, 0.445948490915965; ...
                  0.445948490915965, 0.108103018168070, 0.445948490915965; ...
                  0.445948490915965, 0.445948490915965, 0.108103018168070; ...
                  0.816847572980459, 0.091576213509771, 0.091576213509771; ...
                  0.091576213509771, 0.816847572980459, 0.091576213509771; ...
                  0.091576213509771, 0.091576213509771, 0.816847572980459];
        weight = [0.223381589678011, 0.223381589678011, 0.223381589678011, ...
                  0.109951743655322, 0.109951743655322, 0.109951743655322];
    case 5     % Order 5, nQuad 7
        alpha1 = 0.059715871789770;      beta1 = 0.470142064105115;
        alpha2 = 0.797426985353087;      beta2 = 0.101286507323456;
        lambda = [   1/3,    1/3,    1/3; ...
                  alpha1,  beta1,  beta1; ...
                   beta1, alpha1,  beta1; ...
                   beta1,  beta1, alpha1; ...
                  alpha2,  beta2,  beta2; ...
                   beta2, alpha2,  beta2; ...
                   beta2,  beta2, alpha2];
        weight = [0.225, 0.132394152788506, 0.132394152788506, 0.132394152788506, ...
            0.125939180544827, 0.125939180544827, 0.125939180544827];
    case 6        
        A =[0.249286745170910  0.249286745170910  0.116786275726379
            0.249286745170910  0.501426509658179  0.116786275726379
            0.501426509658179  0.249286745170910  0.116786275726379
            0.063089014491502  0.063089014491502  0.050844906370207
            0.063089014491502  0.873821971016996  0.050844906370207
            0.873821971016996  0.063089014491502  0.050844906370207
            0.310352451033784  0.636502499121399  0.082851075618374
            0.636502499121399  0.053145049844817  0.082851075618374
            0.053145049844817  0.310352451033784  0.082851075618374
            0.636502499121399  0.310352451033784  0.082851075618374
            0.310352451033784  0.053145049844817  0.082851075618374
            0.053145049844817  0.636502499121399  0.082851075618374];
        lambda = [A(:,[1,2]), 1 - sum(A(:,[1,2]),2)];
        weight = A(:,3);
    case 7
        A =[0.333333333333333  0.333333333333333 -0.149570044467682
            0.260345966079040  0.260345966079040  0.175615257433208
            0.260345966079040  0.479308067841920  0.175615257433208
            0.479308067841920  0.260345966079040  0.175615257433208
            0.065130102902216  0.065130102902216  0.053347235608838
            0.065130102902216  0.869739794195568  0.053347235608838
            0.869739794195568  0.065130102902216  0.053347235608838
            0.312865496004874  0.638444188569810  0.077113760890257
            0.638444188569810  0.048690315425316  0.077113760890257
            0.048690315425316  0.312865496004874  0.077113760890257
            0.638444188569810  0.312865496004874  0.077113760890257
            0.312865496004874  0.048690315425316  0.077113760890257
            0.048690315425316  0.638444188569810  0.077113760890257];
        lambda = [A(:,[1,2]), 1 - sum(A(:,[1,2]),2)];
        weight = A(:,3);
    case 8
        A =[0.333333333333333  0.333333333333333  0.144315607677787
            0.081414823414554  0.459292588292723  0.095091634267285
            0.459292588292723  0.081414823414554  0.095091634267285
            0.459292588292723  0.459292588292723  0.095091634267285
            0.658861384496480  0.170569307751760  0.103217370534718
            0.170569307751760  0.658861384496480  0.103217370534718
            0.170569307751760  0.170569307751760  0.103217370534718
            0.898905543365938  0.050547228317031  0.032458497623198
            0.050547228317031  0.898905543365938  0.032458497623198
            0.050547228317031  0.050547228317031  0.032458497623198
            0.008394777409958  0.263112829634638  0.027230314174435
            0.008394777409958  0.728492392955404  0.027230314174435
            0.263112829634638  0.008394777409958  0.027230314174435
            0.728492392955404  0.008394777409958  0.027230314174435
            0.263112829634638  0.728492392955404  0.027230314174435
            0.728492392955404  0.263112829634638  0.027230314174435];
        lambda = [A(:,[1,2]), 1 - sum(A(:,[1,2]),2)];
        weight = A(:,3);
        
        case 9
        A =[0.333333333333333  0.333333333333333  0.097135796282799
            0.020634961602525  0.489682519198738  0.031334700227139
            0.489682519198738  0.020634961602525  0.031334700227139
            0.489682519198738  0.489682519198738  0.031334700227139
            0.125820817014127  0.437089591492937  0.07782754100474
            0.437089591492937  0.125820817014127  0.07782754100474
            0.437089591492937  0.437089591492937  0.07782754100474
            0.623592928761935  0.188203535619033  0.079647738927210
            0.188203535619033  0.623592928761935  0.079647738927210
            0.188203535619033  0.188203535619033  0.079647738927210
            0.910540973211095  0.044729513394453  0.025577675658698
            0.044729513394453  0.910540973211095  0.025577675658698
            0.044729513394453  0.044729513394453  0.025577675658698
            0.036838412054736  0.221962989160766  0.043283539377289
            0.036838412054736  0.741198598784498  0.043283539377289
            0.221962989160766  0.036838412054736  0.043283539377289
            0.741198598784498  0.036838412054736  0.043283539377289
            0.221962989160766  0.741198598784498  0.043283539377289
            0.741198598784498  0.221962989160766  0.043283539377289];
        lambda = [A(:,[1,2]), 1 - sum(A(:,[1,2]),2)];
        weight = A(:,3);
end
%% Verification
% The order of the quadrature rule is verified by the function
% verifyquadpts. See <matlab:ifem('verifyquadpts') verifyquadpts>.
end

function [sortA,i2,j] = myunique(A)
%% MYUNIQUE the same input and output as unique 
%
% Solve the change of unique command in different version of MATLAB
%
% See also: unique
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

matlabversion = version;
if str2double(matlabversion(end-5:end-2)) <= 2012
    [sortA, i2, j] = unique(A,'rows');
else
    [sortA, i2, j] = unique(A,'rows','legacy'); %#ok<*ASGLU>
end
end

function Rerror = relativeerror(node, elem, error, vh)
% 计算二维相对误差
    [Dphi, area] = gradbasis(node, elem);  % 基函数梯度 Dphi (NT, 2, 3)
    
    % 计算数值解的梯度场
    dudx = vh(elem(:,1)).*Dphi(:,1,1) + vh(elem(:,2)).*Dphi(:,1,2)...
          + vh(elem(:,3)).*Dphi(:,1,3);
    dudy = vh(elem(:,1)).*Dphi(:,2,1) + vh(elem(:,2)).*Dphi(:,2,2)...
          + vh(elem(:,3)).*Dphi(:,2,3);
    
    solnDu = [dudx, dudy];       % 数值解梯度矩阵 (NT, 2)
    Du_h1 = sum(solnDu.^2, 2);   % 每个单元梯度模方
    Du_h1 = sum(Du_h1.*area);    % 全域H1半模
    Du_h1 = sqrt(Du_h1);         % H1范数
    
    Rerror = error/Du_h1;        % 相对误差
end

function h = showmesh(node,elem,varargin)
%% SHOWMESH displays a triangular mesh in 2-D.
%
%    [修改说明]
%    面颜色改为白色，边颜色改为淡蓝色([0.7, 0.8, 1])
%
%    showmesh(node,elem) displays a topological 2-dimensional mesh,
%    including planar meshes and surface meshes. The mesh is given by node
%    and elem matrices; see <a href="matlab:ifem('meshdoc')">meshdoc</a> for the mesh data structure: 
%    node and elem.
%
%    showmesh(node,elem,viewangle) changes the display angle. The
%    deault view angle on planar meshes is view(2) and view(3) for surface
%    meshes. 
%        
%    showmesh(node,elem,'param','value','param','value'...) allows
%    additional patch param/value pairs to be used when displaying the
%    mesh.  For example, the default transparency parameter for a surface
%    mesh is set to 0.75. You can overwrite this value by using the param
%    pair ('FaceAlpha', value). The value has to be a number between 0 and
%    1. Other parameters include: 'Facecolor', 'Edgecolor' etc. These
%    parameters are mostly used when displaying a surface mesh.
%
%    To display a 3-dimensional mesh, use showmesh3 or showboundary3.
% 
%   Example:
    % A mesh for a L-shaped domain
    % [node,elem] = squaremesh([-1,1,-1,1],0.5);
    % %[node,elem] = delmesh(node,elem,'x>0 & y<0');
    % figure;
    % showmesh(node,elem);
    % 
    % % A mesh for a unit sphere
    % node = [1,0,0; 0,1,0; -1,0,0; 0,-1,0; 0,0,1; 0,0,-1];
    % elem = [6,1,2; 6,2,3; 6,3,4; 6,4,1; 5,1,4; 5,3,4; 5,3,2; 5,2,1];
    % for i = 1:3
    %     [node,elem] = uniformrefine(node,elem);
    % end
    % r = sqrt(node(:,1).^2 + node(:,2).^2 + node(:,3).^2);
    % node = node./[r r r];
    % figure;
    % subplot(1,2,1);
    % showmesh(node,elem);
    % subplot(1,2,2);
    % showmesh(node,elem,'Facecolor','y','Facealpha',0.5);
%
%   See also showmesh3, showsolution, showboundary3.
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

dim = size(node,2);
nv = size(elem,2);
if (dim==2) && (nv==3) % planar triangulation
    h = trisurf(elem(:,1:3),node(:,1),node(:,2),zeros(size(node,1),1));
    % 修改面颜色为白色，边颜色为淡蓝色
    set(h,'facecolor','w','edgecolor',[ 0.0745 0.6235 1.0000]); % 淡蓝色 RGB: [0.7, 0.8, 1]
    view(2); axis equal; axis tight; axis off;
end
if (dim==2) && (nv==4) % planar quadrilateration
    h = patch('Faces', elem, 'Vertices', node);
    % 修改面颜色为白色，边颜色为淡蓝色
    set(h,'facecolor','w','edgecolor',[ 0.0745 0.6235 1.0000]); % 淡蓝色 RGB: [0.7, 0.8, 1]
    view(2); axis equal; axis tight; axis off;
end
if (dim==3) 
    if size(elem,2) == 3 % surface meshes
        h = trisurf(elem(:,1:3),node(:,1),node(:,2),node(:,3));    
        % 修改面颜色为白色，边颜色为淡蓝色，保留透明度
        set(h,'facecolor','w','edgecolor',[ 0.0745 0.6235 1.0000],'FaceAlpha',0.75); % 淡蓝色 RGB: [0.7, 0.8, 1]   
        view(3); axis equal; axis off; axis tight;    
    elseif size(elem,3) == 4
        showmesh3(node,elem,varargin{:});
        return
    end
end 
if (nargin>2) && ~isempty(varargin) % set display property
    if isnumeric(varargin{1})
        view(varargin{1});
        if nargin>3
            set(h,varargin{2:end});
        end
    else
        set(h,varargin{1:end});        
    end
end
end

function [node,elem,bdFlag,indexMap,tree] = coarsen(node,elem,markedElem,bdFlag)
%% COARSEN coarsen a 2-D triangulation.
%
% [node,elem] = COARSEN(node,elem,markedElem) removes good-to-coarsen
% nodes whose star are marked for coarsening
%
% [node,elem,bdFlag] = COARSEN(node,elem,markedElem,bdFlag) updates
% boundary conditions represented by bdFlag.
%
% [node,elem,bdFlag,indexMap,tree] = COARSEN(node,elem,markedElem,bdFlag)
% outputs two additional information: indexMap and tree. 
%
% - indexMap is the map between nodes in the fine mesh (node in the input)
%   to that in the coarse mesh (node in the output). For example,
%   indexMap(10) = 6 means the 10-th node in the fine grid  is now the 6-th
%   node in the coarse one. indexMap is useful for the interpolation of
%   nodewise function; see also nodeinterpolate
%
% - tree(:,1:3) stores the binary tree of the coarsening. tree(:,1) is the
%   index of parent element in coarsened mesh and tree(:,2:3) are two
%   children indices in original mesh.
%
% Example
%
%     load data/Lshapemesh;
%     set(gcf,'Units','normal'); set(gcf,'Position',[0.25,0.25,0.7,0.4]);
%     subplot(1,3,1); showmesh(node,elem);
%     [node,elem] = coarsen(node,elem,'all');
%     subplot(1,3,2); showmesh(node,elem);
%     [node,elem] = coarsen(node,elem,'all');
%     subplot(1,3,3); showmesh(node,elem);
% 
% Reference page in Help browser
%  <a href="matlab:ifem coarsendoc">ifem coarsendoc</a>
%
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

N = max(elem(:)); NT = size(elem,1);
tree = []; indexMap  = (1:size(node,1))';
if ~exist('node','var') || isempty(node), node = []; end
if ~exist('bdFlag','var') || isempty(bdFlag), bdFlag = []; end
if isempty(markedElem), return; end
if strcmp(markedElem,'all'), markedElem = (1:size(elem,1))'; end
if islogical(markedElem), markedElem = find(markedElem); end

%% Find good-to-coarsen nodes
valence = accumarray(elem(:),ones(3*NT,1),[N 1]);
markedVal = accumarray(elem(markedElem,1),ones(length(markedElem),1),[N 1]);
isIntGoodNode = ((markedVal==valence) & (valence==4));
isBdGoodNode = ((markedVal==valence) & (valence==2));
NTdead = 2*sum(isIntGoodNode) + sum(isBdGoodNode); 
if (NTdead == 0), return; end

%% Remove interiori good-to-coarsen nodes
t2v = sparse([1:NT,1:NT,1:NT], elem(1:NT,:), 1, NT, N);
% Find stars for good-to-coarsen nodes
[ii,jj] = find(t2v(:,isIntGoodNode));
if length(jj)<0
    error('number of good nodes are not correct')
end
nodeStar = reshape(ii,4,sum(isIntGoodNode));
isIntNode = false(size(nodeStar,2),1);
% isIntNode is used to exclude those bd nodes whose val = 4
% case: 1 2 3 4
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(4,:),2)) & ...
      (elem(nodeStar(1,:),2) == elem(nodeStar(2,:),3));  
nodeStar(:,idx)  = nodeStar([1 2 3 4],idx);
isIntNode(idx) = true;
% case: 1 2 4 3
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(3,:),2)) & ...
      (elem(nodeStar(1,:),2) == elem(nodeStar(2,:),3));  
nodeStar(:,idx)  = nodeStar([1 2 4 3],idx); 
isIntNode(idx) = true;
% case: 1 3 2 4
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(4,:),2)) & ...
      (elem(nodeStar(1,:),2) == elem(nodeStar(3,:),3));  
nodeStar(:,idx)  = nodeStar([1 3 2 4],idx); 
isIntNode(idx) = true;
% case: 1 3 4 2
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(2,:),2)) & ...
      (elem(nodeStar(1,:),2) == elem(nodeStar(3,:),3));  
nodeStar(:,idx)  = nodeStar([1 3 4 2],idx); 
isIntNode(idx) = true;
% case: 1 4 2 3
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(3,:),2)) & ...
      (elem(nodeStar(1,:),2) == elem(nodeStar(4,:),3));  
nodeStar(:,idx)  = nodeStar([1 4 2 3],idx); 
isIntNode(idx) = true;
% case: 1 4 3 2
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(2,:),2)) & ...
      (elem(nodeStar(1,:),2) == elem(nodeStar(4,:),3));  
nodeStar(:,idx)  = nodeStar([1 4 3 2],idx); 
isIntNode(idx) = true;
% merge t1 with t2, and t3 with t4
t1 = nodeStar(1,isIntNode); 
t2 = nodeStar(2,isIntNode); 
t3 = nodeStar(3,isIntNode);
t4 = nodeStar(4,isIntNode);
p2 = elem(t1,3); 
p3 = elem(t2,2); 
p4 = elem(t1,2); 
p5 = elem(t3,2);
elem(t1,:) = [p4 p2 p3]; 
elem(t2,1) = 0;
elem(t3,:) = [p5 p3 p2]; 
elem(t4,1) = 0;
% update isIntGoodNode
intGoodNode = find(isIntGoodNode);
isIntGoodNode(intGoodNode(~isIntNode)) = false;

%% Remove boundary good-to-coarsen nodes
% Find stars for good-to-coarsen nodes
[ii,jj] = find(t2v(:,isBdGoodNode));
if length(jj)<0
    error('number of good nodes are not correct')
end
nodeStar = reshape(ii,2,sum(isBdGoodNode));
idx = (elem(nodeStar(1,:),3) == elem(nodeStar(2,:),2));
nodeStar(:,idx)  = nodeStar([2 1],idx); 
t5 = nodeStar(1,:); 
t6 = nodeStar(2,:);
p1 = elem(t5,1);
p2 = elem(t5,3); 
p3 = elem(t6,2); 
p4 = elem(t5,2);
if ~isempty(node)
    v13 = node(p1,:) - node(p3,:);
    v12 = node(p1,:) - node(p2,:);
    v23 = node(p2,:) - node(p3,:);
    % Corner/feature points could be good nodes. Remove these points will
    % change the shape of the domain. We check if nodes are feature points by
    % computing the length differences.
    lengthDiff = (sum(v12.^2,2) + sum(v13.^2,2) - sum(v23.^2,2))./sum(v23.^2,2);
    idx = (sqrt(lengthDiff) < 1e-3);
else
    idx = true(length(t5),1);
end
elem(t5(idx),:) = [p4 p2 p3]; 
elem(t6(idx),1) = 0;
bdGoodNode = find(isBdGoodNode);
isBdGoodNode(bdGoodNode(~idx)) = false;

%% Update boundary edges
if (nargin==4) && (~isempty(bdFlag))
	bdFlag(t1,:) = [bdFlag(t1,2) bdFlag(t2,1) bdFlag(t1,1)];
	bdFlag(t3,:) = [bdFlag(t3,2) bdFlag(t4,1) bdFlag(t3,1)];	
	bdFlag(t5,:) = [bdFlag(t5,2) bdFlag(t6,1) bdFlag(t5,1)];
    bdFlag((elem(:,1) == 0),:) = [];
else
	bdFlag=[];
end

%% Record tree structure
NTdead = 2*sum(isIntGoodNode) + sum(isBdGoodNode);
tree = zeros(NTdead,3,'uint32');
tree(1:NTdead,1) = [t1'; t3'; t5(idx)'];
tree(1:NTdead,2) = [t1'; t3'; t5(idx)'];
tree(1:NTdead,3) = [t2'; t4'; t6(idx)'];

%% Clean node and elem matrices
isRemoved = (elem(:,1) == 0);
elem(isRemoved,:) = [];
inCoarse = true(NT,1);
inCoarse(isRemoved) = false;
elemidxMap = zeros(NT,1);
elemidxMap(inCoarse) = 1:size(elem,1); 
tree(:,1) = elemidxMap(tree(:,1));
if ~isempty(node)
    node(isIntGoodNode | isBdGoodNode,:) = [];
end
indexMap = zeros(N,1);
Ndead = sum(isIntGoodNode) + sum(isBdGoodNode);
indexMap(~(isIntGoodNode | isBdGoodNode)) = 1:(N-Ndead);
elem = indexMap(elem);
end

function u = nodeinterpolate(u,HB)
%% NODEINTERPOLATE interpolate a piecewise linear function.
%
% u = nodeinterpolate(u,HB) interpolate a linear finite element function u
% from a coarse grid to a fine grid or the other way around. The input
% array HB(:,1:3) records hierarchical structure of new nodes HB(:,1) added
% from a coarse grid to a fine one. HB(:,2:3) are two parent nodes of
% HB(:,1). It can be obtained by bisection. 
%
% u = nodeinterpolate(u,indexMap) going from fine to coarse, HB =
% indexMap which is the map between the indices of nodes in the fine grid
% to that of the coarse grid. For example, indexMap(10) = 6 means the 10-th
% node in the fine grid is now the 6-th node in the coarse one. Therefore
% indexMap(k) = 0 means k is removed. indexMap is obtained by coarsening.
%
% Example
%   node = [0,0; 1,0; 1,1; 0,1];
%   elem = [2,3,1; 4,1,3];      
%   u = [0 0 0 1]; 
%   figure(1);
%   subplot(1,3,1); showsolution(node,elem,u,'EdgeColor','k');
%   [node,elem,~,HB] = bisect(node,elem,1);
%   u = nodeinterpolate(u,HB);
%   subplot(1,3,2); showsolution(node,elem,u,'EdgeColor','k');
%   [node,elem,~,~,indexMap] = coarsen(node,elem,1:size(elem,1));
%   u = nodeinterpolate(u,indexMap);
%   subplot(1,3,3); showsolution(node,elem,u,'EdgeColor','k');
%
% See also bisect, coarsen, eleminterpolate
% 
% Copyright (C) Long Chen. See COPYRIGHT.txt for details.

%%
if isempty(HB), return; end
if size(u,2) > size(u,1) % u is a column vector
    u = u';
end
oldN = size(u,1);
newN = max(size(HB,1),max(HB(:,1)));
if oldN >= newN % fine grid to coarse grid
    idx = (HB == 0);
    u(idx,:) = [];
else            % coarse grid to fine grid
    u(newN,:) = 0;         % preallocation
    if min(HB(:,1))>oldN % only new nodes are recorded in HB (2-D bisection)
        u(HB(1:end,1),:) = (u(HB(1:end,2),:)+u(HB(1:end,3),:))/2;
    else        % new nodes is stored starting from oldN (3-D bisection)
        while oldN < newN
            newNode = (oldN+1):newN;
            firstNewNode = newNode((HB(newNode,2) <= oldN) & (HB(newNode,3) <= oldN));        
            u(HB(firstNewNode,1),:) = (u(HB(firstNewNode,2),:) + u(HB(firstNewNode,3),:))/2;
            oldN = firstNewNode(end);
        end
    end
end
end

function V = FF_org(U,poten)
theta = 0.8; theta_c = 1.6;
if poten==1
   V = mobility(U).*U.*(1 - U.*U); 
end
if poten==2
   V = mobility(U).*(theta/2*(log(1-U)-log(1+U))+theta_c*U);   
end
end

function V = FF(U,poten,kappa) 
V = FF_org(U,poten) + kappa*U;
end

function V = mobility(U)
V = U.^2;
end
