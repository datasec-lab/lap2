%% ============================================================
%  Unified variance vs epsilon (ONE figure)
%  Practical q-set; min per-step noise per ε
%     Lap2: 2 b_opt(ε)^2   (red)
%     Gauss: sigma_opt(ε)^2 (blue)
%  Title adds constants if q_opt & C_opt are fixed across ε
%  Table: ε | q_Lap2 | C_opt | b_opt | q_Gauss | sigma_opt
% ============================================================
clear; clc;

%% ---------- Privacy / training setup ----------
delta_target = 1e-5;    L = log(1/delta_target);
E_epochs = 1;                                  % fixed epochs
N_ref    = 86000000; % model size
HN_ref   = psi(N_ref+1) + 0.5772156649015328606;

% epsilon grid and Phi(ε,δ)
eps_grid = [1, 2, 3];%linspace(0.5, 0.75, 2);
Phi      = (sqrt(eps_grid + L) - sqrt(L)).^2;

%% ---------- Policy / caps / practicality ----------
C_min     = 1;           % Lap2 policy: C_opt = C_min
b_max     = 10;           % Lap2 per-step cap
sigma_max = 10;           % Gaussian per-step cap
n=50000;
% Practical batch sizes => practical q-set; must also satisfy T=E/q ≤ T_max
B_set   = [5000];   % tune per task/hw
q_set   = B_set / n;

q_min   = 1e-5;             % absolute floor
q_max   = 1;             % absolute ceiling
T_max   = 7810;            % steps bound
TAU     = 1e-12;            % tiny interior margin

% Feasible practical q-set (discrete)
T_set   = E_epochs ./ q_set;
keep    = (q_set >= q_min) & (q_set <= q_max) & (q_set < 1) & (T_set <= T_max);
q_all   = unique(round(q_set(keep), 12));
if isempty(q_all)
    error('No practical q found. Adjust B_set, q_min/q_max, or T_max.');
end

%% ---------- Optimize per ε (true argmin over practical q) ----------
Lap_q   = nan(size(eps_grid));   % q_opt (Lap2)
Lap_b   = nan(size(eps_grid));   % b_opt (Lap2)
Lap_var = nan(size(eps_grid));   % 2 b^2 (Lap2 unified variance)

Gau_q   = nan(size(eps_grid));   % q_opt (Gaussian)
Gau_sig = nan(size(eps_grid));   % sigma_opt (Gaussian)
Gau_var = nan(size(eps_grid));   % sigma^2 (Gaussian unified variance)

for i = 1:numel(eps_grid)
    Phi_i = Phi(i);

    % ---- Lap2: b = C_min * sqrt( E q HN / (8 Phi) ), choose q in q_all, b < b_max
    b_vals = C_min * sqrt( (E_epochs .* q_all * HN_ref) ./ (8 * Phi_i) );
    feasL  = (b_vals < b_max - TAU);
    if any(feasL)
        [b_star, idxL] = min(b_vals(feasL));
        q_star = q_all(feasL); q_star = q_star(idxL);
        Lap_q(i)   = q_star;
        Lap_b(i)   = b_star;
        Lap_var(i) = 2 * b_star^2;
    end

    % ---- Gaussian: sigma = sqrt( E q / (2 Phi) ), choose q in q_all, sigma < sigma_max
    sig_vals = sqrt( (E_epochs .* q_all) ./ (2 * Phi_i) );
    feasG    = (sig_vals < sigma_max - TAU);
    if any(feasG)
        [sig_star, idxG] = min(sig_vals(feasG));
        q_star = q_all(feasG); q_star = q_star(idxG);
        Gau_q(i)   = q_star;
        Gau_sig(i) = sig_star;
        Gau_var(i) = sig_star^2;
    end
end

%% ---------- ONE plot: unified variance vs ε (log y-axis) ----------
fig = figure('Color','w','Units','inches','Position',[1 1 6.6 3.4]); % compact
baseFont = 12; lwMain = 1.8;
colLap = [0.80 0.15 0.15];  % red
colGau = [0.05 0.25 0.75];  % blue

set(fig,'DefaultAxesFontName','Times','DefaultAxesFontSize',baseFont,...
         'DefaultTextFontName','Times','DefaultTextFontSize',baseFont);

ax = axes(fig); hold(ax,'on'); grid(ax,'on');

maskL = isfinite(Lap_var);
maskG = isfinite(Gau_var);

pL = plot(ax, eps_grid(maskL), Lap_var(maskL), '-',  'Color', colLap, 'LineWidth', lwMain, ...
          'DisplayName','Lap2: 2b_{opt}^{2}(\epsilon)');
pG = plot(ax, eps_grid(maskG), Gau_var(maskG), '--', 'Color', colGau, 'LineWidth', lwMain, ...
          'DisplayName','Gaussian: \sigma_{opt}^{2}(\epsilon)');

xlabel(ax,'\epsilon','Interpreter','tex');
ylabel(ax,'Unified variance (per step)','Interpreter','tex');
set(ax,'YScale','log','YMinorTick','on','LineWidth',0.9);

% Format x ticks as 0.1 style
xt = get(ax,'XTick');
ax.XTickLabel = arrayfun(@(x)sprintf('%.1f', x), xt, 'UniformOutput', false);

% ----- Title: add constants if q_opt & C_opt are fixed across ε -----
tol = 1e-12;
isConstLapQ = any(maskL) && all(abs(Lap_q(maskL) - Lap_q(find(maskL,1))) < tol);
isConstGauQ = any(maskG) && all(abs(Gau_q(maskG) - Gau_q(find(maskG,1))) < tol);
if isConstLapQ && isConstGauQ
    title(ax, sprintf('Unified variance vs \\epsilon   C_{opt}=C_{min}=%.2f,  q_{opt}^{Lap2}=%.6f,  q_{opt}^{Gauss}=%.6f', ...
          C_min, Lap_q(find(maskL,1)), Gau_q(find(maskG,1))), 'Interpreter','tex');
else
    title(ax, sprintf('Unified variance vs \\epsilon   (C_{opt}=C_{min}=%.2f)', C_min), 'Interpreter','tex');
end

legend(ax,[pL pG],'Location','best','Box','off','Interpreter','tex');
set(ax,'TickDir','out','TickLength',[0.010 0.010]);

% Export vector PDF
if ~exist('figs','dir'); mkdir figs; end
set(fig,'Renderer','painters');
exportgraphics(fig,'figs/UnifiedVariance_vs_Epsilon_minNoise.pdf',...
    'ContentType','vector','BackgroundColor','white');

%% ---------- Print final configuration table ----------
fprintf('\n--- Per-ε optimized configs over practical q (min per-step noise) ---\n');
fprintf('%8s | %10s | %10s | %10s || %10s | %10s\n', ...
    'ε','q_Lap2','C_opt','b_opt','q_Gauss','sigma_opt');
fprintf(repmat('-',1,78)); fprintf('\n');
for i = 1:numel(eps_grid)
    ql = Lap_q(i); bl = Lap_b(i); qg = Gau_q(i); sg = Gau_sig(i);
    if isnan(ql), ql_str='-'; else, ql_str=sprintf('%.5f', ql); end
    cl_str = sprintf('%.3f', C_min);   % fixed policy
    if isnan(bl), bl_str='-'; else, bl_str=sprintf('%.5f', bl); end
    if isnan(qg), qg_str='-'; else, qg_str=sprintf('%.5f', qg); end
    if isnan(sg), sg_str='-'; else, sg_str=sprintf('%.5f', sg); end
    fprintf('%8.2f | %10s | %10s | %10s || %10s | %10s\n', ...
        eps_grid(i), ql_str, cl_str, bl_str, qg_str, sg_str);
end