% Function that decomposes single-channel traces (for a particular EEG 
% rhythm) according to a Marked Point Process framework
% Testing stage
% Author: Carlos Loza
% carlos85loza@gmail.com

%%
function MPP_c = MPP_Decomp(X, D, th)
% INPUTS
% X - EEG data. It can be 1. single-trial (row vector) 
% 2. multi-trial/same duration (matrix form), or
% 3. multi-trial/different durations (cell) input of single-channel
% bandpassed EEG data
% KEY: Single traces MUST be row vectors
% D -  Dictionary, M x K matrix
% th - Threshold to discriminate between noise and phasic event component
% according to the Embedding Transform and HOM (Higher-order moments). 
% OUTPUTS
% MPP_c - cell with matrices that have the Marked Point Process features,
% i.e. amplitude, timing and index (each cell element is a particular trial)

% Check if input is cell
X = squeeze(X);
n_tr = size(X,1);               
if iscell(X) == 0
    X = mat2cell(X,ones(1,n_tr));
end

MPP_c = cell(n_tr,1);
for i = 1:n_tr
    [alph, tau, D_idx] = PhEv_nonovp(X{i,1}, D, th);
    MPP_c{i,1} = [tau alph D_idx];
end

end

%%
function [alph, tau, D_idx] = PhEv_nonovp(x, D, th)
% FFT-based algorithm for Decomposing single-channel, single trial EEG 
% traces.
% Non-overlapping phasic events only
% Critically sparse case -> vector quantization approach
% INPUTS:
% x - input time series (single-trial, single channel), N time samples
% D - dictionary, MxK
% th - Threshold to discriminate between noise and phasic event component
% according to the Embedding Transform and HOM (Higher-order moments). 
% OUTPUTS:
% alph - decomposition amplitudes, column vector Lx1
% tau - decomposition timings (centered), column vector Lx1
% D_idx - index of atoms used in the decomposition, column vector Lx1

N = length(x);
[M,~] = size(D); 
stp_fl = 0;                             % Stopping flag for main loop
% L is the number of extracted phasic events. It will depend on the
% temporal structure of the EEG trace as well as on the Dictionary elements
L = ceil((N+M-1)/M);                    % Maximum possible number of non-overlapping atoms
alph = zeros(L,1);
tau = zeros(L,1);
D_idx = zeros(L,1);

% Check if input is row
if iscolumn(x) == 0
    x = x';
end

% Only one run over the dictionary is necessary
corrs = conv2(x,flipud(D));
abs_corrs = abs(corrs);
[max_tau, max_D_idx] = max(abs_corrs,[],2);

% First iteration
[~, idx_max] = max(max_tau);
alph(1,1) = corrs(idx_max, max_D_idx(idx_max));
if abs(alph(1,1)) > th
    if idx_max - M <= 0                         % Condition for left edge
        z_padd = idx_max - 1;
        max_tau(1:z_padd+M) = zeros(z_padd+M,1);
    elseif idx_max > N                          % Condition for right edge
        z_padd = length(max_tau) - idx_max;
        max_tau(idx_max-M+1:end) = zeros(z_padd+M,1);
    else
        max_tau(idx_max-M+1:idx_max+M-1) = zeros(2*M-1,1);
        tau(1,1) = idx_max - M + 1;
        D_idx(1,1) = max_D_idx(idx_max);
    end

    % Remaining iterations
    i = 2;
    while stp_fl == 0
        [tau_p, fl] = check_potential_PhEv(max_tau, M);
        if fl == 0
            for j = 1:length(tau_p)
                [~, idx_max] = max(max_tau);
                alph(i,1) = corrs(idx_max, max_D_idx(idx_max));
                if abs(alph(i,1)) <= th
                    stp_fl = 1;
                    if idx_max - M > 0 && idx_max < N
                        tau(i,1) = idx_max - M + 1;
                        D_idx(i,1) = max_D_idx(idx_max);
                    else
                        alph(i,1) = 0;
                    end
                    i = i + 1;
                    break;
                end
                if idx_max - M <= 0                         % Condition for left edge
                    z_padd = idx_max - 1;
                    max_tau(1:z_padd+M) = zeros(z_padd+M,1);
                elseif idx_max > N                          % Condition for right edge
                    z_padd = length(max_tau) - idx_max;
                    max_tau(idx_max-M+1:end) = zeros(z_padd+M,1);
                else
                    max_tau(idx_max-M+1:idx_max+M-1) = zeros(2*M-1,1);
                    tau(i,1) = idx_max - M + 1;
                    D_idx(i,1) = max_D_idx(idx_max);
                    i = i + 1;
                end
            end
        else
            stp_fl = 1;
        end   
    end
    
    % Update outputs
    alph = alph(1:i-1,1);
    tau = tau(1:i-1,1);
    D_idx = D_idx(1:i-1,1);
    
    % Center timings
    tau = tau + round(M/2);
      
else
    % No M-snippet will/can be slected according to the threshold input
    display('No M-snippets satisfy the threshold criterion')
    alph = zeros(0,0);
    tau = zeros(0,0); 
    D_idx = zeros(0,0); 
end

end

%%
function [tau_p, fl] = check_potential_PhEv(max_tau, M)
% Function to check if there are any potential atoms left to be discovered
% in the non-overlapping case of the decomposition
% if fl = 0 -> no potential phasic events to be found
% if fl = 1 -> potential phasic events still available

max_tau_ones = double((max_tau ~= 0));

aux_fl = conv(max_tau_ones,ones(1,M),'valid');
idx = find(aux_fl == M);

if isempty(idx) == 1
    fl = 1;
    tau_p = 0;
else
    fl = 0;
    aux_idx = find(diff(idx) >= M) + 1;
    tau_p = [idx(1); idx(aux_idx)];
end

end
