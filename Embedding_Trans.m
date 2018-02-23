% Function that performs the Embedding Transform on a set of single-channel
% bandpassed EEG recordings
% Author: Carlos Loza
% carlos85loza@gmail.com

%%
function [X_M_snippet_tr, beta_tr, beta_all] = Embedding_Trans(X,M)
% INPUTS:
% X - EEG data. It can be 1. single-trial (row vector) 
% 2. multi-trial/same duration (matrix form), or
% 3. multi-trial/different durations (cell) input of single-channel
% bandpassed EEG data
% KEY: Single traces MUST be row vectors
% M - Dimensionality of embedding, i.e. duration (in samples) of putative
% events
% OUTPUTS:
% X_M_snippet_tr - M-snippets separated by trials
% beta_tr - Amplitude/Norm of all possible M-dimensional snippets separated
% by trials and mapped directly to the M-snippets in X_M_snippet_tr
% beta_all - Amplitude/Norm of all possible M-dimensional snippets for all
% trials, i.e. Embedding Transform

% Check if input is cell
X = squeeze(X);
n_tr = size(X,1);               
if iscell(X) == 0
    X = mat2cell(X,ones(1,n_tr));
end

X_M_snippet_tr = cell(n_tr,1);
beta_tr = cell(n_tr,1);

% Amplitude of Hilbert Transform
X_abs = cell(n_tr,1);
for i = 1:n_tr
    X_abs{i,1} = abs(hilbert(X{i,1}));
end

% Smooth the instantaneous amplitudes
X_abs_sm = cell(n_tr,1);
aux_M = round(M/2);
spn = round(aux_M/2)*2 - 1;
for i = 1:n_tr
    X_abs_sm{i,1} = smooth(X_abs{i,1},spn);
end

% Find peaks first (Putative neuromodulations)
X_pks = cell(n_tr,1);
min_pk_d = round(1*M);          % Minimum distance between adjacent peaks
for i = 1:n_tr
    [~, pk_loc] = findpeaks(X_abs_sm{i,1},'MinPeakDistance',min_pk_d,'SortStr','descend');
    X_pks{i,1} = pk_loc;
end

% Embedding Transform
beta_all = zeros(0,0);
for i = 1:n_tr
    [X_M_snippet, beta_aux] = Embedding_Trans_all(X{i,1},M,X_pks{i,1});
    beta_all = [beta_all; abs(beta_aux)];       % Stack norms from all trials
    X_M_snippet_tr{i,1} = X_M_snippet;
    beta_tr{i,1} = beta_aux;
end

end

%%
function [X_M_snippet, beta_trial] = Embedding_Trans_all(x,M,pk_loc)
% INPUTS:
% x - Single-channel, single-trial bandpassed EEG trace
% M - Dimensionality of embedding, i.e. duration (in samples) of putative
% events
% pk_loc - Timestamps of peaks in the instantaneous amplitude trace
% corresponding to input x
% OUTPUTS:
% X_M_snippet - 
% beta_trial - Amplitude/Norm of all possible M-dimensional snippets, i.e. 
% Embedding Transform
% X_M_snippet - M-snippets discovered/extracted in the current trial

N = length(x);
L = ceil((N+M-1)/M);    % Maximum possible number of non-overlapping atoms
stp_fl = 0;             % Stopping criterion/flag
X_M_snippet = zeros(M,L);
beta_trial = zeros(L,1);
n_pks = length(pk_loc);

aux_x = x;

% Start with clear peaks
for i = 1:n_pks
    if pk_loc(i) - round(M/2) <= 0
        idx = 1:pk_loc(i) + round(M/2) - 1;
        x_norm = [zeros(1, M - length(idx)) x(idx)];
    elseif pk_loc(i) + round(M/2) - 1 > N
        idx = pk_loc(i) - round(M/2):N; 
        x_norm = [x(idx) zeros(1, M - length(idx))];
    else
        idx = pk_loc(i) - round(M/2):pk_loc(i) + round(M/2) - 1;
        x_norm = x(idx);
    end
    beta_trial(i,1) = norm(x_norm);
    X_M_snippet(:,i) = x_norm';
    aux_x(idx) = zeros(1,length(idx));
end
i = i + 1;

while stp_fl == 0
    % Check if there are any M-snippets left to be discovered
    [tau_p, fl] = check_potential_seg(aux_x, M);
    if fl == 0
        for j = 1:length(tau_p)
            idx = tau_p(j):tau_p(j) + M - 1;
            X_M_snippet(:,i) = x(idx)';
            beta_trial(i,1) = norm(x(idx));
            aux_x(idx) = zeros(1,length(idx));
            i = i + 1;
        end
    else
        stp_fl = 1;
    end
end

beta_trial = beta_trial(1:i-1,1);
X_M_snippet = X_M_snippet(:,i-1);

% % Optional - Not really necessary when trials are long, besides these
% % norms would not have a direct mapping to M-snippets in x
% % Put together all the remaining unconnected temporal samples and compute
% % norm
% x_rem = aux_x(aux_x ~= 0);
% n_rem = floor(length(x_rem)/M);
% beta_rem = zeros(n_rem,1);
% if n_rem > 0
%     for i = 1:n_rem
%         beta_rem(i) = norm(x_rem((i-1)*M+1:i*M));
%     end
%     beta_trial = [beta_trial; beta_rem];
% end

end

%%
function [tau_p, fl] = check_potential_seg(aux_x, M)
% INPUTS:
% aux_x - Single-channel, single-trial bandpassed EEG trace after removing
% already discovered M-snippets
% M - Dimensionality of embedding, i.e. duration (in samples) of putative
% events
% OUTPUTS:
% tau_p - Timestamps of M-snippets left to be extracted/discovered
% fl - Flag to determine end of search
% if fl = 0 -> no M-snippets left to be found
% if fl = 1 -> M-snippets still available

max_tau_ones = double((aux_x ~= 0));

aux_fl = conv(max_tau_ones,ones(1,M),'valid');
idx = find(aux_fl == M);

if isempty(idx) == 1
    fl = 1;
    tau_p = 0;
else
    fl = 0;
    aux_idx = find(diff(idx) >= M) + 1;
    tau_p = [idx(1) idx(aux_idx)];
end

end
