% Function that estimates the cluster centroids, prototype M-snippets,
% dictionary atoms, or filter bank elements for a particular EEG rhythm
% Training stage
% Author: Carlos Loza
% carlos85loza@gmail.com

%%
function [D, dist_cluster_it] = FIR_Learning(X_PhEv, K, n_rep, n_it)
% INPUTS:
% X_PhEv - M-snippets corresponding to phasic event component only. Matrix 
% format.
% K - Number of dictionary atoms to be learned/estimated
% n_rep - Number of replicates of alternating optimization
% n_it - Number of iterations of alternating optimization
% OUTPUTS:
% D - Dictionary, M x K matrix
% dist_cluster_it - Distance of M-snippets to cluster centroids/atoms over 
% iterations for the best case (case with lowest coherence)

switch nargin
    case 1
        display('K (number of clusters) input is needed')
        return
    case 2
        % Number of replicates and number of iterations not provided
        n_rep = 10;
        n_it = 50;      
    case 3
        % Number of replicates provided        
        n_it = 50;
    case 4
        % Number of replicates and number of iterations provided
end

[M,n] = size(X_PhEv);
eps_stp = 10^-2;         % Stopping criterion for alternating optimizations
D_fin = cell(n_rep,1);
dist_cluster_rep_it = cell(n_rep,n_it);

for r = 1:n_rep
    display(['Repetition ' num2str(r) ' of ' num2str(n_rep)])
    % Initial Dictionary
    D = X_PhEv(:,randperm(n,K));
    fl_stp = 0;
    it = 1;
    D_pre = zeros(M,K);
    while fl_stp == 0
        %  Phasic Event Decomposition
        PhEv_D = PhEv_Decomp_train(X_PhEv, D, K);
        % Dictionary Learning
        [D, dist_cluster_rep_it{r,it}] = Dict_Learn(PhEv_D, D, K);
        
        % Convergence criterion
        aux_norm1 = abs(D_pre) - abs(D);
        aux_norm2 = sqrt(sum(aux_norm1.^2,1));
        if mean(aux_norm2) <= eps_stp
            fl_stp = 1;
            D_fin{r,1} = D;
            clear D
        elseif it == n_it
            fl_stp = 1;
            D_fin{r,1} = D;
            clear D
        else
            D_pre = D;
            it = it + 1;
        end
    end
end

% Choose dictionary with lowest mutual coherence
mu_coh = zeros(n_rep,1);
for r = 1:n_rep
    mu_coh(r,1) = Mutual_Coherence(D_fin{r,1});
end
[~, idx] = min(mu_coh);

D = D_fin{idx,1};
dist_cluster_it_aux = dist_cluster_rep_it(idx,:);
dist_cluster_it = cell(n_it,1);
for it = 1:n_it
    if isempty(dist_cluster_it_aux{1,it}) == 0;
        dist_cluster_it{it,1} = dist_cluster_it_aux{1,it};
    end
end
end

%%
function PhEv_D = PhEv_Decomp_train(X_PhEv, D, K)
% INPUTS:
% X_PhEv - M-snippets corresponding to phasic event component only. Matrix 
% format.
% D - Dictionary, M x K matrix
% K - Number of dictionary atoms to be learned/estimated
% OUTPUTS:
% PhEv_D - Sets of M-snippets corresponding to each cluster. Cell format

PhEv_D = cell(K,1);
n = size(X_PhEv,2);

for i = 1:n
    acorr = abs(corr(X_PhEv(:,i),D));
    [~, idx] = max(acorr);
    PhEv_D{idx,1} = [PhEv_D{idx,1} X_PhEv(:,i)];
end

end

%%
function [D_new, dist_K] = Dict_Learn(PhEv_D, D_old, K)
% INPUTS:
% PhEv_D - Sets of M-snippets corresponding to each cluster. Cell format
% D_old - Initial version of Dictionary
% K - Number of dictionary atoms to be learned/estimated
% OUTPUTS:
% D_new - Updated version of Dictionary
% dist_K - DIstances from each M-snippet to its corresponing cluster 
% centroid/atom. Cell format (K total)

% Stopping criterion for MCC-SVD
eps = 10^-4;
D_new = zeros(size(D_old));
dist_K = cell(K,1);

for i = 1:K
    M_snippet = PhEv_D{i,1};
    n_samp = size(M_snippet,2);
    if n_samp == 0
        % Nobody is using this cluster
        D_new(:,i) = D_old(:,i);
        dist_K{i,1} = 0;  
    elseif n_samp == 1
        D_new(:,i) = M_snippet - mean(M_snippet);
        dist_K{i,1} = 0;
    elseif n_samp > 1 && n_samp <= 5
        % Use regular svd, MCC-SVD not necessary
        [aux,~,~] = svds(M_snippet,1);
        D_new(:,i) = aux - mean(aux);
        % Fix/align signs of samples
        M_snippet_s = bsxfun(@times,sign(D_new(:,i)'*M_snippet),M_snippet);
        dist_K{i,1} = sum(bsxfun(@minus,D_new(:,i),M_snippet_s).^2,1);
    else
        % Robust MCC-SVD
        [aux,~,~] = MCC_SVD(M_snippet, eps, 1);
        D_new(:,i) = aux - mean(aux);
        % Fix/align signs of samples
        M_snippet_s = bsxfun(@times,sign(D_new(:,i)'*M_snippet),M_snippet);
        dist_K{i,1} = sum(bsxfun(@minus,D_new(:,i),M_snippet_s).^2,1);
    end
end

end

%%
function mu = Mutual_Coherence(D)
% INPUTS:
% D - Dictionary MxK
% OUTPUTS:
% mu - Coherence

K = size(D,2);

if K == 1
    mu = NaN;
else
    ct_c = 1;
    aux_coh = zeros(0,0);
    for i = 1:K-1
        for j = i+1:K
            aux_coh(ct_c) = D(:,i)'*D(:,j);
            ct_c = ct_c + 1;
        end
    end
    mu = max(abs(aux_coh));
end

end
