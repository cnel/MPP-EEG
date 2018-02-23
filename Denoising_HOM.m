% Function that separates noise component from phasic event component
% according to higher-order statistical moments (HOM)
% Author: Carlos Loza
% carlos85loza@gmail.com

%%
function [X_PhEv, th] = Denoising_HOM(beta_all, beta_tr, X_M_snippet_tr)
% INPUTS:
% beta_all - Amplitude/Norm of all possible M-dimensional snippets for all
% trials, i.e. Embedding Transform. Vector format.
% beta_tr - Amplitude/Norm of all possible M-dimensional snippets separated
% by trials and mapped directly to the M-snippets in X_M_snippet_tr. Cell
% format
% X_M_snippet_tr - M-snippets separated by trials. Cell format.
% OUTPUTS:
% X_PhEv - M-snippets corresponding to phasic event component only. Cell 
% format.
% th - Threshold to discriminate between noise and phasic event component
% according to the Embedding Transform and HOM. 

% Estimate threshold between components
th = Threshold_HOM(beta_all);

% Separate phasic event component only
n_tr = size(X_M_snippet_tr,1);
X_PhEv = zeros(0,0);
for i = 1:n_tr
    beta_aux = beta_tr{i,1};
    idx = find(beta_aux >= th);
    X_PhEv = [X_PhEv X_M_snippet_tr{i,1}(:,idx)];
end

end

%%
function th = Threshold_HOM(beta_all)
% INPUTS:
% beta_all - Amplitude/Norm of all possible M-dimensional snippets for all
% trials, i.e. Embedding Transform
% OUTPUTS:
% th - Threshold to discriminate between noise and phasic event component
% according to the Embedding Transform and HOM

prc_v = 5:95;
beta_pre = abs(beta_all);
skew_v = zeros(1,length(prc_v));
for i = 1:length(prc_v)
    idx = find(beta_pre < prctile(beta_pre,prc_v(i)));
    skew_v(i) = skewness(beta_pre(idx));
end
[~, idx_min] = min(abs(skew_v));            % Skewness value closest to zero
th = prctile(beta_pre,prc_v(idx_min));

end
