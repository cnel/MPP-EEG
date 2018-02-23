% Function that implements robust MCC-SVD
% Zero-mean variant
% Author: Carlos Loza
% carlos85loza@gmail.com
%%
function U = MCC_SVD(X, eps, mr, Corr_sigma)
% INPUTS:
% X - Input samples
% eps - Tolerance. Stopping criterion
% mr - Dimensionality (number of principal components) where outliers can
% be detected. In the case of a MPP approach for EEG, mr is equal to the 
% number of principal components to be estimated
% Corr_sigma - Kernel width for correntropy estimation
% OUTPUTS:
% U - First mr principal components. Matrix form

% KEY: For zero mean bases, force mu_t to be zero EVERYTIME

fl = 0;
% Initialization
[U_t,~,~] = svds(X,mr);
mu_t = mean(X,2);
mu_t = zeros(size(mu_t));
[d, n] = size(X);
ct_max = 50;            % Maximum number of iterations of HQ optimization

X = bsxfun(@minus,X,mu_t);       % Zero-mean variant

if nargin == 3
    ct = 1;
    while fl == 0
        % Kernel width calculation - Silverman's rule       
        aux = (X - (U_t*U_t')*X);
        X_d_t = sum(aux.^2,1);
        sig_e_t = std(X_d_t);
        R_t = iqr(X_d_t);
        Corr_sigma = 1.06*min([sig_e_t R_t/1.34])*n^(-1/5);
        Param_Corr = sum(X.^2,1) - sum((U_t'*X).^2,1);
        
        % HQ alternating optimizations
        p_t = -exp(-Param_Corr/(2*Corr_sigma));
        % Fast calculation of weighted covariance matrix
        JM = zeros(d, d, n);
        for k = 1:n
            JM(:,:,k) = X(:,k)*X(:,k)';
        end
        JM = reshape(JM,d^2,n);
        w_C = reshape(JM*(-p_t)', d, d);
        PCA_param = w_C/trace(w_C);               % Weighted covariance matrix
        
        if mean(isnan(PCA_param(:))) ~= 0
            display('NaN Warning')
        end
        
        if mean(isinf(PCA_param(:))) ~= 0
            display('Inf Warning')
        end
        
        [V,D] = eig(PCA_param);
        [~,idx] = sort(diag(D),'descend');
        V = V(:,idx);
        U_t1 = V(:,1:mr);
        
        % Comparison of successive estimations and stopping flags
        norm_diff = norm(abs(U_t) - abs(U_t1));       
        if ct == ct_max
            fl = 1;
        end      
        if norm_diff < eps
            fl = 1;
        end      
        if fl == 1
            U = U_t1;
        else
            U_t = U_t1;
            ct = ct + 1;
        end
    end  
elseif nargin == 4
    % Kernel width provided
    ct = 1;
    while fl == 0
        Param_Corr = sum(X.^2,1) - sum((U_t'*X).^2,1);
        % HQ alternating optimizations
        p_t = -exp(-Param_Corr/(2*Corr_sigma));
        % Fast calculation of weighted covariance matrix
        JM = zeros(d, d, n);
        for k = 1:n
            JM(:,:,k) = X(:,k)*X(:,k)';
        end
        JM = reshape(JM,d^2,n);
        w_C = reshape(JM*(-p_t)', d, d);
        PCA_param = w_C/trace(w_C);               % Weighted covariance matrix
        
        if mean(isnan(PCA_param(:))) ~= 0
            display('NaN Warning')
        end
        
        if mean(isinf(PCA_param(:))) ~= 0
            display('Inf Warning')
        end
        
        [V,D] = eig(PCA_param);
        [~,idx] = sort(diag(D),'descend');
        V = V(:,idx);
        U_t1 = V(:,1:mr);
        
        % Comparison of successive estimations and stopping flags
        norm_diff = norm(abs(U_t) - abs(U_t1));       
        if ct == ct_max
            fl = 1;
        end      
        if norm_diff < eps
            fl = 1;
        end      
        if fl == 1
            U = U_t1;
        else
            U_t = U_t1;
            ct = ct + 1;
        end
    end  
    
end
end
