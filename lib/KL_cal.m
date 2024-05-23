%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 : 计算两个高斯分布的KL散度，描述两个分布的相似度。数值越小分布越相似。
% 作者 : 周旭
% 日期 : 2022.10.10

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
clear all ; close all;

k1 = 5; % a聚类数目
k2 = 5; % b聚类数目
d = 8; % 参数使用数目

fileset1 = ['E:\multi-wave\'];
fileset2 = ['E:\multi-wave\'];
filemeans1 = [fileset1,'means.mat'];
filemeans2 = [fileset2,'means.mat'];
fileweight1 = [fileset1,'weights.mat'];
fileweight2 = [fileset2,'weights.mat'];
filecovariance1 = [fileset1,'covariances.mat'];
filecovariance2 = [fileset2,'covariances.mat'];
load(filemeans1);
means_GMM1 = means;
load(filecovariance1);
covariance_GMM1 = covariances;
load(filemeans2);
means_GMM2 = means;
load(filecovariance2);
covariance_GMM2 = covariances;
load(fileweight1);
weight_GMM1 = weights;
load(fileweight2);
weight_GMM2 = weights;

% KL散度
KL = zeros(k1,k2);
KL2 = zeros(k1,k2);

for i = 1:k1
    for j = 1:k2
        mu1 = means_GMM1(i,:);
        mu2 = means_GMM2(j,:);
        Sigma1 = reshape(covariance_GMM1(i,:,:),[d,d]);
        Sigma2 = reshape(covariance_GMM2(j,:,:),[d,d]);
        KL(i,j) = 0.5*(log(det(Sigma2) / det(Sigma1)) - d + trace(inv(Sigma2) * Sigma1) + (mu1-mu2) * inv(Sigma2) * (mu1-mu2)');
    end
end

for i = 1:k1
    for j = 1:k2
        mu2 = means_GMM1(i,:);
        mu1 = means_GMM2(j,:);
        Sigma2 = reshape(covariance_GMM1(i,:,:),[d,d]);
        Sigma1 = reshape(covariance_GMM2(j,:,:),[d,d]);
        KL2(i,j) = 0.5*(log(det(Sigma2) / det(Sigma1)) - d + trace(inv(Sigma2) * Sigma1) + (mu1-mu2) * inv(Sigma2) * (mu1-mu2)');
    end
end

for i = 1:k1
    for j = 1:k2
        KL_all(i,j) = min(KL(i,j),KL2(i,j));
    end
end

for i = 1:k1
    for j = 1:k2
        KL_all2(i,j) = (KL(i,j)+KL2(i,j))/2;
    end
end
% 巴氏距离
BD = zeros(k1,k2);

for i = 1:k1
    for j = 1:k2
        mu1 = means_GMM1(i,:);
        mu2 = means_GMM2(j,:);
        Sigma1 = reshape(covariance_GMM1(i,:,:),[d,d]);
        Sigma2 = reshape(covariance_GMM2(j,:,:),[d,d]);
        Sigma_num = 0.5*(Sigma1 + Sigma2);
        BD(i,j) = 0.5*log(det(Sigma_num)./sqrt(det(Sigma1*Sigma2)))+1/8*((mu1-mu2) * inv(Sigma_num) * (mu1-mu2)');
    end
end

% W距离
W = zeros(k1,k2);

for i = 1:k1
    for j = 1:k2
        mu1 = means_GMM1(i,:);
        mu2 = means_GMM2(j,:);
        Sigma1 = reshape(covariance_GMM1(i,:,:),[d,d]);
        Sigma2 = reshape(covariance_GMM2(j,:,:),[d,d]);
        Sigma_num = 0.5*(Sigma1 + Sigma2);
        W(i,j) = sum(mu1-mu2) + trace(Sigma1) + trace(Sigma2) - 2*trace(sqrtm(sqrtm(Sigma1)*Sigma2*sqrtm(Sigma1)));
    end
end