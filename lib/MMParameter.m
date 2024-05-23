%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 : 计算全部穆勒矩阵参数
% 作者 : 周旭
% 日期 : 2022.6.27
% Output : MMT ; MMPD极化分解 ; MM参数 ; 其他参数 ; abrio参数 ; MMCD参数 ; 
%          MMLD参数 ; 角度参数 ；

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ： 穆勒矩阵读取和功能选择
% 作者 ： 周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fclose all ; clear ; close all ; 

tic;
% 穆勒矩阵的读取
fileset = ['E:\1.实验数据\纵切对比12\']; % 粘贴文件路径，末尾添加\
filem11 = [fileset,'m11.mat'];
filename = [fileset,'FinalMM.mat'];
load(filename);
load(filem11);

[H_image,W_image] = size(FinalM11);%确认图像大小

se_size = 5; %滤波核大小

%功能选择，置0不计算
fun_filter =    1;
func_MMT =      1;
func_MMPD =     1;
func_abrio =    0;
func_equ =      0;
func_other =    0;
func_MM =       1;
func_angle =    0;
func_MMCD =     1;
func_MMLD =     0;
func_save =     1;
func_showreal = 1;
func_MMatrix =  0;

%区域选择
%range_regis = [x0,x1,y0,y1]; %任选区域

range_regis = [1,H_image,1,W_image];
FinalM11 = FinalM11(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM12 = FinalM12(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM13 = FinalM13(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM14 = FinalM14(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM21 = FinalM21(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM22 = FinalM22(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM23 = FinalM23(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM24 = FinalM24(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM31 = FinalM31(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM32 = FinalM32(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM33 = FinalM33(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM34 = FinalM34(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM41 = FinalM41(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM42 = FinalM42(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM43 = FinalM43(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 
FinalM44 = FinalM44(range_regis(1):range_regis(2),range_regis(3):range_regis(4)); 

FinalMM(:,:,1) = FinalM11;
FinalMM(:,:,2) = FinalM12;
FinalMM(:,:,3) = FinalM13;
FinalMM(:,:,4) = FinalM14;
FinalMM(:,:,5) = FinalM21;
FinalMM(:,:,6) = FinalM22;
FinalMM(:,:,7) = FinalM23;
FinalMM(:,:,8) = FinalM24;
FinalMM(:,:,9) = FinalM31;
FinalMM(:,:,10) = FinalM32;
FinalMM(:,:,11) = FinalM33;
FinalMM(:,:,12) = FinalM34;
FinalMM(:,:,13) = FinalM41;
FinalMM(:,:,14) = FinalM42;
FinalMM(:,:,15) = FinalM43;
FinalMM(:,:,16) = FinalM44;

toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：均值滤波
% 作者 ：周旭
% 修改时间 ：2022.9.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if fun_filter == 1
FinalM11 = imfilter(FinalM11,fspecial('average',se_size),'replicate');
FinalM12 = imfilter(FinalM12,fspecial('average',se_size),'replicate');
FinalM13 = imfilter(FinalM13,fspecial('average',se_size),'replicate');
FinalM14 = imfilter(FinalM14,fspecial('average',se_size),'replicate');
FinalM21 = imfilter(FinalM21,fspecial('average',se_size),'replicate');
FinalM22 = imfilter(FinalM22,fspecial('average',se_size),'replicate');
FinalM23 = imfilter(FinalM23,fspecial('average',se_size),'replicate');
FinalM24 = imfilter(FinalM24,fspecial('average',se_size),'replicate');
FinalM31 = imfilter(FinalM31,fspecial('average',se_size),'replicate');
FinalM32 = imfilter(FinalM32,fspecial('average',se_size),'replicate');
FinalM33 = imfilter(FinalM33,fspecial('average',se_size),'replicate');
FinalM34 = imfilter(FinalM34,fspecial('average',se_size),'replicate');
FinalM41 = imfilter(FinalM41,fspecial('average',se_size),'replicate');
FinalM42 = imfilter(FinalM42,fspecial('average',se_size),'replicate');
FinalM43 = imfilter(FinalM43,fspecial('average',se_size),'replicate');
FinalM44 = imfilter(FinalM44,fspecial('average',se_size),'replicate');
CalibratedM11 = imfilter(CalibratedM11,fspecial('average',se_size),'replicate');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：计算MMT参数
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMT == 1
tic;

%结构特征参数
MMT_t1 = sqrt((FinalM22 - FinalM33).^2+(FinalM23 + FinalM32).^2)/2; %整体各向异性参数。表征样品整体的线各向异性度，即二向色性和相位延迟都有贡献，仅在强吸收下有数值不稳定的问题，在使用未归一化穆勒矩阵计算时可避免
MMT_b = (FinalM22 + FinalM33)/2; %反应退偏相关参数。和线偏振的退偏有关，同时也受各向异性效应的影响。同时根据医生的经验，这是一个对小粒子敏感的参数。
MMT_A = (2.*MMT_b.*MMT_t1) ./ (MMT_b.^2+MMT_t1.^2); %归一化的各向异性度参数,A为b的归一化参数
MMT_b2 = 1 - MMT_b; %代表样品退偏能力
MMT_beta = abs(FinalM23 - FinalM32) ./ 2; %疑似旋光参数，有旋光时它取值-sin(2a);也可以作为转置对称性破坏的指标之一

Bhls = (FinalM22.*FinalM33) - (FinalM23.*FinalM32); % 中心块的行列式，取值同二向色性和相位延迟都有关
Bfs = sqrt(FinalM22.^2 + FinalM33.^2 + FinalM23.^2 + FinalM32.^2); %中心块的范式，取值同二向色性和相位延迟都有关
cda = FinalM14+FinalM41; %圆二向色性各向异性度

MMT_b_tld = (FinalM22 - FinalM33) / 2 ; %%%仅在推导时用到的中间变量
MMT_beta_tld = (FinalM23 + FinalM32) / 2; %%%仅在推导时用到的中间变量


%棱的模
MMT_t_4243 = sqrt(FinalM42.^2+FinalM43.^2);%下棱的模q,双折射(birefringence) / 将线偏振转化成圆偏振的能力
MMT_t_2434 = sqrt(FinalM24.^2+FinalM34.^2);%右棱的模r,纯线相位延迟(linear retardance) / 将圆偏振转化为线偏振的能力
MMT_t_1213 = sqrt(FinalM12.^2+FinalM13.^2);%上棱的模D,二向色性(dichroism) / 线二向衰减
MMT_t_2131 = sqrt(FinalM21.^2+FinalM31.^2);%左棱的模P,柱散射(column scattering) / 线起偏


%叉乘
%叉乘反应的时角度差sin的函数。因此和角度差参数一样，转置对称时角度查为零。
PDxcheng = (FinalM12.*FinalM31 - FinalM13.*FinalM21) ./ (MMT_t_1213 .* MMT_t_2131);
rqxcheng = (FinalM24.*FinalM43 - FinalM34.*FinalM42) ./ (MMT_t_2434 .* MMT_t_4243);


%转置不对称参数（转动不变量）
%转置对称时参数取零值
MMTPD = sqrt((FinalM12-FinalM21).^2+(FinalM13-FinalM31).^2);
MMTrq = sqrt((FinalM24+FinalM42).^2+(FinalM34+FinalM43).^2);


%棱运算
PLsubDL = MMT_t_2131 - MMT_t_1213;
PLsubDLunited = PLsubDL ./ (MMT_t_2131 + MMT_t_1213);
rLsubqL = MMT_t_2434-MMT_t_4243;
rLsubqLunited = PLsubDL ./ (MMT_t_2434 + MMT_t_4243);
PLsubrL = MMT_t_2131-MMT_t_2434;
PLsubrLunited = PLsubDL ./ (MMT_t_2131 + MMT_t_2434);
DLsubqL = MMT_t_1213 - MMT_t_4243;
DLsubqLunited = PLsubDL ./ (MMT_t_1213 + MMT_t_4243);


% 各向异性方位角参数
MMT_phi2233 = 0.25*(atan2((FinalM23+FinalM32),(FinalM22-FinalM33)))*180/pi;%中心块方位角参量
MMT_phi3121 = 0.5*atan2((FinalM31),(FinalM21))*180/pi;%左棱方位角P
MMT_phi1312 = 0.5*atan2((FinalM13),(FinalM12))*180/pi;%上棱方位角D
MMT_phi4243 = 0.5*atan2((FinalM42),(-FinalM43))*180/pi;%下棱方位角q
MMT_phi2434 = 0.5*atan2((-FinalM24),(FinalM34))*180/pi;%右棱方位角r


%矢量差
%综合考虑棱之间的差和角度之间的差。同 关于转置对称性的破坏 解释。
PDslcha = sqrt(MMT_t_2131.^2+MMT_t_1213.^2-2*MMT_t_2131.*MMT_t_1213.*cos(rad2deg((MMT_phi3121-MMT_phi1312))));%PD矢量差
rqslcha = sqrt(MMT_t_2434.^2+MMT_t_4243.^2-2*MMT_t_2434.*MMT_t_4243.*cos(rad2deg((MMT_phi2434-MMT_phi4243))));%rq矢量差

toc;
disp('MMT参数计算完成');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：守恒量&恒等式
% 作者 ：周旭
% 修改时间 ：2022.6.27
% 说明 ：E1~E9无退偏时恒等于零，有退偏时并非转动不变量.对于各向同性样本，且
%        单次散射，且正前向或正背向接受的穆勒矩阵Es恒为零，这个参数同散射次
%        数有关。

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_equ == 1
tic;


%守恒量
MMT_P1 = abs(FinalM43+FinalM34);
MMT_P2 = sqrt(FinalM34.^2+FinalM43.^2);
MMT_P3 = sqrt(abs(FinalM34.*FinalM43));
MMT_P4 = abs(FinalM42+FinalM24);
MMT_P5 = sqrt(FinalM42.^2+FinalM24.^2);
MMT_P6 = sqrt(abs(FinalM42.*FinalM24));
MMT_P7 = FinalM22./max(FinalM22(:))+FinalM33./max(FinalM33(:))-FinalM44./max(FinalM44(:));
MMT_P8 = (FinalM42.*FinalM24+FinalM22+FinalM34.*FinalM43+FinalM33)./3;
MMT_P9 = abs(FinalM42.^2+FinalM43.^2-FinalM24.^2-FinalM34.^2);
MMT_P10 = FinalM24.*FinalM34-FinalM42.*FinalM43;
MMT_P11 = abs(FinalM24+FinalM34);
MMT_P12 = abs(FinalM42+FinalM43);


%恒等式
Es = FinalM11 - FinalM22 - FinalM33 + FinalM44;
E1 = (FinalM11+FinalM22).^2 - (FinalM12+FinalM21).^2 - (FinalM33+FinalM44).^2 - (FinalM34-FinalM43).^2; 
E2 = (FinalM11-FinalM22).^2 - (FinalM12-FinalM21).^2 - (FinalM33-FinalM44).^2 - (FinalM34+FinalM43).^2;
E3 = (FinalM11+FinalM21).^2 - (FinalM12+FinalM22).^2 - (FinalM13+FinalM23).^2 - (FinalM14+FinalM24).^2; 
E4 = (FinalM11-FinalM21).^2 - (FinalM12-FinalM22).^2 - (FinalM13-FinalM23).^2 - (FinalM14-FinalM24).^2;
E5 = (FinalM11+FinalM12).^2 - (FinalM21+FinalM22).^2 - (FinalM31+FinalM32).^2 - (FinalM41+FinalM42).^2;
E6 = (FinalM11-FinalM12).^2 - (FinalM21-FinalM22).^2 - (FinalM31-FinalM32).^2 - (FinalM41-FinalM42).^2;
E7 = (FinalM13.*FinalM14 - FinalM23.*FinalM24).*(FinalM33.^2-FinalM34.^2+FinalM43.^2-FinalM44.^2) - (FinalM33.*FinalM34 + FinalM43.*FinalM44).*(FinalM13.^2-FinalM14.^2-FinalM23.^2+FinalM24.^2);
E8 = (FinalM31.*FinalM41 - FinalM32.*FinalM42).*(FinalM33.^2-FinalM43.^2+FinalM34.^2-FinalM44.^2) - (FinalM33.*FinalM43 + FinalM34.*FinalM44).*(FinalM31.^2-FinalM41.^2-FinalM32.^2+FinalM42.^2);
E9 = (FinalM14.*FinalM23 - FinalM32.*FinalM42).*(FinalM33.^2-FinalM43.^2+FinalM34.^2-FinalM44.^2) - (FinalM42.*FinalM31 + FinalM41.*FinalM32).*(FinalM14.^2-FinalM24.^2-FinalM13.^2+FinalM23.^2);


toc;
disp('守恒量恒等式计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：abrio参数
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_abrio == 1
tic;

temp_1 = (FinalM12-FinalM42) ./ (FinalM44-FinalM14);
temp_2 = (FinalM43-FinalM13) ./ (FinalM44-FinalM14);
MMT_Abrio_R = atan(sqrt(temp_1.^2 + temp_2.^2))*180/pi; %相位延迟  parameter MMT_Abrio_R
MMT_Abrio_theta = 1/2*atan2(temp_1,temp_2)*180/pi; %角度分布  parameter MMT_Abrio_theta

toc;
disp('abrio参数计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：MM参数（Gil在2016年论文中提到的，没有研究其物理意义和应用）
% 作者 ：周旭
% 修改时间 ：2022.6.27
% 参考文献 ：Invariant quantities of a Mueller matrix under rotation and
%            retarder transformations.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MM == 1
tic;

MM_Det = zeros(H_image,W_image);
for i = 1:H_image
    for j = 1:W_image
        mm = [FinalM11(i,j),FinalM12(i,j),FinalM13(i,j),FinalM14(i,j);...
            FinalM21(i,j),FinalM22(i,j),FinalM23(i,j),FinalM24(i,j);...
            FinalM31(i,j),FinalM32(i,j),FinalM33(i,j),FinalM34(i,j);...
            FinalM41(i,j),FinalM42(i,j),FinalM43(i,j),FinalM44(i,j);];
        MM_Det(i,j) = det(mm); %mm的行列式
    end
end

MM_Norm = FinalM11.^2+FinalM12.^2+FinalM13.^2+FinalM14.^2+...
          FinalM21.^2+FinalM22.^2+FinalM23.^2+FinalM24.^2+...
          FinalM31.^2+FinalM32.^2+FinalM33.^2+FinalM34.^2+...
          FinalM41.^2+FinalM42.^2+FinalM43.^2+FinalM44.^2;%MM的范数
MM_Trace = FinalM11 + FinalM22 + FinalM33 + FinalM44; %MM的迹
P_vec = sqrt(FinalM21.^2 + FinalM31.^2 + FinalM41.^2); 
D_vec = sqrt(FinalM12.^2 + FinalM13.^2 + FinalM14.^2);
P_dot_D = FinalM12 .* FinalM21 + FinalM13 .* FinalM31 + FinalM14 .* FinalM41;

toc;
disp('MM参数计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：其他参数
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_other == 1
tic;


%退偏参数
LDOP =  (FinalM21 + FinalM22) ./ (FinalM11 + FinalM12); %纯线退偏


%四条棱的方位角
alpha_P = atan2(FinalM31, FinalM21) / 2;
alpha_D = atan2(FinalM13, FinalM12) / 2;
alpha_r = atan2(-FinalM24, FinalM34) / 2;
alpha_q = atan2(FinalM42, -FinalM43) / 2;


%cos角
alpha_DP_cos = acos((FinalM11.*FinalM21 + FinalM13.*FinalM31) ./ (sqrt(FinalM12.^2 + FinalM13.^2).*sqrt(FinalM21.^2 + FinalM31.^2)));
alpha_rq_cos = acos((FinalM24.*FinalM42 + FinalM34.*FinalM43) ./ (sqrt(FinalM24.^2 + FinalM34.^2).*sqrt(FinalM42.^2 + FinalM43.^2)));


%sin角
alpha_DP_sin = asin((FinalM12 .* FinalM31 - FinalM13 .* FinalM21) ./ (sqrt(FinalM21.^2 + FinalM31.^2) .* sqrt(FinalM12.^2 + FinalM13.^2)));
alpha_rq_sin = asin((FinalM24 .* FinalM43 - FinalM34 .* FinalM42) ./ (sqrt(FinalM24.^2 + FinalM34.^2) .* sqrt(FinalM42.^2 + FinalM43.^2)));

toc;
disp('其他参数计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：MMPD参数(Lu-Chipman极化分解）
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMPD == 1
tic;

[wid,len] = size(FinalM11);
MMPD_D = zeros(wid,len);
MMPD_DELTA = zeros(wid,len);
MMPD_delta = zeros(wid,len);
MMPD_R = zeros(wid,len);
MMPD_psi = zeros(wid,len);
MMPD_theta = zeros(wid,len);
M = zeros(4,4);

for i = 1:wid
    for j = 1:len
        M(1,1) = FinalM11(i,j);
        M(1,2) = FinalM12(i,j);
        M(1,3) = FinalM13(i,j);
        M(1,4) = FinalM14(i,j);
        M(2,1) = FinalM21(i,j);
        M(2,2) = FinalM22(i,j);
        M(2,3) = FinalM23(i,j);
        M(2,4) = FinalM24(i,j);
        M(3,1) = FinalM31(i,j);
        M(3,2) = FinalM32(i,j);
        M(3,3) = FinalM33(i,j);
        M(3,4) = FinalM34(i,j);
        M(4,1) = FinalM41(i,j);
        M(4,2) = FinalM42(i,j);
        M(4,3) = FinalM43(i,j);
        M(4,4) = FinalM44(i,j);
        D = sqrt(M(1,2)^2 + M(1,3)^2 + M(1,4)^2) / M(1,1); % D = [0,1]

        if D > 1
            M(1,2) = M(1,2) / D;
            M(1,3) = M(1,3) / D;
            M(1,4) = M(1,4) / D;
            D = 1; % 避免影响后续计算
        end

        D_vector = [M(1,2),M(1,3),M(1,4)]';
        m_D = sqrt(1-D^2)*eye(3) + (1-sqrt(1-D^2))*(D_vector*D_vector');
        M_D = [1 D_vector';D_vector m_D];

        if det(M_D) ~= 0 %M_D的行列式为零不可逆
            M_plus = M / M_D;
            %M_plus = M * inv(M_D); % M / M_D 速度更快，准确率更高
            m_plus = M_plus(2:4,2:4);
            temp_m_plus = m_plus * m_plus';
            lamda = eig(temp_m_plus); %返回一个列向量，包含temp_m_plus的特征值
            if det(m_plus) ~= 0
                m_DELTA = (temp_m_plus + (sqrt(lamda(1)*lamda(2))+sqrt(lamda(2)*lamda(3))+sqrt(lamda(1)*lamda(3)))*eye(3)) \...
                        ((sqrt(lamda(1)) + sqrt(lamda(2)) + sqrt(lamda(3))) * temp_m_plus + sqrt(lamda(1)*lamda(2)*lamda(3))*eye(3)) * sign(det(m_plus)); %% inv改为\
                DELTA = 1 - abs(trace(m_DELTA)) / 3; % 退偏相关 deita = [0,1]
                m_R = m_DELTA \ m_plus; 
                % m_R = inv(m_delta) * m_plus;
                R = acos((trace(m_R)+1)/2 - 1); %总的相位延迟
                delta = acos(sqrt((m_R(1,1) + m_R(2,2))^2 + (m_R(2,1) - m_R(1,2))^2) - 1); % 线性相位延迟
                psi = 0.5*atan((m_R(2,1) - m_R(1,2)) / (m_R(1,1) - m_R(2,2)));
                % psi = 0.5*atan2((m_R(2,1)-m_R(1,2)),(m_R(1,1)-m_R(2,2)));
                delta = real(delta);
                psi = real(psi);
                m_psi(1,1) = cos(2*psi);
                m_psi(1,2) = sin(2*psi);
                m_psi(1,3) = 0;
                m_psi(2,1) = -sin(2*psi);
                m_psi(2,2) = cos(2*psi);
                m_psi(2,3) = 0;
                m_psi(3,1) = 0;
                m_psi(3,2) = 0;
                m_psi(3,3) = 1;
                m_LR = m_R / m_psi;
                r1 = 1 / (2 * sin(delta)) * (m_LR(2,3) - m_LR(3,2));
                r2 = 1 / (2 * sin(delta)) * (m_LR(3,1) - m_LR(1,3));
                r1 = real(r1);
                r2 = real(r2);
                theta = (0.5 * atan2(r2,r1) * 180/pi);
            else
                [V,DD] = eig(temp_m_plus);
                U_T = inv(sqrt(DD)) * inv(V) * m_plus;
                U = U_T';
                m_DELTA = sign(det(m_plus)) * (sqrt(DD(1,1)) * V(:,1) * V(:,1)' + sqrt(DD(2,2)) * V(:,2) * V(:,2)' + sqrt(DD(3,3)) * V(:,3) * V(:,3)');
                m_R = sign(det(m_plus)) * (V(:,1) * U(:,1)' + V(:,2) * U(:,2)' + V(:,3) * U(:,3)');
                delta = 1 - abs(trace(m_DELTA)) / 3;
                R = real(acos((1 + trace(m_R)) / 2 - 1));
                ita = acos(sqrt((m_R(1,1) + m_R(2,2))^2 + (m_R(2,1) - m_R(1,2))^2) - 1);
                psi = 0.5 * atan((m_R(2,1) - m_R(1,2)) / (m_R(1,1) - m_R(2,2)));
                m_psi(1,1) = cos(2 * psi);
                m_psi(1,2) = sin(2 * psi);
                m_psi(1,3) = 0;
                m_psi(2,1) =  - sin(2 * psi);
                m_psi(2,2) = cos(2 * psi);
                m_psi(2,3) = 0;
                m_psi(3,1) = 0;
                m_psi(3,2) = 0;
                m_psi(3,3) = 1;
                m_LR = m_R / m_psi;
                r1 = 1 / (2 * sin(delta)) * (m_LR(2,3) - m_LR(3,2));
                r2 = 1 / (2 * sin(delta)) * (m_LR(3,1) - m_LR(1,3));
                r1 = real(r1);
                r2 = real(r2);
                theta = (0.5 * atan2(r2,r1) * 180 / pi);
            end

        else
            P_vector = [M(2,1),M(3,1),M(4,1)]' / M(1,1);
            P_module = norm(P_vector);
            P_unit = P_vector / P_module;
            DELTA = 1 - P_module;
            R_vector = cross(P_unit,D_vector) / norm(cross(P_unit,D_vector)) * acos(dot(P_unit,D_vector));
            R = norm(R_vector);
        end
        MMPD_D(i,j) = D;% 二向色性
        MMPD_DELTA(i,j) = DELTA;% 退偏
        MMPD_delta(i,j) = delta;% 相位延迟角
        MMPD_psi(i,j) = psi;% 旋光角
        MMPD_theta(i,j) = theta;% 快轴
        MMPD_R(i,j) = R; % 总相位延迟
     
    end
end
toc;
disp('MMPD参数计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：角度参数
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_angle == 1
tic;

%物理意义未知
angle_1 = atand((FinalM24+FinalM42)/(FinalM34+FinalM43));  
angle_2 = atand((FinalM12-FinalM21)/(FinalM13-FinalM31)); 
angle_3 = atand((FinalM12/FinalM22)/(FinalM13/FinalM33));  
angle_4 = atand((FinalM42/FinalM22)/(-FinalM43/FinalM33));

toc;
disp('WJC角度参数计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：MMCD参数（Cloude Decomposition Parameter)
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMCD == 1
tic;

% CalibratedM11 = FinalM11 .* CalibratedM11;
% CalibratedM12 = FinalM12 .* CalibratedM11;
% CalibratedM13 = FinalM13 .* CalibratedM11;
% CalibratedM14 = FinalM14 .* CalibratedM11;
% CalibratedM21 = FinalM21 .* CalibratedM11;
% CalibratedM22 = FinalM22 .* CalibratedM11;
% CalibratedM23 = FinalM23 .* CalibratedM11;
% CalibratedM24 = FinalM24 .* CalibratedM11;
% CalibratedM31 = FinalM31 .* CalibratedM11;
% CalibratedM32 = FinalM32 .* CalibratedM11;
% CalibratedM33 = FinalM33 .* CalibratedM11;
% CalibratedM34 = FinalM34 .* CalibratedM11;
% CalibratedM41 = FinalM41 .* CalibratedM11;
% CalibratedM42 = FinalM42 .* CalibratedM11;
% CalibratedM43 = FinalM43 .* CalibratedM11;
% CalibratedM44 = FinalM44 .* CalibratedM11;
% 
% H11 = (CalibratedM11 + CalibratedM12 + CalibratedM21 + CalibratedM22) / 4;
% H12 = (CalibratedM13 + CalibratedM23 - 1i * (CalibratedM14 + CalibratedM24)) / 4;
% H13 = (CalibratedM31 + CalibratedM32 - 1i * (CalibratedM41 + CalibratedM42)) / 4;
% H14 = (CalibratedM33 - CalibratedM44 - 1i * (CalibratedM34 + CalibratedM43)) / 4;
% H21 = (CalibratedM13 + CalibratedM23 + 1i * (CalibratedM14 + CalibratedM24)) / 4;
% H22 = (CalibratedM11 - CalibratedM12 + CalibratedM21 - CalibratedM22) / 4;
% H23 = (CalibratedM33 + CalibratedM44 + 1i * (CalibratedM34 - CalibratedM43)) / 4;
% H24 = (CalibratedM31 - CalibratedM32 - 1i * (CalibratedM41 - CalibratedM42))  / 4;
% H31 = (CalibratedM31 + CalibratedM32 + 1i * (CalibratedM41 + CalibratedM42)) / 4;
% H32 = (CalibratedM33 + CalibratedM44 - 1i * (CalibratedM34 - CalibratedM43)) / 4;
% H33 = (CalibratedM11 + CalibratedM12 - CalibratedM21 - CalibratedM22) / 4;
% H34 = (CalibratedM13 - CalibratedM23 - 1i * (CalibratedM14 - CalibratedM24)) / 4;
% H41 = (CalibratedM33 - CalibratedM44 + 1i * (CalibratedM34 + CalibratedM43)) / 4;
% H42 = (CalibratedM31 - CalibratedM32 + 1i * (CalibratedM41 - CalibratedM42)) / 4;
% H43 = (CalibratedM13 - CalibratedM23 + 1i * (CalibratedM14 - CalibratedM24)) / 4;
% H44 = (CalibratedM11 - CalibratedM12 - CalibratedM21 + CalibratedM22) / 4;

H11 = (FinalM11 + FinalM12 + FinalM21 + FinalM22) / 4;
H12 = (FinalM13 + FinalM23 + 1i * (FinalM14 + FinalM24)) / 4;
H13 = (FinalM31 + FinalM32 + 1i * (FinalM41 + FinalM42)) / 4;
H14 = (FinalM33 - FinalM44 + 1i * (FinalM34 + FinalM43)) / 4;
H21 = (FinalM13 + FinalM23 - 1i * (FinalM14 + FinalM24)) / 4;
H22 = (FinalM11 - FinalM12 + FinalM21 - FinalM22) / 4;
H23 = (FinalM33 + FinalM44 - 1i * (FinalM34 - FinalM43)) / 4;
H24 = (FinalM31 - FinalM32 + 1i * (FinalM41 - FinalM42))  / 4;
H31 = (FinalM31 + FinalM32 - 1i * (FinalM41 + FinalM42)) / 4;
H32 = (FinalM33 + FinalM44 + 1i * (FinalM34 - FinalM43)) / 4;
H33 = (FinalM11 + FinalM12 - FinalM21 - FinalM22) / 4;
H34 = (FinalM13 - FinalM23 + 1i * (FinalM14 - FinalM24)) / 4;
H41 = (FinalM33 - FinalM44 - 1i * (FinalM34 + FinalM43)) / 4;
H42 = (FinalM31 - FinalM32 - 1i * (FinalM41 - FinalM42)) / 4;
H43 = (FinalM13 - FinalM23 - 1i * (FinalM14 - FinalM24)) / 4;
H44 = (FinalM11 - FinalM12 - FinalM21 + FinalM22) / 4;

% H11 = (FinalM11 + FinalM12 + FinalM21 + FinalM22) / 4;
% H12 = (FinalM13 + FinalM23 + 1i * (FinalM14 + FinalM24)) / 4;
% H13 = (FinalM31 + FinalM32 - 1i * (FinalM41 + FinalM42)) / 4;
% H14 = (FinalM33 + FinalM44 + 1i * (FinalM34 - FinalM43)) / 4;
% H21 = (FinalM13 + FinalM23 - 1i * (FinalM14 + FinalM24)) / 4;
% H22 = (FinalM11 - FinalM12 + FinalM21 - FinalM22) / 4;
% H23 = (FinalM33 - FinalM44 - 1i * (FinalM34 + FinalM43)) / 4;
% H24 = (FinalM31 - FinalM32 - 1i * (FinalM41 - FinalM42))  / 4;
% H31 = (FinalM31 + FinalM32 + 1i * (FinalM41 + FinalM42)) / 4;
% H32 = (FinalM33 - FinalM44 + 1i * (FinalM34 + FinalM43)) / 4;
% H33 = (FinalM11 + FinalM12 - FinalM21 - FinalM22) / 4;
% H34 = (FinalM13 - FinalM23 + 1i * (FinalM14 - FinalM24)) / 4;
% H41 = (FinalM33 + FinalM44 - 1i * (FinalM34 - FinalM43)) / 4;
% H42 = (FinalM31 - FinalM32 + 1i * (FinalM41 - FinalM42)) / 4;
% H43 = (FinalM13 - FinalM23 - 1i * (FinalM14 - FinalM24)) / 4;
% H44 = (FinalM11 - FinalM12 - FinalM21 + FinalM22) / 4;

[wid,len] = size(FinalM11);
MMCD_lambdas = zeros(wid,len,4);

for i = 1:wid
    for j = 1:len
        hh = [H11(i,j),H12(i,j),H13(i,j),H14(i,j);...
            H21(i,j),H22(i,j),H23(i,j),H24(i,j);...
            H31(i,j),H32(i,j),H33(i,j),H34(i,j);...
            H41(i,j),H42(i,j),H43(i,j),H44(i,j)];
        MMCD_lambdas(i,j,:) = real(sort(eig(hh),'descend'));
    end
end

% MMCD的本征值
MMCD_lambda1 = MMCD_lambdas(:,:,1);
MMCD_lambda2 = MMCD_lambdas(:,:,2);
MMCD_lambda3 = MMCD_lambdas(:,:,3);
MMCD_lambda4 = MMCD_lambdas(:,:,4);
%-------------------------------------------------------------------------%
% 包含退偏的关键信息，任意无退偏矩阵的 lambdas = [1,0,0,0]; 当样品有退偏的
% 时候，会得到多个非零本征值，完全退偏时有 lambdas = [1,1,1,1] 去其他情况下
% lambdas介于[0,1]之间。
%-------------------------------------------------------------------------%

% 偏振纯度指标
MMCD_P1 = (MMCD_lambda1 - MMCD_lambda2) ./ CalibratedM11;
MMCD_P2 = (MMCD_lambda1 + MMCD_lambda2 - 2*MMCD_lambda3) ./ CalibratedM11 ;
MMCD_P3 = (MMCD_lambda1 + MMCD_lambda2 + MMCD_lambda3 - 3*MMCD_lambda4) ./ CalibratedM11 ;

% 整体纯度指标
MMCD_PI = sqrt((MMCD_P1.^2 + MMCD_P2.^2 + MMCD_P3.^2) / 3);

% 退偏指标
MMCD_PD = sqrt((2.*MMCD_P1.^2 + 2/3.*MMCD_P2.^2 + 1/3.*MMCD_P3.^2) / 3);

% 退偏熵
MMCD_S = -((MMCD_lambda1 .* log(MMCD_lambda1)/log(4)) + (MMCD_lambda2 .* log(MMCD_lambda2)/log(4)) + (MMCD_lambda3 .* log(MMCD_lambda3)/log(4)) + (MMCD_lambda4 .* log(MMCD_lambda4)/log(4)));
MMCD_S = real(MMCD_S);

toc;
disp('MMCD参数计算完成！');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：MMLD参数（Logarithm Decomposition Parameter)
% 作者 ：周旭
% 修改时间 ：2022.6.27
% 要求 ：缪勒矩阵的行列式需为正值，避免本征值出现虚部。如果行列式为负，则对数分解法失效，应考虑
%        其他分解算法。

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMLD == 1
tic;

MM = zeros([size(FinalMM,[1 2]),4,4]);
MM(:,:,1,1) = FinalM11;
MM(:,:,1,2) = FinalM12;
MM(:,:,1,3) = FinalM13;
MM(:,:,1,4) = FinalM14;
MM(:,:,2,1) = FinalM21;
MM(:,:,2,2) = FinalM22;
MM(:,:,2,3) = FinalM23;
MM(:,:,2,4) = FinalM24;
MM(:,:,3,1) = FinalM31;
MM(:,:,3,2) = FinalM32;
MM(:,:,3,3) = FinalM33;
MM(:,:,3,4) = FinalM34;
MM(:,:,4,1) = FinalM41;
MM(:,:,4,2) = FinalM42;
MM(:,:,4,3) = FinalM43;
MM(:,:,4,4) = FinalM44;

[wid,len] = size(FinalM11,[1 2]);
A = zeros([wid,len,4,4]);
for i = 1:4
    for j = 1:4
        temp = MM(:,:,i,j);
        A(:,:,i,j) = temp(1:end,1:end);
    end
end

L = zeros([wid,len,4,4]);
temp = zeros(1,1,4,4);
temp(1,1,:,:) = diag([1,-1,-1,-1]);
G = zeros([wid,len,4,4]) + temp;
for i = 1:wid
    for j = 1:len
        L(i,j,:,:) = logm(squeeze(A(i,j,:,:)));
    end
end

T = zeros([wid,len,4,4]);
L2 = permute(L,[1,2,4,3]);
for i = 1:wid
    for j = len
        T(i,j,:,:) = squeeze(G(i,j,:,:)) * squeeze(L2(i,j,:,:)) * squeeze(G(i,j,:,:));
    end
end

MMLD_Lm = (L - T) / 2; 
MMLD_Lu = (L + T) / 2;
temp = zeros(1,1,4,4);
temp(1,1,:,:) = eye(4);
MMLD_Lu = MMLD_Lu - MMLD_Lu(:,:,1,1) .* (zeros([wid,len,4,4]) + temp);

% Lm 纯偏振参量。随厚度的变化近似位线性规律
MMLD_D = sqrt(MMLD_Lm(:,:,1,2).^2 + MMLD_Lm(:,:,1,3).^2);
MMLD_delta = sqrt(MMLD_Lm(:,:,2,4).^2 + MMLD_Lm(:,:,3,4).^2);
MMLD_alpha = MMLD_Lm(:,:,2,3) / 2; % 圆双折射
MMLD_CD = MMLD_Lm(:,:,1,4); % 圆二向色性

% Lu 退偏参量。随厚度变化近似位二次函数规律
% a22代表0-90°线退偏 ；a33代表±45°线退偏 ； a44代表圆退偏。正常情况下取值为负，负值越大退偏效应越强
MMLD_a22 = MMLD_Lu(:,:,2,2);
MMLD_a33 = MMLD_Lu(:,:,3,3);
MMLD_a44 = MMLD_Lu(:,:,4,4);

MMLD_aL = (MMLD_Lu(:,:,2,2) + MMLD_Lu(:,:,3,3)) / 2;
MMLD_aLA = sqrt((MMLD_Lu(:,:,2,2) - MMLD_Lu(:,:,3,3)).^2 + (MMLD_Lu(:,:,2,3) + MMLD_Lu(:,:,3,2)).^2) / 2;


MMLD_D = imresize(real(MMLD_D),[wid len],'nearest');
MMLD_delta = imresize(real(MMLD_delta),[wid len],'nearest');
MMLD_alpha = imresize(real(MMLD_alpha),[wid len],'nearest');
MMLD_CD = imresize(real(MMLD_CD),[wid len],'nearest');
MMLD_a22 = imresize(real(MMLD_a22),[wid len],'nearest');
MMLD_a33 = imresize(real(MMLD_a33),[wid len],'nearest');
MMLD_aL = imresize(real(MMLD_aL),[wid len],'nearest');
MMLD_a44 = imresize(real(MMLD_a44),[wid len],'nearest');
MMLD_aLA = imresize(real(MMLD_aLA),[wid len],'nearest');

toc;
disp('MMLD参数计算完成！');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 功能 ：数据保存与显示
% 作者 ：周旭
% 修改时间 ：2022.6.27

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
if func_save == 1

if func_MMT == 1

figure(1);
imagesc(MMT_t1,[tsprctile(MMT_t1(:),2),tsprctile(MMT_t1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t1');
saveas(figure(1),strcat(fileset,'MMT_t1.tif'));
save([fileset,'MMT_t1.mat'],'MMT_t1');

figure(2);
imagesc(MMT_b,[tsprctile(MMT_b(:),2),tsprctile(MMT_b(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ b');
saveas(figure(2),strcat(fileset,'MMT_b.tif'));
save([fileset,'MMT_b.mat'],'MMT_b');

figure(3);
imagesc(MMT_A,[tsprctile(MMT_A(:),2),tsprctile(MMT_A(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ A');
saveas(figure(3),strcat(fileset,'MMT_A.tif'));
save([fileset,'MMT_A.mat'],'MMT_A');

figure(4);
imagesc(MMT_b2,[tsprctile(MMT_b2(:),2),tsprctile(MMT_b2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ b2');
saveas(figure(4),strcat(fileset,'MMT_b2.tif'));
save([fileset,'MMT_b2.mat'],'MMT_b2');

figure(5);
imagesc(MMT_beta,[tsprctile(MMT_beta(:),2),tsprctile(MMT_beta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ beta');
saveas(figure(5),strcat(fileset,'MMT_beta.tif'));
save([fileset,'MMT_beta.mat'],'MMT_beta');

figure(6);
imagesc(Bhls,[tsprctile(Bhls(:),2),tsprctile(Bhls(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('Bhls');
saveas(figure(6),strcat(fileset,'Bhls.tif'));
save([fileset,'Bhls.mat'],'Bhls');

figure(7);
imagesc(Bfs,[tsprctile(Bfs(:),2),tsprctile(Bfs(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('Bfs');
saveas(figure(7),strcat(fileset,'Bfs.tif'));
save([fileset,'Bfs.mat'],'Bfs');

figure(8);
imagesc(cda,[tsprctile(cda(:),2),tsprctile(cda(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('cda');
saveas(figure(8),strcat(fileset,'cda.tif'));
save([fileset,'cda.mat'],'cda');

figure(9);
imagesc(MMT_t_4243,[tsprctile(MMT_t_4243(:),2),tsprctile(MMT_t_4243(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 4243');
saveas(figure(9),strcat(fileset,'MMT_t_4243.tif'));
save([fileset,'MMT_t_4243.mat'],'MMT_t_4243');

figure(10);
imagesc(MMT_t_2434,[tsprctile(MMT_t_2434(:),2),tsprctile(MMT_t_2434(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 2434');
saveas(figure(10),strcat(fileset,'MMT_t_2434.tif'));
save([fileset,'MMT_t_2434.mat'],'MMT_t_2434');

figure(11);
imagesc(MMT_t_1213,[tsprctile(MMT_t_1213(:),2),tsprctile(MMT_t_1213(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 1213');
saveas(figure(11),strcat(fileset,'MMT_t_1213.tif'));
save([fileset,'MMT_t_1213.mat'],'MMT_t_1213');

figure(12);
imagesc(MMT_t_2131,[tsprctile(MMT_t_2131(:),2),tsprctile(MMT_t_2131(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 2131');
saveas(figure(12),strcat(fileset,'MMT_t_2131.tif'));
save([fileset,'MMT_t_2131.mat'],'MMT_t_2131');

figure(13);
imagesc(PDxcheng,[tsprctile(PDxcheng(:),2),tsprctile(PDxcheng(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PDxcheng');
saveas(figure(13),strcat(fileset,'PDxcheng.tif'));
save([fileset,'PDxcheng.mat'],'PDxcheng');

figure(14);
imagesc(rqxcheng,[tsprctile(rqxcheng(:),2),tsprctile(rqxcheng(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rqxcheng');
saveas(figure(14),strcat(fileset,'rqxcheng.tif'));
save([fileset,'rqxcheng.mat'],'rqxcheng');

figure(15);
imagesc(MMTPD,[tsprctile(MMTPD(:),2),tsprctile(MMTPD(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMTPD');
saveas(figure(15),strcat(fileset,'MMTPD.tif'));
save([fileset,'MMTPD.mat'],'MMTPD');

figure(16);
imagesc(MMTrq,[tsprctile(MMTrq(:),2),tsprctile(MMTrq(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMTrq');
saveas(figure(16),strcat(fileset,'MMTrq.tif'));
save([fileset,'MMTrq.mat'],'MMTrq');

figure(17);
imagesc(PLsubDL,[tsprctile(PLsubDL(:),2),tsprctile(PLsubDL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubDL');
saveas(figure(17),strcat(fileset,'PLsubDL.tif'));
save([fileset,'PLsubDL.mat'],'PLsubDL');

figure(18);
imagesc(PLsubDLunited,[tsprctile(PLsubDLunited(:),2),tsprctile(PLsubDLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubDLunited');
saveas(figure(18),strcat(fileset,'PLsubDLunited.tif'));
save([fileset,'PLsubDLunited.mat'],'PLsubDLunited');

figure(19);
imagesc(rLsubqL,[tsprctile(rLsubqL(:),2),tsprctile(rLsubqL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rLsubqL');
saveas(figure(19),strcat(fileset,'rLsubqL.tif'));
save([fileset,'rLsubqL.mat'],'rLsubqL');

figure(20);
imagesc(rLsubqLunited,[tsprctile(rLsubqLunited(:),2),tsprctile(rLsubqLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rLsubqLunited');
saveas(figure(20),strcat(fileset,'rLsubqLunited.tif'));
save([fileset,'rLsubqLunited.mat'],'rLsubqLunited');

figure(21);
imagesc(PLsubrL,[tsprctile(PLsubrL(:),2),tsprctile(PLsubrL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubrL');
saveas(figure(21),strcat(fileset,'PLsubrL.tif'));
save([fileset,'PLsubrL.mat'],'PLsubrL');

figure(22);
imagesc(PLsubrLunited,[tsprctile(PLsubrLunited(:),2),tsprctile(PLsubrLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubrLunited');
saveas(figure(22),strcat(fileset,'PLsubrLunited.tif'));
save([fileset,'PLsubrLunited.mat'],'PLsubrLunited');

figure(23);
imagesc(DLsubqL,[tsprctile(DLsubqL(:),2),tsprctile(DLsubqL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('DLsubqL');
saveas(figure(23),strcat(fileset,'DLsubqL.tif'));
save([fileset,'DLsubqL.mat'],'DLsubqL');

figure(24);
imagesc(DLsubqLunited,[tsprctile(DLsubqLunited(:),2),tsprctile(DLsubqLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('DLsubqLunited');
saveas(figure(24),strcat(fileset,'DLsubqLunited.tif'));
save([fileset,'DLsubqLunited.mat'],'DLsubqLunited');

figure(25);
imagesc(MMT_phi2233,[tsprctile(MMT_phi2233(:),2),tsprctile(MMT_phi2233(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi2233');
saveas(figure(25),strcat(fileset,'MMT_phi2233.tif'));
save([fileset,'MMT_phi2233.mat'],'MMT_phi2233');

figure(26);
imagesc(MMT_phi3121,[tsprctile(MMT_phi3121(:),2),tsprctile(MMT_phi3121(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi3121');
saveas(figure(26),strcat(fileset,'MMT_phi3121.tif'));
save([fileset,'MMT_phi3121.mat'],'MMT_phi3121');

figure(27);
imagesc(MMT_phi1312,[tsprctile(MMT_phi1312(:),2),tsprctile(MMT_phi1312(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi1312');
saveas(figure(27),strcat(fileset,'MMT_phi1312.tif'));
save([fileset,'MMT_phi1312.mat'],'MMT_phi1312');

figure(28);
imagesc(MMT_phi4243,[tsprctile(MMT_phi4243(:),2),tsprctile(MMT_phi4243(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi4243');
saveas(figure(28),strcat(fileset,'MMT_phi4243.tif'));
save([fileset,'MMT_phi4243.mat'],'MMT_phi4243');

figure(29);
imagesc(MMT_phi2434,[tsprctile(MMT_phi2434(:),2),tsprctile(MMT_phi2434(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi2434');
saveas(figure(29),strcat(fileset,'MMT_phi2434.tif'));
save([fileset,'MMT_phi2434.mat'],'MMT_phi2434');

figure(30);
imagesc(PDslcha,[tsprctile(PDslcha(:),2),tsprctile(PDslcha(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PDslcha');
saveas(figure(30),strcat(fileset,'PDslcha.tif'));
save([fileset,'PDslcha.mat'],'PDslcha');

figure(31);
imagesc(rqslcha,[tsprctile(rqslcha(:),2),tsprctile(rqslcha(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rqslcha');
saveas(figure(31),strcat(fileset,'rqslcha.tif'));
save([fileset,'rqslcha.mat'],'rqslcha');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_equ == 1

figure(32);
imagesc(MMT_P1,[tsprctile(MMT_P1(:),2),tsprctile(MMT_P1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P1');
saveas(figure(32),strcat(fileset,'MMT_P1.tif'));
save([fileset,'MMT_P1.mat'],'MMT_P1');

figure(33);
imagesc(MMT_P2,[tsprctile(MMT_P2(:),2),tsprctile(MMT_P2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P2');
saveas(figure(33),strcat(fileset,'MMT_P2.tif'));
save([fileset,'MMT_P2.mat'],'MMT_P2');

figure(34);
imagesc(MMT_P3,[tsprctile(MMT_P3(:),2),tsprctile(MMT_P3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P3');
saveas(figure(34),strcat(fileset,'MMT_P3.tif'));
save([fileset,'MMT_P3.mat'],'MMT_P3');

figure(35);
imagesc(MMT_P4,[tsprctile(MMT_P4(:),2),tsprctile(MMT_P4(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P4');
saveas(figure(35),strcat(fileset,'MMT_P4.tif'));
save([fileset,'MMT_P4.mat'],'MMT_P4');

figure(36);
imagesc(MMT_P5,[tsprctile(MMT_P5(:),2),tsprctile(MMT_P5(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P5');
saveas(figure(36),strcat(fileset,'MMT_P5.tif'));
save([fileset,'MMT_P5.mat'],'MMT_P5');

figure(37);
imagesc(MMT_P6,[tsprctile(MMT_P6(:),2),tsprctile(MMT_P6(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P6');
saveas(figure(37),strcat(fileset,'MMT_P6.tif'));
save([fileset,'MMT_P6.mat'],'MMT_P6');

figure(38);
imagesc(MMT_P7,[tsprctile(MMT_P7(:),2),tsprctile(MMT_P7(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P7');
saveas(figure(38),strcat(fileset,'MMT_P7.tif'));
save([fileset,'MMT_P7.mat'],'MMT_P7');

figure(39);
imagesc(MMT_P8,[tsprctile(MMT_P8(:),2),tsprctile(MMT_P8(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P8');
saveas(figure(39),strcat(fileset,'MMT_P8.tif'));
save([fileset,'MMT_P8.mat'],'MMT_P8');

figure(40);
imagesc(MMT_P9,[tsprctile(MMT_P9(:),2),tsprctile(MMT_P9(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P9');
saveas(figure(40),strcat(fileset,'MMT_P9.tif'));
save([fileset,'MMT_P9.mat'],'MMT_P9');

figure(41);
imagesc(MMT_P10,[tsprctile(MMT_P10(:),2),tsprctile(MMT_P10(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P10');
saveas(figure(41),strcat(fileset,'MMT_P10.tif'));
save([fileset,'MMT_P10.mat'],'MMT_P10');

figure(42);
imagesc(MMT_P11,[tsprctile(MMT_P11(:),2),tsprctile(MMT_P11(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P11');
saveas(figure(42),strcat(fileset,'MMT_P11.tif'));
save([fileset,'MMT_P11.mat'],'MMT_P11');

figure(43);
imagesc(Es,[tsprctile(Es(:),2),tsprctile(Es(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('Es');
saveas(figure(43),strcat(fileset,'Es.tif'));
save([fileset,'Es.mat'],'Es');

figure(44);
imagesc(E1,[tsprctile(E1(:),2),tsprctile(E1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E1');
saveas(figure(44),strcat(fileset,'E1.tif'));
save([fileset,'E1.mat'],'E1');

figure(45);
imagesc(E2,[tsprctile(E2(:),2),tsprctile(E2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E2');
saveas(figure(45),strcat(fileset,'E2.tif'));
save([fileset,'E2.mat'],'E2');

figure(46);
imagesc(E3,[tsprctile(E3(:),2),tsprctile(E3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E3');
saveas(figure(46),strcat(fileset,'E3.tif'));
save([fileset,'E3.mat'],'E3');

figure(47);
imagesc(E4,[tsprctile(E4(:),2),tsprctile(E4(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E4');
saveas(figure(47),strcat(fileset,'E4.tif'));
save([fileset,'E4.mat'],'E4');

figure(48);
imagesc(E5,[tsprctile(E5(:),2),tsprctile(E5(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E5');
saveas(figure(48),strcat(fileset,'E5.tif'));
save([fileset,'E5.mat'],'E5');

figure(49);
imagesc(E6,[tsprctile(E6(:),2),tsprctile(E6(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E6');
saveas(figure(49),strcat(fileset,'E6.tif'));
save([fileset,'E6.mat'],'E6');

figure(50);
imagesc(E7,[tsprctile(E7(:),2),tsprctile(E7(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E7');
saveas(figure(50),strcat(fileset,'E7.tif'));
save([fileset,'E7.mat'],'E7');

figure(51);
imagesc(E8,[tsprctile(E8(:),2),tsprctile(E8(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E8');
saveas(figure(51),strcat(fileset,'E8.tif'));
save([fileset,'E8.mat'],'E8');

figure(52);
imagesc(E9,[tsprctile(E9(:),2),tsprctile(E9(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E9');
saveas(figure(52),strcat(fileset,'E9.tif'));
save([fileset,'E9.mat'],'E9');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_abrio == 1

figure(53);
imagesc(temp_1,[tsprctile(temp_1(:),2),tsprctile(temp_1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('temp_ 1');
saveas(figure(53),strcat(fileset,'temp_1.tif'));
save([fileset,'temp_1.mat'],'temp_1');

figure(54);
imagesc(temp_2,[tsprctile(temp_2(:),2),tsprctile(temp_2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('temp_ 2');
saveas(figure(54),strcat(fileset,'temp_2.tif'));
save([fileset,'temp_2.mat'],'temp_2');

figure(55);
imagesc(MMT_Abrio_R,[tsprctile(MMT_Abrio_R(:),2),tsprctile(MMT_Abrio_R(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ Abrio_ R');
saveas(figure(55),strcat(fileset,'MMT_Abrio_R.tif'));
save([fileset,'MMT_Abrio_R.mat'],'MMT_Abrio_R');

figure(56);
imagesc(MMT_Abrio_theta,[tsprctile(MMT_Abrio_theta(:),2),tsprctile(MMT_Abrio_theta(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ Abrio_ theta');
saveas(figure(56),strcat(fileset,'MMT_Abrio_theta.tif'));
save([fileset,'MMT_Abrio_theta.mat'],'MMT_Abrio_theta');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MM == 1

figure(57);
imagesc(MM_Det,[tsprctile(MM_Det(:),2),tsprctile(MM_Det(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Det');
saveas(figure(57),strcat(fileset,'MM_Det.tif'));
save([fileset,'MM_Det.mat'],'MM_Det');

figure(58);
imagesc(MM_Norm,[tsprctile(MM_Norm(:),2),tsprctile(MM_Norm(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Norm');
saveas(figure(58),strcat(fileset,'MM_Norm.tif'));
save([fileset,'MM_Norm.mat'],'MM_Norm');

figure(59);
imagesc(MM_Trace,[tsprctile(MM_Trace(:),2),tsprctile(MM_Trace(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Trace');
saveas(figure(59),strcat(fileset,'MM_Trace.tif'));
save([fileset,'MM_Trace.mat'],'MM_Trace');

figure(60);
imagesc(P_vec,[tsprctile(P_vec(:),2),tsprctile(P_vec(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('P_ vec');
saveas(figure(60),strcat(fileset,'P_vec.tif'));
save([fileset,'P_vec.mat'],'P_vec');

figure(61);
imagesc(D_vec,[tsprctile(D_vec(:),2),tsprctile(D_vec(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('D_ vec');
saveas(figure(61),strcat(fileset,'D_vec.tif'));
save([fileset,'D_vec.mat'],'D_vec');

figure(62);
imagesc(P_dot_D,[tsprctile(P_dot_D(:),2),tsprctile(P_dot_D(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('P_ dot_ D');
saveas(figure(62),strcat(fileset,'P_dot_D.tif'));
save([fileset,'P_dot_D.mat'],'P_dot_D');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_other == 1

figure(63);
imagesc(LDOP,[tsprctile(LDOP(:),2),tsprctile(LDOP(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('LDOP');
saveas(figure(63),strcat(fileset,'LDOP.tif'));
save([fileset,'LDOP.mat'],'LDOP');

figure(64);
imagesc(alpha_P,[tsprctile(alpha_P(:),2),tsprctile(alpha_P(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ P');
saveas(figure(64),strcat(fileset,'alpha_P.tif'));
save([fileset,'alpha_P.mat'],'alpha_P');

figure(65);
imagesc(alpha_D,[tsprctile(alpha_D(:),2),tsprctile(alpha_D(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ D');
saveas(figure(65),strcat(fileset,'alpha_D.tif'));
save([fileset,'alpha_D.mat'],'alpha_D');

figure(66);
imagesc(alpha_r,[tsprctile(alpha_r(:),2),tsprctile(alpha_r(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ r');
saveas(figure(66),strcat(fileset,'alpha_r.tif'));
save([fileset,'alpha_r.mat'],'alpha_r');

figure(67);
imagesc(alpha_q,[tsprctile(alpha_q(:),2),tsprctile(alpha_q(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ q');
saveas(figure(67),strcat(fileset,'alpha_q.tif'));
save([fileset,'alpha_q.mat'],'alpha_q');

% figure(68);
% imagesc(alpha_DP_cos,[tsprctile(alpha_DP_cos(:),2),tsprctile(alpha_DP_cos(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ DP_ cos');
% saveas(figure(68),strcat(fileset,'alpha_DP_cos.tif'));
% save([fileset,'alpha_DP_cos.mat'],'alpha_DP_cos');

% figure(69);
% imagesc(alpha_rq_cos,[tsprctile(alpha_rq_cos(:),2),tsprctile(alpha_rq_cos(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ rq_ cos');
% saveas(figure(69),strcat(fileset,'alpha_rq_cos.tif'));
% save([fileset,'alpha_rq_cos.mat'],'alpha_rq_cos');
% 
% figure(70);
% imagesc(alpha_DP_sin,[tsprctile(alpha_DP_sin(:),2),tsprctile(alpha_DP_sin(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ DP_ sin');
% saveas(figure(70),strcat(fileset,'alpha_DP_sin.tif'));
% save([fileset,'alpha_DP_sin.mat'],'alpha_DP_sin');
% 
% figure(71);
% imagesc(alpha_rq_sin,[tsprctile(alpha_rq_sin(:),2),tsprctile(alpha_rq_sin(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ rq_ sin');
% saveas(figure(71),strcat(fileset,'alpha_rq_sin.tif'));
% save([fileset,'alpha_rq_sin.mat'],'alpha_rq_sin');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMPD == 1

figure(72);
imagesc(MMPD_D,[tsprctile(MMPD_D(:),2),tsprctile(MMPD_D(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ D');
saveas(figure(72),strcat(fileset,'MMPD_D.tif'));
save([fileset,'MMPD_D.mat'],'MMPD_D');

figure(73);
imagesc(MMPD_DELTA,[tsprctile(MMPD_DELTA(:),2),tsprctile(MMPD_DELTA(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ DELTA');
saveas(figure(73),strcat(fileset,'MMPD_DELTA.tif'));
save([fileset,'MMPD_DELTA.mat'],'MMPD_DELTA');

figure(74);
imagesc(MMPD_delta,[tsprctile(MMPD_delta(:),2),tsprctile(MMPD_delta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ delta');
saveas(figure(74),strcat(fileset,'MMPD_delta.tif'));
save([fileset,'MMPD_delta.mat'],'MMPD_delta');

figure(75);
imagesc(MMPD_psi,[tsprctile(MMPD_psi(:),2),tsprctile(MMPD_psi(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ psi');
saveas(figure(75),strcat(fileset,'MMPD_psi.tif'));
save([fileset,'MMPD_psi.mat'],'MMPD_psi');

figure(76);
imagesc(MMPD_theta,[tsprctile(MMPD_theta(:),2),tsprctile(MMPD_theta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ theta');
saveas(figure(76),strcat(fileset,'MMPD_theta.tif'));
save([fileset,'MMPD_theta.mat'],'MMPD_theta');

figure(77);
imagesc(MMPD_R,[tsprctile(MMPD_R(:),2),tsprctile(MMPD_R(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ R');
saveas(figure(77),strcat(fileset,'MMPD_R.tif'));
save([fileset,'MMPD_R.mat'],'MMPD_R');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_angle == 1
% 
% figure(78);
% imagesc(angle_1,[tsprctile(angle_1(:),2),tsprctile(angle_1(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 1');
% saveas(figure(78),strcat(fileset,'angle_1.tif'));
% save([fileset,'angle_1.mat'],'angle_1');
% 
% figure(79);
% imagesc(angle_2,[tsprctile(angle_2(:),2),tsprctile(angle_2(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 2');
% saveas(figure(79),strcat(fileset,'angle_2.tif'));
% save([fileset,'angle_2.mat'],'angle_2');
% 
% figure(80);
% imagesc(angle_3,[tsprctile(angle_3(:),2),tsprctile(angle_3(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 3');
% saveas(figure(80),strcat(fileset,'angle_3.tif'));
% save([fileset,'angle_3.mat'],'angle_3');
% 
% figure(81);
% imagesc(angle_4,[tsprctile(angle_4(:),2),tsprctile(angle_4(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 4');
% saveas(figure(81),strcat(fileset,'angle_4.tif'));
% save([fileset,'angle_4.mat'],'angle_4');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if func_MMCD == 1

figure(82);
imagesc(MMCD_lambda1,[tsprctile(MMCD_lambda1(:),2),tsprctile(MMCD_lambda1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda1');
saveas(figure(82),strcat(fileset,'MMCD_lambda1.tif'));
save([fileset,'MMCD_lambda1.mat'],'MMCD_lambda1');

figure(83);
imagesc(MMCD_lambda2,[tsprctile(MMCD_lambda2(:),2),tsprctile(MMCD_lambda2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda2');
saveas(figure(83),strcat(fileset,'MMCD_lambda2.tif'));
save([fileset,'MMCD_lambda2.mat'],'MMCD_lambda2');

figure(84);
imagesc(MMCD_lambda3,[tsprctile(MMCD_lambda3(:),2),tsprctile(MMCD_lambda3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda3');
saveas(figure(84),strcat(fileset,'MMCD_lambda3.tif'));
save([fileset,'MMCD_lambda3.mat'],'MMCD_lambda3');

figure(85);
imagesc(MMCD_lambda4,[tsprctile(MMCD_lambda4(:),2),tsprctile(MMCD_lambda4(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda4');
saveas(figure(85),strcat(fileset,'MMCD_lambda4.tif'));
save([fileset,'MMCD_lambda4.mat'],'MMCD_lambda4');

figure(86);
imagesc(MMCD_P1,[tsprctile(MMCD_P1(:),2),tsprctile(MMCD_P1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P1');
saveas(figure(86),strcat(fileset,'MMCD_P1.tif'));
save([fileset,'MMCD_P1.mat'],'MMCD_P1');

figure(87);
imagesc(MMCD_P2,[tsprctile(MMCD_P2(:),2),tsprctile(MMCD_P2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P2');
saveas(figure(87),strcat(fileset,'MMCD_P2.tif'));
save([fileset,'MMCD_P2.mat'],'MMCD_P2');

figure(88);
imagesc(MMCD_P3,[tsprctile(MMCD_P3(:),2),tsprctile(MMCD_P3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P3');
saveas(figure(88),strcat(fileset,'MMCD_P3.tif'));
save([fileset,'MMCD_P3.mat'],'MMCD_P3');

figure(89);
imagesc(MMCD_PI,[tsprctile(MMCD_PI(:),2),tsprctile(MMCD_PI(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ PI');
saveas(figure(89),strcat(fileset,'MMCD_PI.tif'));
save([fileset,'MMCD_PI.mat'],'MMCD_PI');

figure(90);
imagesc(MMCD_PD,[tsprctile(MMCD_PD(:),2),tsprctile(MMCD_PD(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ PD');
saveas(figure(90),strcat(fileset,'MMCD_PD.tif'));
save([fileset,'MMCD_PD.mat'],'MMCD_PD');

figure(91);
imagesc(MMCD_S,[tsprctile(MMCD_S(:),2),tsprctile(MMCD_S(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ S');
saveas(figure(91),strcat(fileset,'MMCD_S.tif'));
save([fileset,'MMCD_S.mat'],'MMCD_S');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMLD == 1

figure(92);
imagesc(MMLD_D,[tsprctile(MMLD_D(:),2),tsprctile(MMLD_D(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ D');
saveas(figure(92),strcat(fileset,'MMLD_D.tif'));
save([fileset,'MMLD_D.mat'],'MMLD_D');

figure(93);
imagesc(MMLD_delta,[tsprctile(MMLD_delta(:),2),tsprctile(MMLD_delta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ delta');
saveas(figure(93),strcat(fileset,'MMLD_delta.tif'));
save([fileset,'MMLD_delta.mat'],'MMLD_delta');

figure(94);
imagesc(MMLD_alpha,[tsprctile(MMLD_alpha(:),2),tsprctile(MMLD_alpha(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ alpha');
saveas(figure(94),strcat(fileset,'MMLD_alpha.tif'));
save([fileset,'MMLD_alpha.mat'],'MMLD_alpha');

figure(95);
imagesc(MMLD_CD,[tsprctile(MMLD_CD(:),2),tsprctile(MMLD_CD(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ CD');
saveas(figure(95),strcat(fileset,'MMLD_CD.tif'));
save([fileset,'MMLD_CD.mat'],'MMLD_CD');

figure(96);
imagesc(MMLD_a22,[tsprctile(MMLD_a22(:),2),tsprctile(MMLD_a22(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ a22');
saveas(figure(96),strcat(fileset,'MMLD_a22.tif'));
save([fileset,'MMLD_a22.mat'],'MMLD_a22');

figure(97);
imagesc(MMLD_a33,[tsprctile(MMLD_a33(:),2),tsprctile(MMLD_a33(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ a33');
saveas(figure(97),strcat(fileset,'MMLD_a33.tif'));
save([fileset,'MMLD_a33.mat'],'MMLD_a33');

figure(98);
imagesc(MMLD_aL,[tsprctile(MMLD_aL(:),2),tsprctile(MMLD_aL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ aL');
saveas(figure(98),strcat(fileset,'MMLD_aL.tif'));
save([fileset,'MMLD_aL.mat'],'MMLD_aL');

figure(99);
imagesc(MMLD_a44,[tsprctile(MMLD_a44(:),2),tsprctile(MMLD_a44(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ a44');
saveas(figure(99),strcat(fileset,'MMLD_a44.tif'));
save([fileset,'MMLD_a44.mat'],'MMLD_a44');

figure(100);
imagesc(MMLD_aLA,[tsprctile(MMLD_aLA(:),2),tsprctile(MMLD_aLA(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ aLA');
saveas(figure(100),strcat(fileset,'MMLD_aLA.tif'));
save([fileset,'MMLD_aLA.mat'],'MMLD_aLA');

toc;
disp('图像保存完成！');
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_showreal == 1

if func_MMT == 1
figure(1);
imagesc(MMT_t1);   colormap('jet'); colorbar; title('MMT_ t1');

figure(2);
imagesc(MMT_b);  colormap('jet'); colorbar; title('MMT_ b');

figure(3);
imagesc(MMT_A); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ A');

figure(4);
imagesc(MMT_b2); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ b2');

figure(5);
imagesc(MMT_beta); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ beta');

figure(6);
imagesc(Bhls); axis equal; axis off; colormap('jet'); colorbar; title('Bhls');
saveas(figure(6),strcat(fileset,'Bhls_real.tif'));

figure(7);
imagesc(Bfs); axis equal; axis off; colormap('jet'); colorbar; title('Bfs');
saveas(figure(7),strcat(fileset,'Bfs_real.tif'));

figure(8);
imagesc(cda); axis equal; axis off; colormap('jet'); colorbar; title('cda');
saveas(figure(8),strcat(fileset,'cda_real.tif'));

figure(9);
imagesc(MMT_t_4243); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 4243');
saveas(figure(9),strcat(fileset,'MMT_t_4243_real.tif'));

figure(10);
imagesc(MMT_t_2434); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 2434');
saveas(figure(10),strcat(fileset,'MMT_t_2434_real.tif'));

figure(11);
imagesc(MMT_t_1213); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 1213');
saveas(figure(11),strcat(fileset,'MMT_t_1213_real.tif'));

figure(12);
imagesc(MMT_t_2131); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 2131');
saveas(figure(12),strcat(fileset,'MMT_t_2131_real.tif'));

figure(13);
imagesc(PDxcheng); axis equal; axis off; colormap('jet'); colorbar; title('PDxcheng');
saveas(figure(13),strcat(fileset,'PDxcheng_real.tif'));

figure(14);
imagesc(rqxcheng); axis equal; axis off; colormap('jet'); colorbar; title('rqxcheng');
saveas(figure(14),strcat(fileset,'rqxcheng_real.tif'));

figure(15);
imagesc(MMTPD); axis equal; axis off; colormap('jet'); colorbar; title('MMTPD');
saveas(figure(15),strcat(fileset,'MMTPD_real.tif'));

figure(16);
imagesc(MMTrq); axis equal; axis off; colormap('jet'); colorbar; title('MMTrq');
saveas(figure(16),strcat(fileset,'MMTrq_real.tif'));

figure(17);
imagesc(PLsubDL); axis equal; axis off; colormap('jet'); colorbar; title('PLsubDL');
saveas(figure(17),strcat(fileset,'PLsubDL_real.tif'));

figure(18);
imagesc(PLsubDLunited); axis equal; axis off; colormap('jet'); colorbar; title('PLsubDLunited');

figure(19);
imagesc(rLsubqL); axis equal; axis off; colormap('jet'); colorbar; title('rLsubqL');
saveas(figure(19),strcat(fileset,'rLsubqL_real.tif'));

figure(20);
imagesc(rLsubqLunited); axis equal; axis off; colormap('jet'); colorbar; title('rLsubqLunited');

figure(21);
imagesc(PLsubrL); axis equal; axis off; colormap('jet'); colorbar; title('PLsubrL');
saveas(figure(21),strcat(fileset,'PLsubrL_real.tif'));

figure(22);
imagesc(PLsubrLunited); axis equal; axis off; colormap('jet'); colorbar; title('PLsubrLunited');

figure(23);
imagesc(DLsubqL); axis equal; axis off; colormap('jet'); colorbar; title('DLsubqL');
saveas(figure(23),strcat(fileset,'DLsubqL_real.tif'));

figure(24);
imagesc(DLsubqLunited); axis equal; axis off; colormap('jet'); colorbar; title('DLsubqLunited');

figure(25);
imagesc(MMT_phi2233); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi2233');

figure(26);
imagesc(MMT_phi3121); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi3121');

figure(27);
imagesc(MMT_phi1312); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi1312');

figure(28);
imagesc(MMT_phi4243); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi4243');

figure(29);
imagesc(MMT_phi2434); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi2434');

figure(30);
imagesc(PDslcha); axis equal; axis off; colormap('jet'); colorbar; title('PDslcha');
saveas(figure(30),strcat(fileset,'PDslcha_real.tif'));

figure(31);
imagesc(rqslcha); axis equal; axis off; colormap('jet'); colorbar; title('rqslcha');
saveas(figure(31),strcat(fileset,'rqslcha_real.tif'));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_equ == 1

figure(32);
imagesc(MMT_P1); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P1');

figure(33);
imagesc(MMT_P2); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P2');

figure(34);
imagesc(MMT_P3); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P3');

figure(35);
imagesc(MMT_P4); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P4');

figure(36);
imagesc(MMT_P5); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P5');

figure(37);
imagesc(MMT_P6); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P6');

figure(38);
imagesc(MMT_P7); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P7');

figure(39);
imagesc(MMT_P8); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P8');

figure(40);
imagesc(MMT_P9); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P9');

figure(41);
imagesc(MMT_P10); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P10');

figure(42);
imagesc(MMT_P11); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P11');

figure(43);
imagesc(Es); axis equal; axis off; colormap('jet'); colorbar; title('Es');

figure(44);
imagesc(E1); axis equal; axis off; colormap('jet'); colorbar; title('E1');

figure(45);
imagesc(E2); axis equal; axis off; colormap('jet'); colorbar; title('E2');

figure(46);
imagesc(E3); axis equal; axis off; colormap('jet'); colorbar; title('E3');

figure(47);
imagesc(E4); axis equal; axis off; colormap('jet'); colorbar; title('E4');

figure(48);
imagesc(E5); axis equal; axis off; colormap('jet'); colorbar; title('E5');

figure(49);
imagesc(E6); axis equal; axis off; colormap('jet'); colorbar; title('E6');

figure(50);
imagesc(E7); axis equal; axis off; colormap('jet'); colorbar; title('E7');

figure(51);
imagesc(E8); axis equal; axis off; colormap('jet'); colorbar; title('E8');

figure(52);
imagesc(E9); axis equal; axis off; colormap('jet'); colorbar; title('E9');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_abrio == 1

figure(53);
imagesc(temp_1); axis equal; axis off; colormap('jet'); colorbar; title('temp_ 1');

figure(54);
imagesc(temp_2); axis equal; axis off; colormap('jet'); colorbar; title('temp_ 2');

figure(55);
imagesc(MMT_Abrio_R); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ Abrio_ R');

figure(56);
imagesc(MMT_Abrio_theta); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ Abrio_ theta');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MM == 1

figure(57);
imagesc(MM_Det); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Det');
saveas(figure(57),strcat(fileset,'MM_Det_real.tif'));

figure(58);
imagesc(MM_Norm); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Norm');
saveas(figure(58),strcat(fileset,'MM_Norm_real.tif'));

figure(59);
imagesc(MM_Trace); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Trace');
saveas(figure(59),strcat(fileset,'MM_Trace_real.tif'));

figure(60);
imagesc(P_vec); axis equal; axis off; colormap('jet'); colorbar; title('P_ vec');
saveas(figure(60),strcat(fileset,'P_vec_real.tif'));

figure(61);
imagesc(D_vec); axis equal; axis off; colormap('jet'); colorbar; title('D_ vec');
saveas(figure(61),strcat(fileset,'D_vec_real.tif'));

figure(62);
imagesc(P_dot_D); axis equal; axis off; colormap('jet'); colorbar; title('P_ dot_ D');
saveas(figure(62),strcat(fileset,'P_dot_D_real.tif'));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_other == 1

figure(63);
imagesc(LDOP); axis equal; axis off; colormap('jet'); colorbar; title('LDOP');

figure(64);
imagesc(alpha_P); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ P');

figure(65);
imagesc(alpha_D); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ D');

figure(66);
imagesc(alpha_r); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ r');

figure(67);
imagesc(alpha_q); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ q');

% figure(68);
% imagesc(alpha_DP_cos,[tsprctile(alpha_DP_cos(:),2),tsprctile(alpha_DP_cos(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ DP_ cos');
% saveas(figure(68),strcat(fileset,'alpha_DP_cos.tif'));
% save([fileset,'alpha_DP_cos.mat'],'alpha_DP_cos');

figure(69);
imagesc(alpha_rq_cos); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ rq_ cos');

figure(70);
imagesc(alpha_DP_sin); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ DP_ sin');

figure(71);
imagesc(alpha_rq_sin); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ rq_ sin');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMPD == 1

figure(72);
imagesc(MMPD_D); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ D');
saveas(figure(72),strcat(fileset,'MMPD_D_real.tif'));

figure(73);
imagesc(MMPD_DELTA); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ DELTA');
saveas(figure(73),strcat(fileset,'MMPD_DELTA_real.tif'));

figure(74);
imagesc(MMPD_delta); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ delta');
saveas(figure(74),strcat(fileset,'MMPD_delta_real.tif'));

figure(75);
imagesc(MMPD_psi); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ psi');
saveas(figure(75),strcat(fileset,'MMPD_psi_real.tif'));

figure(76);
imagesc(MMPD_theta); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ theta');
saveas(figure(76),strcat(fileset,'MMPD_theta_real.tif'));

figure(77);
imagesc(MMPD_R); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ R');
saveas(figure(77),strcat(fileset,'MMPD_R_real.tif'));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_angle == 1

figure(78);
imagesc(angle_1); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 1');

figure(79);
imagesc(angle_2); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 2');

figure(80);
imagesc(angle_3); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 3');

figure(81);
imagesc(angle_4); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 4');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if func_MMCD == 1

figure(82);
imagesc(MMCD_lambda1); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda1');
saveas(figure(82),strcat(fileset,'MMCD_lambda1_real.tif'));

figure(83);
imagesc(MMCD_lambda2); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda2');
saveas(figure(83),strcat(fileset,'MMCD_lambda2_real.tif'));

figure(84);
imagesc(MMCD_lambda3); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda3');
saveas(figure(84),strcat(fileset,'MMCD_lambda3_real.tif'));

figure(85);
imagesc(MMCD_lambda4); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda4');
saveas(figure(85),strcat(fileset,'MMCD_lambda4_real.tif'));

figure(86);
imagesc(MMCD_P1); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P1');
saveas(figure(86),strcat(fileset,'MMCD_P1_real.tif'));

figure(87);
imagesc(MMCD_P2); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P2');
saveas(figure(87),strcat(fileset,'MMCD_P2_real.tif'));

figure(88);
imagesc(MMCD_P3); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P3');
saveas(figure(88),strcat(fileset,'MMCD_P3_real.tif'));

figure(89);
imagesc(MMCD_PI); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ PI');
saveas(figure(89),strcat(fileset,'MMCD_PI_real.tif'));

figure(90);
imagesc(MMCD_PD); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ PD');
saveas(figure(90),strcat(fileset,'MMCD_PD_real.tif'));
figure(91);
imagesc(MMCD_S); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ S');
saveas(figure(91),strcat(fileset,'MMCD_S_real.tif'));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMLD == 1

figure(92);
imagesc(MMLD_D); axis equal; axis off; colormap('jet');  title('MMLD_ D');
saveas(figure(92),strcat(fileset,'MMLD_D_real.tif'));
figure(93);
imagesc(MMLD_delta); axis equal; axis off; colormap('jet');  title('MMLD_ delta');
saveas(figure(93),strcat(fileset,'MMLD_delta_real.tif'));
figure(94);
imagesc(MMLD_alpha); axis equal; axis off; colormap('jet'); title('MMLD_ alpha');
saveas(figure(94),strcat(fileset,'MMLD_alpha_real.tif'));
figure(95);
imagesc(MMLD_CD); axis equal; axis off; colormap('jet');  title('MMLD_ CD');
saveas(figure(95),strcat(fileset,'MMLD_CD_real.tif'));
figure(96);
imagesc(MMLD_a22); axis equal; axis off; colormap('jet');  title('MMLD_ a22');
saveas(figure(96),strcat(fileset,'MMLD_a22_real.tif'));
figure(97);
imagesc(MMLD_a33); axis equal; axis off; colormap('jet');  title('MMLD_ a33');
saveas(figure(97),strcat(fileset,'MMLD_aL_real.tif'));
figure(98);
imagesc(MMLD_aL); axis equal; axis off; colormap('jet');  title('MMLD_ aL');
saveas(figure(98),strcat(fileset,'MMLD_aL_real.tif'));
figure(99);
imagesc(MMLD_a44); axis equal; axis off; colormap('jet');  title('MMLD_ a44');
saveas(figure(99),strcat(fileset,'MMLD_a44_real.tif'));

figure(100);
imagesc(MMLD_aLA); axis equal; axis off; colormap('jet');  title('MMLD_ aLA');
saveas(figure(100),strcat(fileset,'MMLD_aLA_real.tif'));
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if func_MMatrix == 1

CalibratedM11 = FinalM11 .* CalibratedM11;
CalibratedM12 = FinalM12 .* CalibratedM11;
CalibratedM13 = FinalM13 .* CalibratedM11;
CalibratedM14 = FinalM14 .* CalibratedM11;
CalibratedM21 = FinalM21 .* CalibratedM11;
CalibratedM22 = FinalM22 .* CalibratedM11;
CalibratedM23 = FinalM23 .* CalibratedM11;
CalibratedM24 = FinalM24 .* CalibratedM11;
CalibratedM31 = FinalM31 .* CalibratedM11;
CalibratedM32 = FinalM32 .* CalibratedM11;
CalibratedM33 = FinalM33 .* CalibratedM11;
CalibratedM34 = FinalM34 .* CalibratedM11;
CalibratedM41 = FinalM41 .* CalibratedM11;
CalibratedM42 = FinalM42 .* CalibratedM11;
CalibratedM43 = FinalM43 .* CalibratedM11;
CalibratedM44 = FinalM44 .* CalibratedM11;

figure(1);
imagesc(CalibratedM11); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM11');

figure(2);
imagesc(CalibratedM12); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM12');

figure(3);
imagesc(CalibratedM13); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM13');

figure(4);
imagesc(CalibratedM14); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM14');

figure(5);
imagesc(CalibratedM21); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM21');

figure(6);
imagesc(CalibratedM22); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM22');

figure(7);
imagesc(CalibratedM23); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM23');

figure(8);
imagesc(CalibratedM24); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM24');

figure(9);
imagesc(CalibratedM31); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM31');

figure(10);
imagesc(CalibratedM32); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM32');

figure(11);
imagesc(CalibratedM33); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM33');

figure(12);
imagesc(CalibratedM34); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM34');

figure(13);
imagesc(CalibratedM41); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM41');

figure(14);
imagesc(CalibratedM42); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM42');

figure(15);
imagesc(CalibratedM43); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM43');

figure(16);
imagesc(CalibratedM44); axis equal; axis off; colormap(jet); colorbar; title('CalibratedM44');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%








