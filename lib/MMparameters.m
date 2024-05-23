%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Function: fast calculate all Mueller matrix parameters and make PBPs groups.
%  Author: Zhou Xu.
%  Date: 2023.2.8
%  Output: Mueller matrix parameters and PBPs groups.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Function: loadmat FinalMM and select function.
%  Author: Zhou Xu
%  Date: 2023.2.8
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
fclose all; clear; close all;

% loadmat Mueller matrix
tic;
fileset = ["C:\Users\Administrator\Desktop\新建文件夹 (2)\2\"];

for k = 1:size(fileset,1)
    tic;
    filem11=[fileset(k,1)+"\m11.mat"];
    filename=[fileset(k,1)+"\FinalMM.mat"];
    filem11=char(filem11);
    filename=char(filename);
    filepath=char(fileset(k,1)+"\pbps\");
    disp(filepath);
    load(filem11);
    load(filename);
    [H_image,W_image] = size(FinalM11);
    mkdir(char(filepath))
    % select function.
    func_MMT   =    1;
    func_MMPD  =    1;
    func_abrio =    1;
    func_equ   =    1;
    func_other =    1;
    func_MM    =    1;
    func_angle =    1;
    func_MMCD  =    1;
    func_MMLD  =    0;
    func_pic   =    1;
    func_save  =    1;
    func_PBP   =    0;

    FinalMM(:,:,1) = ones(H_image,W_image);
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

%%
%  Function: calculate MMT parameters

    if func_MMT == 1
        tic;
        MMT_t1 = sqrt((FinalM22 - FinalM33).^2+(FinalM23 + FinalM32).^2)/2;
        MMT_b = (FinalM22 + FinalM33)/2; 
        MMT_A = (2.*MMT_b.*MMT_t1) ./ (MMT_b.^2+MMT_t1.^2); 
        MMT_b2 = 1 - MMT_b; 
        MMT_beta = abs(FinalM23 - FinalM32) ./ 2; 
        Bhls = (FinalM22.*FinalM33) - (FinalM23.*FinalM32); 
        Bfs = sqrt(FinalM22.^2 + FinalM33.^2 + FinalM23.^2 + FinalM32.^2); 
        cda = FinalM14+FinalM41;
        MMT_b_tld = (FinalM22 - FinalM33) / 2 ;
        MMT_beta_tld = (FinalM23 + FinalM32) / 2; 
        MMT_t_4243 = sqrt(FinalM42.^2+FinalM43.^2);
        MMT_t_2434 = sqrt(FinalM24.^2+FinalM34.^2);
        MMT_t_1213 = sqrt(FinalM12.^2+FinalM13.^2);
        MMT_t_2131 = sqrt(FinalM21.^2+FinalM31.^2);
        PDxcheng = (FinalM12.*FinalM31 - FinalM13.*FinalM21) ./ (MMT_t_1213 .* MMT_t_2131);
        rqxcheng = (FinalM24.*FinalM43 - FinalM34.*FinalM42) ./ (MMT_t_2434 .* MMT_t_4243);
        MMTPD = sqrt((FinalM12-FinalM21).^2+(FinalM13-FinalM31).^2);
        MMTrq = sqrt((FinalM24+FinalM42).^2+(FinalM34+FinalM43).^2);
        PLsubDL = MMT_t_2131 - MMT_t_1213;
        PLsubDLunited = PLsubDL ./ (MMT_t_2131 + MMT_t_1213);
        rLsubqL = MMT_t_2434-MMT_t_4243;
        rLsubqLunited = PLsubDL ./ (MMT_t_2434 + MMT_t_4243);
        PLsubrL = MMT_t_2131-MMT_t_2434;
        PLsubrLunited = PLsubDL ./ (MMT_t_2131 + MMT_t_2434);
        DLsubqL = MMT_t_1213 - MMT_t_4243;
        DLsubqLunited = PLsubDL ./ (MMT_t_1213 + MMT_t_4243);
        MMT_phi2233 = 0.25*(atan2((FinalM23+FinalM32),(FinalM22-FinalM33)))*180/pi;
        MMT_phi3121 = 0.5*atan2((FinalM31),(FinalM21))*180/pi;
        MMT_phi1312 = 0.5*atan2((FinalM13),(FinalM12))*180/pi;
        MMT_phi4243 = 0.5*atan2((FinalM42),(-FinalM43))*180/pi;
        MMT_phi2434 = 0.5*atan2((-FinalM24),(FinalM34))*180/pi;
        PDslcha = sqrt(MMT_t_2131.^2+MMT_t_1213.^2-2*MMT_t_2131.*MMT_t_1213.*cos(rad2deg((MMT_phi3121-MMT_phi1312))));
        rqslcha = sqrt(MMT_t_2434.^2+MMT_t_4243.^2-2*MMT_t_2434.*MMT_t_4243.*cos(rad2deg((MMT_phi2434-MMT_phi4243))));
        disp('MMT calculation completed');
        toc;
     end
 
%%
%  Function: calculate equal parameters   

    if func_equ == 1
        tic;
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
        disp('equal calculation completed');
        toc;
    end

%%
% Function: calculate abrio parameters
    
    if func_abrio == 1
        tic;
        temp_1 = (FinalM12-FinalM42) ./ (FinalM44-FinalM14);
        temp_2 = (FinalM43-FinalM13) ./ (FinalM44-FinalM14);
        MMT_Abrio_R = atan(sqrt(temp_1.^2 + temp_2.^2))*180/pi;
        MMT_Abrio_theta = 1/2*atan2(temp_1,temp_2)*180/pi;
        disp('abrio calculation completed')
        toc;
    end

%%
% Function: calculate MM parameters

    if func_MM == 1
        MM_Det = zeros(H_image,W_image);
        for i = 1:H_image
            for j = 1:W_image
                mm = [FinalM11(i,j),FinalM12(i,j),FinalM13(i,j),FinalM14(i,j);...
                    FinalM21(i,j),FinalM22(i,j),FinalM23(i,j),FinalM24(i,j);...
                    FinalM31(i,j),FinalM32(i,j),FinalM33(i,j),FinalM34(i,j);...
                    FinalM41(i,j),FinalM42(i,j),FinalM43(i,j),FinalM44(i,j);];
                MM_Det(i,j) = det(mm); 
            end
        end
        MM_Norm = FinalM11.^2+FinalM12.^2+FinalM13.^2+FinalM14.^2+...
                  FinalM21.^2+FinalM22.^2+FinalM23.^2+FinalM24.^2+...
                  FinalM31.^2+FinalM32.^2+FinalM33.^2+FinalM34.^2+...
                  FinalM41.^2+FinalM42.^2+FinalM43.^2+FinalM44.^2;
        MM_Trace = FinalM11 + FinalM22 + FinalM33 + FinalM44; 
        P_vec = sqrt(FinalM21.^2 + FinalM31.^2 + FinalM41.^2); 
        D_vec = sqrt(FinalM12.^2 + FinalM13.^2 + FinalM14.^2);
        P_dot_D = FinalM12 .* FinalM21 + FinalM13 .* FinalM31 + FinalM14 .* FinalM41;
        disp('MM calculation completed');
        toc;
    end

%%
% Function: calculate other parameters

    if func_other == 1
        tic;
        LDOP =  (FinalM21 + FinalM22) ./ (FinalM11 + FinalM12); 
        alpha_P = atan2(FinalM31, FinalM21) / 2;
        alpha_D = atan2(FinalM13, FinalM12) / 2;
        alpha_r = atan2(-FinalM24, FinalM34) / 2;
        alpha_q = atan2(FinalM42, -FinalM43) / 2;
        alpha_DP_cos = acos((FinalM11.*FinalM21 + FinalM13.*FinalM31) ./ (sqrt(FinalM12.^2 + FinalM13.^2).*sqrt(FinalM21.^2 + FinalM31.^2)));
        alpha_rq_cos = acos((FinalM24.*FinalM42 + FinalM34.*FinalM43) ./ (sqrt(FinalM24.^2 + FinalM34.^2).*sqrt(FinalM42.^2 + FinalM43.^2)));       
        alpha_DP_sin = asin((FinalM12 .* FinalM31 - FinalM13 .* FinalM21) ./ (sqrt(FinalM21.^2 + FinalM31.^2) .* sqrt(FinalM12.^2 + FinalM13.^2)));
        alpha_rq_sin = asin((FinalM24 .* FinalM43 - FinalM34 .* FinalM42) ./ (sqrt(FinalM24.^2 + FinalM34.^2) .* sqrt(FinalM42.^2 + FinalM43.^2)));
        disp('other calculation completed');
        toc;
    end     

%%
% Function: calculate MMPD parameters

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
                D = sqrt(M(1,2)^2 + M(1,3)^2 + M(1,4)^2) / M(1,1); 
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
        disp('MMPD calculation completed');
        toc;
    end

%%
% Function: calculate angle parameters
    
    if func_angle == 1
        tic;
        angle_1 = atand((FinalM24+FinalM42)/(FinalM34+FinalM43));  
        angle_2 = atand((FinalM12-FinalM21)/(FinalM13-FinalM31)); 
        angle_3 = atand((FinalM12/FinalM22)/(FinalM13/FinalM33));  
        angle_4 = atand((FinalM42/FinalM22)/(-FinalM43/FinalM33));
        disp('angle calculation completed');
        toc;
    end    

%%
% Function: calculate MMCD parameters

    if func_MMCD == 1
        tic;
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
        MMCD_lambda1 = MMCD_lambdas(:,:,1);
        MMCD_lambda2 = MMCD_lambdas(:,:,2);
        MMCD_lambda3 = MMCD_lambdas(:,:,3);
        MMCD_lambda4 = MMCD_lambdas(:,:,4);
        MMCD_P1 = (MMCD_lambda1 - MMCD_lambda2) ./ FinalM11;
        MMCD_P2 = (MMCD_lambda1 + MMCD_lambda2 - 2*MMCD_lambda3) ./ FinalM11 ;
        MMCD_P3 = (MMCD_lambda1 + MMCD_lambda2 + MMCD_lambda3 - 3*MMCD_lambda4) ./ FinalM11 ;
        MMCD_PI = sqrt((MMCD_P1.^2 + MMCD_P2.^2 + MMCD_P3.^2) / 3);
        MMCD_PD = sqrt((2.*MMCD_P1.^2 + 2/3.*MMCD_P2.^2 + 1/3.*MMCD_P3.^2) / 3);
        MMCD_S = -((MMCD_lambda1 .* log(MMCD_lambda1)/log(4)) + (MMCD_lambda2 .* log(MMCD_lambda2)/log(4)) + (MMCD_lambda3 .* log(MMCD_lambda3)/log(4)) + (MMCD_lambda4 .* log(MMCD_lambda4)/log(4)));
        MMCD_S = real(MMCD_S);
        disp('MMCD calculation completed');
        toc;
    end

%%
% Function: calculate MMLD parameters

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
        MMLD_D = sqrt(MMLD_Lm(:,:,1,2).^2 + MMLD_Lm(:,:,1,3).^2);
        MMLD_delta = sqrt(MMLD_Lm(:,:,2,4).^2 + MMLD_Lm(:,:,3,4).^2);
        MMLD_alpha = MMLD_Lm(:,:,2,3) / 2; % 圆双折射
        MMLD_CD = MMLD_Lm(:,:,1,4); % 圆二向色性
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
        disp('MMLD calculation completed');
        toc;
    end

%%
% Function: save patameters
            save([filepath,'FinalM12.mat'],'FinalM12');
            save([filepath,'FinalM13.mat'],'FinalM13');
            save([filepath,'FinalM14.mat'],'FinalM14');
            save([filepath,'FinalM21.mat'],'FinalM21');
            save([filepath,'FinalM22.mat'],'FinalM22');
            save([filepath,'FinalM23.mat'],'FinalM23');
            save([filepath,'FinalM24.mat'],'FinalM24');
            save([filepath,'FinalM31.mat'],'FinalM31');
            save([filepath,'FinalM32.mat'],'FinalM32');
            save([filepath,'FinalM33.mat'],'FinalM33');
            save([filepath,'FinalM34.mat'],'FinalM34');
            save([filepath,'FinalM41.mat'],'FinalM41');
            save([filepath,'FinalM42.mat'],'FinalM42');
            save([filepath,'FinalM43.mat'],'FinalM43');
            save([filepath,'FinalM44.mat'],'FinalM44');

    if func_save == 1
        if func_MMT == 1
            save([filepath,'MMT_t1.mat'],'MMT_t1');
            save([filepath,'MMT_b.mat'],'MMT_b');
            save([filepath,'MMT_A.mat'],'MMT_A');
            save([filepath,'MMT_b2.mat'],'MMT_b2');
            save([filepath,'MMT_beta.mat'],'MMT_beta');
            save([filepath,'Bhls.mat'],'Bhls');
            save([filepath,'Bfs.mat'],'Bfs');
            save([filepath,'cda.mat'],'cda');
            save([filepath,'MMT_t_4243.mat'],'MMT_t_4243');
            save([filepath,'MMT_t_2434.mat'],'MMT_t_2434');
            save([filepath,'MMT_t_1213.mat'],'MMT_t_1213');
            save([filepath,'MMT_t_2131.mat'],'MMT_t_2131');
            save([filepath,'PDxcheng.mat'],'PDxcheng');
            save([filepath,'rqxcheng.mat'],'rqxcheng');
            save([filepath,'MMTrq.mat'],'MMTrq');
            save([filepath,'MMTPD.mat'],'MMTPD');
            save([filepath,'PLsubDL.mat'],'PLsubDL');
            save([filepath,'PLsubDLunited.mat'],'PLsubDLunited');
            save([filepath,'rLsubqL.mat'],'rLsubqL');
            save([filepath,'rLsubqLunited.mat'],'rLsubqLunited');
            save([filepath,'PLsubrL.mat'],'PLsubrL');
            save([filepath,'PLsubrLunited.mat'],'PLsubrLunited');
            save([filepath,'DLsubqL.mat'],'DLsubqL');
            save([filepath,'DLsubqLunited.mat'],'DLsubqLunited');
            save([filepath,'MMT_phi2233.mat'],'MMT_phi2233');
            save([filepath,'MMT_phi3121.mat'],'MMT_phi3121');
            save([filepath,'MMT_phi1312.mat'],'MMT_phi1312');
            save([filepath,'MMT_phi4243.mat'],'MMT_phi4243');
            save([filepath,'MMT_phi2434.mat'],'MMT_phi2434');
            save([filepath,'PDslcha.mat'],'PDslcha');
            save([filepath,'rqslcha.mat'],'rqslcha');
         end
    
        if func_equ == 1
            save([filepath,'MMT_P1.mat'],'MMT_P1');
            save([filepath,'MMT_P2.mat'],'MMT_P2');
            save([filepath,'MMT_P3.mat'],'MMT_P3');
            save([filepath,'MMT_P4.mat'],'MMT_P4');
            save([filepath,'MMT_P5.mat'],'MMT_P5');
            save([filepath,'MMT_P6.mat'],'MMT_P6');
            save([filepath,'MMT_P7.mat'],'MMT_P7');
            save([filepath,'MMT_P8.mat'],'MMT_P8');
            save([filepath,'MMT_P9.mat'],'MMT_P9');
            save([filepath,'MMT_P10.mat'],'MMT_P10');
            save([filepath,'MMT_P11.mat'],'MMT_P11');
            save([filepath,'MMT_P12.mat'],'MMT_P12');
            save([filepath,'Es.mat'],'Es');
            save([filepath,'E1.mat'],'E1');
            save([filepath,'E2.mat'],'E2');
            save([filepath,'E3.mat'],'E3');
            save([filepath,'E4.mat'],'E4');
            save([filepath,'E5.mat'],'E5');
            save([filepath,'E6.mat'],'E6');
            save([filepath,'E7.mat'],'E7');
            save([filepath,'E8.mat'],'E8');
            save([filepath,'E9.mat'],'E9');
        end

        if func_abrio == 1
            save([filepath,'temp_1.mat'],'temp_1');
            save([filepath,'temp_2.mat'],'temp_2');
            save([filepath,'MMT_Abrio_R.mat'],'MMT_Abrio_R');
            save([filepath,'MMT_Abrio_theta.mat'],'MMT_Abrio_theta');
        end

        if func_MM == 1
            save([filepath,'MM_Det.mat'],'MM_Det');
            save([filepath,'MM_Norm.mat'],'MM_Norm');
            save([filepath,'MM_Trace.mat'],'MM_Trace');
            save([filepath,'P_vec.mat'],'P_vec');
            save([filepath,'D_vec.mat'],'D_vec');
            save([filepath,'P_dot_D.mat'],'P_dot_D');   
        end

        if func_other == 1
            save([filepath,'LDOP.mat'],'LDOP');
            save([filepath,'alpha_P.mat'],'alpha_P');
            save([filepath,'alpha_D.mat'],'alpha_D');
            save([filepath,'alpha_r.mat'],'alpha_r');
            save([filepath,'alpha_q.mat'],'alpha_q');
            save([filepath,'alpha_DP_cos.mat'],'alpha_DP_cos');
            save([filepath,'alpha_rq_cos.mat'],'alpha_rq_cos');
            save([filepath,'alpha_DP_sin.mat'],'alpha_DP_sin');
            save([filepath,'alpha_rq_sin.mat'],'alpha_rq_sin');
        end

        if func_MMPD == 1
            save([filepath,'MMPD_D.mat'],'MMPD_D');
            save([filepath,'MMPD_DELTA.mat'],'MMPD_DELTA');
            save([filepath,'MMPD_delta.mat'],'MMPD_delta');
            save([filepath,'MMPD_psi.mat'],'MMPD_psi');
            save([filepath,'MMPD_theta.mat'],'MMPD_theta');
            save([filepath,'MMPD_R.mat'],'MMPD_R');
        end

        if func_angle == 1
            save([filepath,'angle_1.mat'],'angle_1');
            save([filepath,'angle_2.mat'],'angle_2');
            save([filepath,'angle_3.mat'],'angle_3');
            save([filepath,'angle_4.mat'],'angle_4');
        end

       if func_MMCD == 1
            save([filepath,'MMCD_lambda1.mat'],'MMCD_lambda1');
            save([filepath,'MMCD_lambda2.mat'],'MMCD_lambda2');
            save([filepath,'MMCD_lambda3.mat'],'MMCD_lambda3');
            save([filepath,'MMCD_lambda4.mat'],'MMCD_lambda4');
            save([filepath,'MMCD_P1.mat'],'MMCD_P1');
            save([filepath,'MMCD_P2.mat'],'MMCD_P2');
            save([filepath,'MMCD_P3.mat'],'MMCD_P3');
            save([filepath,'MMCD_PI.mat'],'MMCD_PI');
            save([filepath,'MMCD_PD.mat'],'MMCD_PD');
            save([filepath,'MMCD_S.mat'],'MMCD_S');
        end

        if func_MMLD == 1
            save([filepath,'MMLD_D.mat'],'MMLD_D');
            save([filepath,'MMLD_delta.mat'],'MMLD_delta');
            save([filepath,'MMLD_alpha.mat'],'MMLD_alpha');
            save([filepath,'MMLD_CD.mat'],'MMLD_CD');
            save([filepath,'MMLD_a22.mat'],'MMLD_a22');
            save([filepath,'MMLD_a33.mat'],'MMLD_a33');
            save([filepath,'MMLD_aL.mat'],'MMLD_aL');
            save([filepath,'MMLD_a44.mat'],'MMLD_a44');
            save([filepath,'MMLD_aLA.mat'],'MMLD_aLA');
        end
        disp('savemat completed');
        toc;
    end
%%
% Function: save pictures
    if func_pic == 1
        if func_MMT == 1
        tic;
        figure(1);
        imagesc(FinalM12,[tsprctile(FinalM12(:),2),tsprctile(FinalM12(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM12');
        saveas(figure(1),strcat(filepath,'FinalM12.tif'));        
        figure(2);
        imagesc(FinalM13,[tsprctile(FinalM13(:),2),tsprctile(FinalM13(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM13');
        saveas(figure(2),strcat(filepath,'FinalM13.tif'));        
        figure(3);
        imagesc(FinalM14,[tsprctile(FinalM14(:),2),tsprctile(FinalM14(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM14');
        saveas(figure(3),strcat(filepath,'FinalM14.tif'));        
        figure(4);
        imagesc(FinalM21,[tsprctile(FinalM21(:),2),tsprctile(FinalM21(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM21');
        saveas(figure(4),strcat(filepath,'FinalM21.tif'));       
        figure(5);
        imagesc(FinalM22,[tsprctile(FinalM22(:),2),tsprctile(FinalM22(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM22');
        saveas(figure(5),strcat(filepath,'FinalM22.tif'));       
        figure(6);
        imagesc(FinalM23,[tsprctile(FinalM23(:),2),tsprctile(FinalM23(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM23');
        saveas(figure(6),strcat(filepath,'FinalM23.tif'));       
        figure(7);
        imagesc(FinalM24,[tsprctile(FinalM24(:),2),tsprctile(FinalM24(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM24');
        saveas(figure(7),strcat(filepath,'FinalM24.tif'));       
        figure(8);
        imagesc(FinalM31,[tsprctile(FinalM31(:),2),tsprctile(FinalM31(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM31');
        saveas(figure(8),strcat(filepath,'FinalM31.tif'));       
        figure(9);
        imagesc(FinalM32,[tsprctile(FinalM32(:),2),tsprctile(FinalM32(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM32');
        saveas(figure(9),strcat(filepath,'FinalM32.tif'));      
        figure(10);
        imagesc(FinalM33,[tsprctile(FinalM33(:),2),tsprctile(FinalM33(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM33');
        saveas(figure(10),strcat(filepath,'FinalM33.tif'));      
        figure(11);
        imagesc(FinalM34,[tsprctile(FinalM34(:),2),tsprctile(FinalM34(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM34');
        saveas(figure(11),strcat(filepath,'FinalM34.tif'));     
        figure(12);
        imagesc(FinalM41,[tsprctile(FinalM41(:),2),tsprctile(FinalM41(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM41');
        saveas(figure(12),strcat(filepath,'FinalM41.tif'));     
        figure(13);
        imagesc(FinalM42,[tsprctile(FinalM42(:),2),tsprctile(FinalM42(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM42');
        saveas(figure(13),strcat(filepath,'FinalM42.tif'));     
        figure(14);
        imagesc(FinalM43,[tsprctile(FinalM43(:),2),tsprctile(FinalM43(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM43');
        saveas(figure(14),strcat(filepath,'FinalM43.tif'));
        figure(15);
        imagesc(FinalM44,[tsprctile(FinalM44(:),2),tsprctile(FinalM44(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('FinalM44');
        saveas(figure(15),strcat(filepath,'FinalM44.tif'));

        figure(1);
        imagesc(MMT_t1,[tsprctile(MMT_t1(:),2),tsprctile(MMT_t1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t1');
        saveas(figure(1),strcat(filepath,'MMT_t1.tif'));        
        figure(2);
        imagesc(MMT_b,[tsprctile(MMT_b(:),2),tsprctile(MMT_b(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ b');
        saveas(figure(2),strcat(filepath,'MMT_b.tif'));        
        figure(3);
        imagesc(MMT_A,[tsprctile(MMT_A(:),2),tsprctile(MMT_A(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ A');
        saveas(figure(3),strcat(filepath,'MMT_A.tif'));        
        figure(4);
        imagesc(MMT_b2,[tsprctile(MMT_b2(:),2),tsprctile(MMT_b2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ b2');
        saveas(figure(4),strcat(filepath,'MMT_b2.tif'));       
        figure(5);
        imagesc(MMT_beta,[tsprctile(MMT_beta(:),2),tsprctile(MMT_beta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ beta');
        saveas(figure(5),strcat(filepath,'MMT_beta.tif'));       
        figure(6);
        imagesc(Bhls,[tsprctile(Bhls(:),2),tsprctile(Bhls(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('Bhls');
        saveas(figure(6),strcat(filepath,'Bhls.tif'));       
        figure(7);
        imagesc(Bfs,[tsprctile(Bfs(:),2),tsprctile(Bfs(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('Bfs');
        saveas(figure(7),strcat(filepath,'Bfs.tif'));       
        figure(8);
        imagesc(cda,[tsprctile(cda(:),2),tsprctile(cda(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('cda');
        saveas(figure(8),strcat(filepath,'cda.tif'));       
        figure(9);
        imagesc(MMT_t_4243,[tsprctile(MMT_t_4243(:),2),tsprctile(MMT_t_4243(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 4243');
        saveas(figure(9),strcat(filepath,'MMT_t_4243.tif'));      
        figure(10);
        imagesc(MMT_t_2434,[tsprctile(MMT_t_2434(:),2),tsprctile(MMT_t_2434(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 2434');
        saveas(figure(10),strcat(filepath,'MMT_t_2434.tif'));      
        figure(11);
        imagesc(MMT_t_1213,[tsprctile(MMT_t_1213(:),2),tsprctile(MMT_t_1213(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 1213');
        saveas(figure(11),strcat(filepath,'MMT_t_1213.tif'));     
        figure(12);
        imagesc(MMT_t_2131,[tsprctile(MMT_t_2131(:),2),tsprctile(MMT_t_2131(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ t_ 2131');
        saveas(figure(12),strcat(filepath,'MMT_t_2131.tif'));     
        figure(13);
        imagesc(PDxcheng,[tsprctile(PDxcheng(:),2),tsprctile(PDxcheng(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PDxcheng');
        saveas(figure(13),strcat(filepath,'PDxcheng.tif'));     
        figure(14);
        imagesc(rqxcheng,[tsprctile(rqxcheng(:),2),tsprctile(rqxcheng(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rqxcheng');
        saveas(figure(14),strcat(filepath,'rqxcheng.tif'));
        figure(15);
        imagesc(MMTPD,[tsprctile(MMTPD(:),2),tsprctile(MMTPD(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMTPD');
        saveas(figure(15),strcat(filepath,'MMTPD.tif'));
        figure(16);
        imagesc(MMTrq,[tsprctile(MMTrq(:),2),tsprctile(MMTrq(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMTrq');
        saveas(figure(16),strcat(filepath,'MMTrq.tif'));        
        figure(17);
        imagesc(PLsubDL,[tsprctile(PLsubDL(:),2),tsprctile(PLsubDL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubDL');
        saveas(figure(17),strcat(filepath,'PLsubDL.tif'));      
        figure(18);
        imagesc(PLsubDLunited,[tsprctile(PLsubDLunited(:),2),tsprctile(PLsubDLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubDLunited');
        saveas(figure(18),strcat(filepath,'PLsubDLunited.tif'));       
        figure(19);
        imagesc(rLsubqL,[tsprctile(rLsubqL(:),2),tsprctile(rLsubqL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rLsubqL');
        saveas(figure(19),strcat(filepath,'rLsubqL.tif'));
        figure(20);
        imagesc(rLsubqLunited,[tsprctile(rLsubqLunited(:),2),tsprctile(rLsubqLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rLsubqLunited');
        saveas(figure(20),strcat(filepath,'rLsubqLunited.tif'));       
        figure(21);
        imagesc(PLsubrL,[tsprctile(PLsubrL(:),2),tsprctile(PLsubrL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubrL');
        saveas(figure(21),strcat(filepath,'PLsubrL.tif'));        
        figure(22);
        imagesc(PLsubrLunited,[tsprctile(PLsubrLunited(:),2),tsprctile(PLsubrLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PLsubrLunited');
        saveas(figure(22),strcat(filepath,'PLsubrLunited.tif'));       
        figure(23);
        imagesc(DLsubqL,[tsprctile(DLsubqL(:),2),tsprctile(DLsubqL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('DLsubqL');
        saveas(figure(23),strcat(filepath,'DLsubqL.tif'));      
        figure(24);
        imagesc(DLsubqLunited,[tsprctile(DLsubqLunited(:),2),tsprctile(DLsubqLunited(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('DLsubqLunited');
        saveas(figure(24),strcat(filepath,'DLsubqLunited.tif'));      
        figure(25);
        imagesc(MMT_phi2233,[tsprctile(MMT_phi2233(:),2),tsprctile(MMT_phi2233(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi2233');
        saveas(figure(25),strcat(filepath,'MMT_phi2233.tif'));       
        figure(26);
        imagesc(MMT_phi3121,[tsprctile(MMT_phi3121(:),2),tsprctile(MMT_phi3121(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi3121');
        saveas(figure(26),strcat(filepath,'MMT_phi3121.tif'));        
        figure(27);
        imagesc(MMT_phi1312,[tsprctile(MMT_phi1312(:),2),tsprctile(MMT_phi1312(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi1312');
        saveas(figure(27),strcat(filepath,'MMT_phi1312.tif'));     
        figure(28);
        imagesc(MMT_phi4243,[tsprctile(MMT_phi4243(:),2),tsprctile(MMT_phi4243(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi4243');
        saveas(figure(28),strcat(filepath,'MMT_phi4243.tif'));      
        figure(29);
        imagesc(MMT_phi2434,[tsprctile(MMT_phi2434(:),2),tsprctile(MMT_phi2434(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ phi2434');
        saveas(figure(29),strcat(filepath,'MMT_phi2434.tif'));        
        figure(30);
        imagesc(PDslcha,[tsprctile(PDslcha(:),2),tsprctile(PDslcha(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('PDslcha');
        saveas(figure(30),strcat(filepath,'PDslcha.tif'));        
        figure(31);
        imagesc(rqslcha,[tsprctile(rqslcha(:),2),tsprctile(rqslcha(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('rqslcha');
        saveas(figure(31),strcat(filepath,'rqslcha.tif'));     
        end

        if func_equ == 1
        figure(32);
        imagesc(MMT_P1,[tsprctile(MMT_P1(:),2),tsprctile(MMT_P1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P1');
        saveas(figure(32),strcat(filepath,'MMT_P1.tif'));
        figure(33);
        imagesc(MMT_P2,[tsprctile(MMT_P2(:),2),tsprctile(MMT_P2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P2');
        saveas(figure(33),strcat(filepath,'MMT_P2.tif'));
        
        figure(34);
        imagesc(MMT_P3,[tsprctile(MMT_P3(:),2),tsprctile(MMT_P3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P3');
        saveas(figure(34),strcat(filepath,'MMT_P3.tif'));
        
        figure(35);
        imagesc(MMT_P4,[tsprctile(MMT_P4(:),2),tsprctile(MMT_P4(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P4');
        saveas(figure(35),strcat(filepath,'MMT_P4.tif'));
        
        figure(36);
        imagesc(MMT_P5,[tsprctile(MMT_P5(:),2),tsprctile(MMT_P5(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P5');
        saveas(figure(36),strcat(filepath,'MMT_P5.tif'));
        
        figure(37);
        imagesc(MMT_P6,[tsprctile(MMT_P6(:),2),tsprctile(MMT_P6(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P6');
        saveas(figure(37),strcat(filepath,'MMT_P6.tif'));
        
        figure(38);
        imagesc(MMT_P7,[tsprctile(MMT_P7(:),2),tsprctile(MMT_P7(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P7');
        saveas(figure(38),strcat(filepath,'MMT_P7.tif'));
        
        figure(39);
        imagesc(MMT_P8,[tsprctile(MMT_P8(:),2),tsprctile(MMT_P8(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P8');
        saveas(figure(39),strcat(filepath,'MMT_P8.tif'));
        
        figure(40);
        imagesc(MMT_P9,[tsprctile(MMT_P9(:),2),tsprctile(MMT_P9(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P9');
        saveas(figure(40),strcat(filepath,'MMT_P9.tif'));
        
        figure(41);
        imagesc(MMT_P10,[tsprctile(MMT_P10(:),2),tsprctile(MMT_P10(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P10');
        saveas(figure(41),strcat(filepath,'MMT_P10.tif'));
        
        figure(42);
        imagesc(MMT_P11,[tsprctile(MMT_P11(:),2),tsprctile(MMT_P11(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P11');
        saveas(figure(42),strcat(filepath,'MMT_P11.tif'));
        
        figure(42);
        imagesc(MMT_P12,[tsprctile(MMT_P12(:),2),tsprctile(MMT_P12(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMT_ P12');
        saveas(figure(42),strcat(filepath,'MMT_P12.tif'));

        figure(43);
        imagesc(Es,[tsprctile(Es(:),2),tsprctile(Es(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('Es');
        saveas(figure(43),strcat(filepath,'Es.tif'));
        
        figure(44);
        imagesc(E1,[tsprctile(E1(:),2),tsprctile(E1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E1');
        saveas(figure(44),strcat(filepath,'E1.tif'));
        
        figure(45);
        imagesc(E2,[tsprctile(E2(:),2),tsprctile(E2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E2');
        saveas(figure(45),strcat(filepath,'E2.tif'));
        
        figure(46);
        imagesc(E3,[tsprctile(E3(:),2),tsprctile(E3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E3');
        saveas(figure(46),strcat(filepath,'E3.tif'));
        
        figure(47);
        imagesc(E4,[tsprctile(E4(:),2),tsprctile(E4(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E4');
        saveas(figure(47),strcat(filepath,'E4.tif'));
        
        figure(48);
        imagesc(E5,[tsprctile(E5(:),2),tsprctile(E5(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E5');
        saveas(figure(48),strcat(filepath,'E5.tif'));
        
        figure(49);
        imagesc(E6,[tsprctile(E6(:),2),tsprctile(E6(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E6');
        saveas(figure(49),strcat(filepath,'E6.tif'));
        
        figure(50);
        imagesc(E7,[tsprctile(E7(:),2),tsprctile(E7(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E7');
        saveas(figure(50),strcat(filepath,'E7.tif'));
        
        figure(51);
        imagesc(E8,[tsprctile(E8(:),2),tsprctile(E8(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E8');
        saveas(figure(51),strcat(filepath,'E8.tif'));
        
        figure(52);
        imagesc(E9,[tsprctile(E9(:),2),tsprctile(E9(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('E9');
        saveas(figure(52),strcat(filepath,'E9.tif'));
        end
        
        if func_abrio == 1
        
        figure(53);
        imagesc(temp_1,[tsprctile(temp_1(:),2),tsprctile(temp_1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('temp_ 1');
        saveas(figure(53),strcat(filepath,'temp_1.tif'));
        
        figure(54);
        imagesc(temp_2,[tsprctile(temp_2(:),2),tsprctile(temp_2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('temp_ 2');
        saveas(figure(54),strcat(filepath,'temp_2.tif'));
        
        figure(55);
        imagesc(MMT_Abrio_R,[tsprctile(MMT_Abrio_R(:),2),tsprctile(MMT_Abrio_R(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ Abrio_ R');
        saveas(figure(55),strcat(filepath,'MMT_Abrio_R.tif'));
        
        figure(56);
        imagesc(MMT_Abrio_theta,[tsprctile(MMT_Abrio_theta(:),2),tsprctile(MMT_Abrio_theta(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('MMT_ Abrio_ theta');
        saveas(figure(56),strcat(filepath,'MMT_Abrio_theta.tif'));
        
        end

        if func_MM == 1
        
        figure(57);
        imagesc(MM_Det,[tsprctile(MM_Det(:),2),tsprctile(MM_Det(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Det');
        saveas(figure(57),strcat(filepath,'MM_Det.tif'));
        
        figure(58);
        imagesc(MM_Norm,[tsprctile(MM_Norm(:),2),tsprctile(MM_Norm(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Norm');
        saveas(figure(58),strcat(filepath,'MM_Norm.tif'));
        
        figure(59);
        imagesc(MM_Trace,[tsprctile(MM_Trace(:),2),tsprctile(MM_Trace(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MM_ Trace');
        saveas(figure(59),strcat(filepath,'MM_Trace.tif'));
        
        figure(60);
        imagesc(P_vec,[tsprctile(P_vec(:),2),tsprctile(P_vec(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('P_ vec');
        saveas(figure(60),strcat(filepath,'P_vec.tif'));
        
        figure(61);
        imagesc(D_vec,[tsprctile(D_vec(:),2),tsprctile(D_vec(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('D_ vec');
        saveas(figure(61),strcat(filepath,'D_vec.tif'));
        
        figure(62);
        imagesc(P_dot_D,[tsprctile(P_dot_D(:),2),tsprctile(P_dot_D(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('P_ dot_ D');
        saveas(figure(62),strcat(filepath,'P_dot_D.tif'));
        
        end
        
        if func_other == 1
        
        figure(63);
        imagesc(LDOP,[tsprctile(LDOP(:),2),tsprctile(LDOP(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('LDOP');
        saveas(figure(63),strcat(filepath,'LDOP.tif'));
        
        figure(64);
        imagesc(alpha_P,[tsprctile(alpha_P(:),2),tsprctile(alpha_P(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ P');
        saveas(figure(64),strcat(filepath,'alpha_P.tif'));
        
        figure(65);
        imagesc(alpha_D,[tsprctile(alpha_D(:),2),tsprctile(alpha_D(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ D');
        saveas(figure(65),strcat(filepath,'alpha_D.tif'));
        
        figure(66);
        imagesc(alpha_r,[tsprctile(alpha_r(:),2),tsprctile(alpha_r(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ r');
        saveas(figure(66),strcat(filepath,'alpha_r.tif'));
        
        figure(67);
        imagesc(alpha_q,[tsprctile(alpha_q(:),2),tsprctile(alpha_q(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ q');
        saveas(figure(67),strcat(filepath,'alpha_q.tif'));
        
%         figure(68);
%         imagesc(alpha_DP_cos,[tsprctile(alpha_DP_cos(:),2),tsprctile(alpha_DP_cos(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ DP_ cos');
%         saveas(figure(68),strcat(filepath,'alpha_DP_cos.tif'));
%         
%         figure(69);
%         imagesc(alpha_rq_cos,[tsprctile(alpha_rq_cos(:),2),tsprctile(alpha_rq_cos(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ rq_ cos');
%         saveas(figure(69),strcat(filepath,'alpha_rq_cos.tif'));
%         
%         figure(70);
%         imagesc(alpha_DP_sin,[tsprctile(alpha_DP_sin(:),2),tsprctile(alpha_DP_sin(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ DP_ sin');
%         saveas(figure(70),strcat(filepath,'alpha_DP_sin.tif'));
%         
%         figure(71);
%         imagesc(alpha_rq_sin,[tsprctile(alpha_rq_sin(:),2),tsprctile(alpha_rq_sin(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('alpha_ rq_ sin');
%         saveas(figure(71),strcat(filepath,'alpha_rq_sin.tif'));
        
        end
        
        if func_MMPD == 1
        
        figure(72);
        imagesc(MMPD_D,[tsprctile(MMPD_D(:),2),tsprctile(MMPD_D(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ D');
        saveas(figure(72),strcat(filepath,'MMPD_D.tif'));
        
        figure(73);
        imagesc(MMPD_DELTA,[tsprctile(MMPD_DELTA(:),2),tsprctile(MMPD_DELTA(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ DELTA');
        saveas(figure(73),strcat(filepath,'MMPD_DELTA.tif'));
        
        figure(74);
        imagesc(MMPD_delta,[tsprctile(MMPD_delta(:),2),tsprctile(MMPD_delta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ delta');
        saveas(figure(74),strcat(filepath,'MMPD_delta.tif'));
        
        figure(75);
        imagesc(MMPD_psi,[tsprctile(MMPD_psi(:),2),tsprctile(MMPD_psi(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ psi');
        saveas(figure(75),strcat(filepath,'MMPD_psi.tif'));
        
        figure(76);
        imagesc(MMPD_theta,[tsprctile(MMPD_theta(:),2),tsprctile(MMPD_theta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ theta');
        saveas(figure(76),strcat(filepath,'MMPD_theta.tif'));
        
        figure(77);
        imagesc(MMPD_R,[tsprctile(MMPD_R(:),2),tsprctile(MMPD_R(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMPD_ R');
        saveas(figure(77),strcat(filepath,'MMPD_R.tif'));
        
        end
        
        
        if func_angle == 1
        % 
        % figure(78);
        % imagesc(angle_1,[tsprctile(angle_1(:),2),tsprctile(angle_1(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 1');
        % saveas(figure(78),strcat(filepath,'angle_1.tif'));
        % 
        % figure(79);
        % imagesc(angle_2,[tsprctile(angle_2(:),2),tsprctile(angle_2(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 2');
        % saveas(figure(79),strcat(filepath,'angle_2.tif'));
        % 
        % figure(80);
        % imagesc(angle_3,[tsprctile(angle_3(:),2),tsprctile(angle_3(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 3');
        % saveas(figure(80),strcat(filepath,'angle_3.tif'));
        % 
        % figure(81);
        % imagesc(angle_4,[tsprctile(angle_4(:),2),tsprctile(angle_4(:),98)]); axis equal; axis off; colormap('hsv'); colorbar; title('angle_ 4');
        % saveas(figure(81),strcat(filepath,'angle_4.tif'));
        % 
        end
        if func_MMCD == 1
        
        figure(82);
        imagesc(MMCD_lambda1,[tsprctile(MMCD_lambda1(:),2),tsprctile(MMCD_lambda1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda1');
        saveas(figure(82),strcat(filepath,'MMCD_lambda1.tif'));
        
        figure(83);
        imagesc(MMCD_lambda2,[tsprctile(MMCD_lambda2(:),2),tsprctile(MMCD_lambda2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda2');
        saveas(figure(83),strcat(filepath,'MMCD_lambda2.tif'));
        
        figure(84);
        imagesc(MMCD_lambda3,[tsprctile(MMCD_lambda3(:),2),tsprctile(MMCD_lambda3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda3');
        saveas(figure(84),strcat(filepath,'MMCD_lambda3.tif'));
        
        figure(85);
        imagesc(MMCD_lambda4,[tsprctile(MMCD_lambda4(:),2),tsprctile(MMCD_lambda4(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ lambda4');
        saveas(figure(85),strcat(filepath,'MMCD_lambda4.tif'));
        
        figure(86);
        imagesc(MMCD_P1,[tsprctile(MMCD_P1(:),2),tsprctile(MMCD_P1(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P1');
        saveas(figure(86),strcat(filepath,'MMCD_P1.tif'));
        
        figure(87);
        imagesc(MMCD_P2,[tsprctile(MMCD_P2(:),2),tsprctile(MMCD_P2(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P2');
        saveas(figure(87),strcat(filepath,'MMCD_P2.tif'));
        
        figure(88);
        imagesc(MMCD_P3,[tsprctile(MMCD_P3(:),2),tsprctile(MMCD_P3(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ P3');
        saveas(figure(88),strcat(filepath,'MMCD_P3.tif'));
        
        figure(89);
        imagesc(MMCD_PI,[tsprctile(MMCD_PI(:),2),tsprctile(MMCD_PI(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ PI');
        saveas(figure(89),strcat(filepath,'MMCD_PI.tif'));
        
        figure(90);
        imagesc(MMCD_PD,[tsprctile(MMCD_PD(:),2),tsprctile(MMCD_PD(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ PD');
        saveas(figure(90),strcat(filepath,'MMCD_PD.tif'));
        
        figure(91);
        imagesc(MMCD_S,[tsprctile(MMCD_S(:),2),tsprctile(MMCD_S(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMCD_ S');
        saveas(figure(91),strcat(filepath,'MMCD_S.tif'));
        
        end
                
        if func_MMLD == 1
        
        figure(92);
        imagesc(MMLD_D,[tsprctile(MMLD_D(:),2),tsprctile(MMLD_D(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ D');
        saveas(figure(92),strcat(filepath,'MMLD_D.tif'));
        
        figure(93);
        imagesc(MMLD_delta,[tsprctile(MMLD_delta(:),2),tsprctile(MMLD_delta(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ delta');
        saveas(figure(93),strcat(filepath,'MMLD_delta.tif'));
        
        figure(94);
        imagesc(MMLD_alpha,[tsprctile(MMLD_alpha(:),2),tsprctile(MMLD_alpha(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ alpha');
        saveas(figure(94),strcat(filepath,'MMLD_alpha.tif'));
        
        figure(95);
        imagesc(MMLD_CD,[tsprctile(MMLD_CD(:),2),tsprctile(MMLD_CD(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ CD');
        saveas(figure(95),strcat(filepath,'MMLD_CD.tif'));
        
        figure(96);
        imagesc(MMLD_a22,[tsprctile(MMLD_a22(:),2),tsprctile(MMLD_a22(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ a22');
        saveas(figure(96),strcat(filepath,'MMLD_a22.tif'));
        
        figure(97);
        imagesc(MMLD_a33,[tsprctile(MMLD_a33(:),2),tsprctile(MMLD_a33(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ a33');
        saveas(figure(97),strcat(filepath,'MMLD_a33.tif'));
        
        figure(98);
        imagesc(MMLD_aL,[tsprctile(MMLD_aL(:),2),tsprctile(MMLD_aL(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ aL');
        saveas(figure(98),strcat(filepath,'MMLD_aL.tif'));
        
        figure(99);
        imagesc(MMLD_a44,[tsprctile(MMLD_a44(:),2),tsprctile(MMLD_a44(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ a44');
        saveas(figure(99),strcat(filepath,'MMLD_a44.tif'));
        
        figure(100);
        imagesc(MMLD_aLA,[tsprctile(MMLD_aLA(:),2),tsprctile(MMLD_aLA(:),98)]); axis equal; axis off; colormap('jet'); colorbar; title('MMLD_ aLA');
        saveas(figure(100),strcat(filepath,'MMLD_aLA.tif'));
        end
        disp('save picetures completed！');
    end
        toc;
%%
% Function: save PBPs group

    if func_PBP == 1
       tic;
       pbp_name = ["FinalM14", "FinalM41", "FinalM44"...
                   "MMT_t1", "MMT_b", "MMT_b2", "MMT_A", "MMT_Abrio_R", "MMT_beta", "Bhls", "Bfs"...
                   "MMT_t_1213", "MMT_t_2131", "MMT_t_4243", "MMT_t_2434", "PDxcheng", "rqxcheng", "MMTPD", "MMTrq"...
                   "PLsubDL", "rLsubqL", "PLsubrL", "DLsubqL", "PDslcha", "rqslcha", "MM_Det", "MM_Norm", "MM_Trace"...
                   "P_vec", "D_vec", "P_dot_D", "LDOP"...
                   "MMT_P1","MMT_P2","MMT_P3","MMT_P4","MMT_P5","MMT_P6","MMT_P7","MMT_P8","MMT_P9","MMT_P10","MMT_P11","MMT_P12"...
                   "E1","E2","E3","E4","E5","E6","Es"...
                   "MMPD_D", "MMPD_delta", "MMPD_R"...
                   "MMCD_lambda1", "MMCD_lambda2", "MMCD_lambda3", "MMCD_lambda4", "MMCD_P1", "MMCD_P2", "MMCD_P3"...
                   "MMCD_PI", "MMCD_PD", "MMCD_S"];
    
       % pbp_name = ["FinalM14", "FinalM41", "FinalM44"...
       %             "MMT_t1", "MMT_b", "MMT_Abrio_R", "MMT_beta"...
       %             "MMT_t_1213", "PDxcheng", "rqxcheng", "MMTPD", "MMTrq"...
       %             "PLsubDL", "rLsubqL", "PDslcha", "rqslcha", "MM_Det", "P_dot_D"...
       %             "MMCD_lambda2"...
       %             "MMLD_a22", "MMLD_a33", "MMLD_a44", "MMLD_aLA"];
      
       for i = 1:size(pbp_name,2)
           load(char([filepath+pbp_name(i)+".mat"]))
       end

       pbps=zeros(size(FinalM12));

       for i = 1:size(pbp_name,2)
           pbps(:,:,i)=eval(pbp_name(i));
       end
       save([char(fileset(k,1)),'\pbps.mat'],'pbps');
       disp('save PBPs group completed');
       toc;

    end
end
toc;