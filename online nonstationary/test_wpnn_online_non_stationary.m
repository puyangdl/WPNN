%% Online non stationary - synthetic dataset
% Code to test synthetic dataset using WPNN online Non-Stationary
% algorithm (Algorithm 4)
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022
%% REPEAT N TIMES
clc; clear all; close all;  set(0,'DefaultFigureWindowStyle','docked');

waveletv={'bsplines_linear','bsplines_quad','bsplines_cubic'};
nom={'linear','quadratic','cubic'};

num = 2*10^4;
samples = 10;
load(['./datasets/non_stationary_dataset_2d_',num2str(num),'_',num2str(samples),'samples.mat'])

x = data{1};
x_train = xTrain{1};
y_train = yTrain{1};

% Best hyper-params
Nvec=[100];%[100 300 500 700 1000];
levelv=1;%1:1:5

repeat_num = 1;
for w = 1%length(waveletv)
    wavelet=waveletv{w};
    for k =1:length(levelv)
        level = levelv(k);
        for i=1:length(Nvec)
            N=Nvec(i);
            alpha=1/N;
            for j=1:repeat_num
                fprintf('%s, data window = %d, j0=%d, data idx %d \n',wavelet, N ,level,j)
                
                [time_tmp, acc_tmp, time_tmp_train,testing_lbl,time_used_evaluation_tmp,pdf_tmp] = fun_WPNN_pdf_online_nonstationary_prequential(x_train,y_train,level,wavelet,alpha);
             
                time_used(j,:) = time_tmp(1,:);
                accuracy(j,:) = acc_tmp(1,:);
                time_used_train(j,:) = time_tmp_train(1,:);
                test_label(j,:) = testing_lbl;
                time_used_evaluation(j,:) = time_used_evaluation_tmp(1,:);
                pdf{j,:} = pdf_tmp;
                
            end
            total_accuracy(i,:)=mean(accuracy,1);
            total_time_used(i,:) = mean(time_used,1);
            total_time_used_train(i,:) = mean(time_used_train,1);
            total_time_used_evaluation(i,:) = mean(time_used_evaluation,1);

        end
        
        accuracy_nonstationary{k}.wavelet = wavelet;
        accuracy_nonstationary{k}.level = level;
        accuracy_nonstationary{k}.accuracy = total_accuracy;
        accuracy_nonstationary{k}.total_time = total_time_used;
        accuracy_nonstationary{k}.training_time = total_time_used_train;
        accuracy_nonstationary{k}.total_time_used_evaluation = total_time_used_evaluation;
        accuracy_nonstationary{k}.window_size = Nvec;
    end
    
    
    save(['./results/online_non_stationary_',wavelet,'_repeat_',num2str(repeat_num),'_',date,'.mat'],'accuracy_nonstationary','repeat_num','Nvec','-v7.3')
end