% Code to test synthetic dataset using WPNN online Stationary
% algorithm (Algorithm 3)
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022


%% Online Stationary  - Synthetic dataset
% Repeat N Times
clc; clear all; close all;  set(0,'DefaultFigureWindowStyle','docked');
waveletv={'bsplines_linear','bsplines_quad','bsplines_cubic'};
nom={'linear','quadratic','cubic'};

%load data
num = 2*10^4;
load(['./datasets/online_stationary_datasets.mat']);

%optimal hyperparameters: linear b-spline + j0=1
levelv=1;

repeat_num = 1;

x = data{1};
x_test = xTest{1};
x_train = xTrain{1};
y_test = yTest{1};
y_train = yTrain{1};

for w = 1:1
    wavelet=waveletv{w};
    for k =1:length(levelv)
        level = levelv(k);
        for j=1:repeat_num
            fprintf('%s, j0=%d, data idx %d \n',wavelet,level,j)
            
            [time_tmp, acc_tmp, time_tmp_train,testing_lbl,time_used_evaluation_tmp,pdf_tmp] = fun_WPNN_pdf_online_stationary_prequential(x_train,y_train,level,wavelet);
            
            fprintf('Current accuracy: %.4f\n\n',mean(acc_tmp))
            
            time_used(j,:) = time_tmp(1,:);
            accuracy(j,:) = acc_tmp(1,:);
            time_used_train(j,:) = time_tmp_train(1,:);
            test_label(j,:) = testing_lbl;
            time_used_evaluation(j,:) = time_used_evaluation_tmp(1,:);
            pdf{j,:} = pdf_tmp;
            
        end
        %save results
        i=1;
        total_accuracy(i,:)=mean(accuracy,1);
        total_time_used(i,:) = mean(time_used,1);
        total_time_used_train(i,:) = mean(time_used_train,1);
        
        total_time_used_evaluation(i,:) = mean(time_used_evaluation,1);
        
        accuracy_stationary{k}.wavelet = wavelet;
        accuracy_stationary{k}.level = level;
        accuracy_stationary{k}.accuracy = total_accuracy;
        accuracy_stationary{k}.total_time = total_time_used;
        accuracy_stationary{k}.training_time = total_time_used_train;
        accuracy_stationary{k}.total_time_used_evaluation = total_time_used_evaluation;
    end
    
    save(['./results/online_stationary_',wavelet,'_',num2str(num),'samples_repeat_',num2str(repeat_num),'_',date,'.mat'],'accuracy_stationary','repeat_num','-v7.3')
end
