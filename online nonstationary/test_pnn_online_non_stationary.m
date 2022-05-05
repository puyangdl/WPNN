% Code to test synthetic dataset using Gaussian-KDE PNN
% 
% Note:
%     As discussed in the paper, 
%     this code contains two part: (a) using a sliding window with size N to select the most N data
%                                                  (b) using the latest data only
%
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022
%% PNN -  Synthetic dataset  
%Repeat for N times using window
clc; clear all; close all;  set(0,'DefaultFigureWindowStyle','docked');

num = 2*10^4;
samples = 10;
load(['./datasets/non_stationary_dataset_2d_',num2str(num),'_',num2str(samples),'samples.mat'])

Nvec=[500];
bandwidth_set = [0.001];

[lx,dimension]=size(data{1});
repeat_num = 1;

x = data{1};
x_test = xTest{1};
x_train = xTrain{1};
y_test = yTest{1};
y_train = yTrain{1};

for bd = 1:length(bandwidth_set)
    bandwidth = bandwidth_set(bd);
    for i = 1:length(Nvec)
        N = Nvec(i);
        for j=1:repeat_num%1%dataset_num
            fprintf('kde, data idx %d, window size = %d, bandwidth %.4f \n',j,N ,bandwidth)
            
            [time_tmp, acc_tmp,testing_lbl] = fun_kde_testing_online_window(x_train,y_train,x_test,bandwidth,y_test,N);
            time_used(j,:) = time_tmp(1,:);
            accuracy(j,:) = acc_tmp(1,:);
            test_label(j,:) = testing_lbl;
        end
        total_accuracy(i,:)=mean(accuracy,1);
        total_test_label{i,:} = test_label;
        
    end
    accuracy_stationary{bd}.accuracy = total_accuracy;
    accuracy_stationary{bd}.time = total_time_used;
    accuracy_stationary{bd}.bandwidth = bandwidth;
    accuracy_stationary{bd}.testing_label = total_test_label;
    save(['./results/online_non_stationary_kde_dataset_',num2str(num),'_samples_window_repeat_',num2str(repeat_num),'_',date,'.mat'],'accuracy_stationary','Nvec','bandwidth_set','repeat_num')
    
end

%% Repeat for N times using latest prequential

clc; clear all; close all;  set(0,'DefaultFigureWindowStyle','docked');


num = 2*10^4;
samples = 10;

load(['./datasets/non_stationary_dataset_2d_',num2str(num),'_',num2str(samples),'samples.mat'])

bandwidth_set = [0.01];

repeat_num = 100;

x = data{1};
x_test = xTest{1};
x_train = xTrain{1};
y_test = yTest{1};
y_train = yTrain{1};

for bd = 1:length(bandwidth_set)
    bandwidth = bandwidth_set(bd);
    i=1;
    for j=1:repeat_num
        fprintf('kde, data idx %d, bandwidth %.4f \n',j,bandwidth)
        
        [time_tmp, acc_tmp,testing_lbl] = fun_kde_testing_online_latest_pt(x_train,y_train,x_test,bandwidth,y_test);
        time_used(j,:) = time_tmp(1,:);
        accuracy(j,:) = acc_tmp(1,:);
        test_label(j,:) = testing_lbl;
    end
    total_accuracy(i,:)=mean(accuracy,1);
    total_time_used(i,:) = mean(time_used,1);
    total_test_label{i,:} = test_label;
    
    accuracy_stationary{bd}.accuracy = total_accuracy;
    accuracy_stationary{bd}.time = total_time_used;
    accuracy_stationary{bd}.bandwidth = bandwidth;
    accuracy_stationary{bd}.testing_label = total_test_label;
    
    save(['./results/online_non_stationary_kde_dataset_20000_samples_latest_pt_repeat_',num2str(repeat_num),'_',date,'.mat'],'accuracy_stationary','repeat_num','-v7.3')
end
