% Online Stationary Algorithm
% Store pdf in each timestep, normalise then calculate accuracy
% 
% @Input
%     xTrain: data input
%     label: label for calculating current classification rate
%     j0: dialation parameter
%     wavelet: order of wavelet
% 
% @Output
%     time_used: total running time = training time+pdf evaluation time
%     acc: classification rate at the current timestamp
%     time_used_train: training time
%     predicted_label: predicted label
%     time_used_evaluation: total time to calculate the current pdf for current input
%     pdf: normalised pdf at each timestamp
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022

function [time_used,acc,time_used_train,predicted_label,time_used_evaluation, pdf] = fun_WPNN_pdf_online_stationary_prequential(xTrain, label,j0,wavelet)

time_used=[];
time_used_train = [];
time_used_evaluation = [];
pdf_cat_latest=[0 0];
pdf = [];
acc = [];
predicted_label = {};

% network initialisation
x1= xTrain(label == 1,:);
[n,m,constant,r,R,k]=fun_wpnn_initialisation(x1,wavelet,j0);
M1 = size(k,1);
w1 = zeros(1,M1);
w2 = zeros(1,M1);
N_train = size(xTrain,1);

% network start
for t = 1:N_train
    
    %prequential error -> test before train
    %test t+1 -> train t+1 data -> test t+2 -> train t+2......
    
    %******************** class evaluation ********************
    tic
    b_frame = fun_relevant_frame_for_given_datapoint(xTrain(t,:),r,m,constant,R,n);
    [a,b] = fun_wpnn_evaluation_dual_class(xTrain(t,:),w1,w2,r,m,constant,k,R,n,wavelet,b_frame);
    
    %normalisation
    pdf_cat = cat(1,a,b).';    
    pdf_cat_latest = pdf_cat_latest+sum(pdf_cat,1);    
    pdf_cat_norm = pdf_cat./pdf_cat_latest;
    
    %predicted labels
    [~,lblTesting]=nanmax(pdf_cat_norm,[],2);
    time_eval_tmp = toc;
    
    pdf = [pdf; pdf_cat_norm];
    
    time_used_evaluation = [time_used_evaluation time_eval_tmp];
    
    % For calculating eval metrics only: Remove pdf&label when not being predicted
    % Special case at timestamp 1, both pdfs = 0, since w1,w2 ={0}
    sum_pTesting = nansum(pdf_cat_norm(end,:),2);    
    lblTesting(sum_pTesting==0) = 0;
    
    acc = [acc 100*(1-(sum(lblTesting~=label(t))/length(label(t))))];
    
    predicted_label{t} =  lblTesting;
    
    %update the network parameters
    if(label(t) == 1)
         tic
         w1 = fun_wpnn_online_updating_stationary(xTrain(t,:),r,m,constant,k,R,n,wavelet,M1,w1,t,b_frame);
         time_train_c1 = toc;
    else
        tic
        w2 = fun_wpnn_online_updating_stationary(xTrain(t,:),r,m,constant,k,R,n,wavelet,M1,w2,t,b_frame);
        time_train_c1 = toc;
    end
    
    time_used_train = [time_used_train time_train_c1];
    time_used = [time_used time_train_c1+time_eval_tmp];

end
end