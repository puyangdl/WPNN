% Online Non-Stationary Algorithm
% Store pdf in each timestep, normalise then calculate accuracy
% 
% @Input
%     xTrain: data input
%     label: label for calculating current classification rate
%     j0: dialation parameter
%     wavelet: order of wavelet
%     alpha: discount factor
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
function [time_used,acc,time_used_train,lbl,time_used_evaluation, pdf] = fun_WPNN_pdf_online_nonstationary_prequential(xTrain, label, j0,wavelet,alpha)

nclasses = length(unique(label));
time_used=[];
time_used_train = [];
time_used_evaluation = [];
tmp_nansum_now=[0 0];
pdf = [];
acc = [];
lbl = {};

x1= xTrain(label == 1,:);
x2= xTrain(label == 2,:);

[n,m,constant,r,R,k]=fun_wpnn_initialisation(x1,wavelet,j0);

M1 = size(k,1);
w1 = zeros(1,M1);

w2 = zeros(1,M1);

N_train = size(xTrain,1);

for t = 1:N_train
    
    %Prequential
    %******************** class evaluation ******************** -> input x1
    tic
    b_frame = fun_relevant_frame_for_given_datapoint(xTrain(t,:),r,m,constant,R,n);
    [a,b] = fun_wpnn_evaluation_dual_class(xTrain(t,:),w1,w2,r,m,constant,k,R,n,wavelet,b_frame);

    tmp = cat(1,a,b).';
    
    tmp_nansum_now = tmp_nansum_now+sum(tmp,1);
    
    tmp_norm = tmp./tmp_nansum_now;
    
    [~,lblTesting]=nanmax(tmp_norm,[],2);
    time_eval_tmp = toc;
    time_used_evaluation = [time_used_evaluation time_eval_tmp];
    
    pdf = [pdf; tmp_norm];
    
    %Remove pdf&label when not being predicted
    sum_pTesting = nansum(tmp_norm(end,:),2);

    lblTesting(sum_pTesting==0) = 0;
    acc = [acc 100*(1-(sum(lblTesting~=label(t))/length(label(t))))];
    
    lbl{t} =  lblTesting;
    
    if(label(t) == 1)
        tic
        w1 = fun_wpnn_online_updating_nonstationary(xTrain(t,:),r,m,constant,k,R,n,wavelet,M1,w1,alpha,b_frame);
        time_train_c1 = toc;
    else
        tic
        w2 = fun_wpnn_online_updating_nonstationary(xTrain(t,:),r,m,constant,k,R,n,wavelet,M1,w2,alpha,b_frame);
        time_train_c1 = toc;
    end
    
    time_used_train = [time_used_train time_train_c1];
    time_used = [time_used time_train_c1+time_eval_tmp];

end
end



