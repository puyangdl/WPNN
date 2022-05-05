% PNN using the latest point only
%
% @Input:
%   xTrain: x train
%   label: y train
%   xTest: x test
%   bandwidth: bandwidth for the KDE
%   y_test: y test
%               
%@Output
%   time_used: total computation time
%   acc: accuracy in each timestamp
%   lbl: computed label
% 
% Author: E. S. Garcia-Trevino, Pu Yang, and J. A. Barria
% Paper: Wavelet Probabilistic Neural Networks, 
%            IEEE Transactions on Neural Networks and Learning Systems, 
%            2022
% Institution: Imperial College London
% Date: Jan-2022
%
function [time_used,acc,lbl] = fun_kde_testing_online_latest_pt(xTrain, label, xTest, bandwidth, y_test)

nclasses = length(unique(label));
time_used=[];
time_used_train = [];
acc = [];
lbl={};
x1= xTrain(label == 1,:);
x2= xTrain(label == 2,:);
y1=label(label == 1,:);
y2=label(label == 2,:);

N1 = size(x1,1);
N2 = size(x2,1);
N_train = size(xTrain,1);

train_set_1 = [];
train_set_2 = [];zeros(1,size(xTrain,2));
tmp=[];
tmp_norm_now=[0 0];

for t = 1:N_train
 
    %prequential error -> test before train
    
    %******************** class evaluation ********************
    %note: different from wpnn, evaluation for pnn a/b return transpose
    %version compared to WPNN -> different cat() dimension used
    tic
    a =  funmyKDE(xTrain(t,:),train_set_1,bandwidth);
    b =  funmyKDE(xTrain(t,:),train_set_2,bandwidth);
    
    tmp = cat(2,a,b);
    tmp_norm_now = tmp_norm_now+nansum(tmp,1);
    tmp_norm = tmp./tmp_norm_now;
    
    [~,lblTesting]=nanmax(tmp_norm,[],2);
    
    sum_pTesting = nansum(tmp_norm,2);
        
    if(isnan(a) && isnan(b))
        lblTesting = 0;
    end

    acc = [acc 100*(1-(sum(lblTesting~=label(t))/length(label(t))))];

    lbl{t} = lblTesting;
    if(label(t) == 1)
            train_set_1 = [train_set_1; xTrain(t,:)];
    else
            train_set_2 = [train_set_2; xTrain(t,:)];
    end
    time_tmp_c1 = toc;
    time_used = [time_used time_tmp_c1];
end

end

