% PNN using sliding window
%
% @Input:
%   xTrain: x train
%   label: y train
%   xTest: x test
%   bandwidth: bandwidth for the KDE
%   y_test: y test
%   window: sliding window size
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
function [time_used,acc,lbl] = fun_kde_testing_online_window(xTrain, label, xTest, bandwidth, y_test,window)

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
train_set_2 = [];

for t = 1:N_train
    if(t~=1)
        size_1 = size(train_set_1,1);
        idx_1 = min(window, size_1);
        if(length(train_set_1)<window+1 )
            if(~isempty(train_set_1))
                xTrain_1_win = train_set_1(1:idx_1,:);
            end
        elseif(size_1 < N1)
                xTrain_1_win = train_set_1(size_1-window+1:end,:);
        end
        
        size_2 = size(train_set_2,1);
        idx_2 = min(window, size_2);
        if(length(train_set_2)<window+1 )
            if(train_set_2)
                xTrain_2_win = train_set_2(1:idx_2,:);
            end
        elseif(size_2 < N2)
                xTrain_2_win = train_set_2(size_2-window+1:end,:);
        end
        
        
    else
         xTrain_1_win = train_set_1;
         xTrain_2_win = train_set_2;
    end
    %prequential error -> test before train
    
    %******************** class evaluation ********************
    tic
    a =  funmyKDE(xTrain(t,:),xTrain_1_win,bandwidth);
    b =  funmyKDE(xTrain(t,:),xTrain_2_win,bandwidth);
    
    pTesting = cat(2,a,b);
    
    [~,lblTesting]=nanmax(pTesting,[],2);
    
    %Remove pdf&label when not being predicted
    sum_pTesting = sum(pTesting,2);
    lblTesting(sum_pTesting==0) = 0;
    if((isnan(a)&isnan(b)))
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