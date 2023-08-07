%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Run trained classifier and obtain classifier outputs
% Inputs:
% 1. model
% 2. data directory
% 3. patient id
%
% Outputs:
% 1. outcome
% 2. outcome probability
% 3. CPC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [outcome_binary, outcome_probability, cpc] = team_testing_code(model,input_directory,patient_id,verbose)

% transfer entropy - channel combination
channels = {'Fp1', 'Fp2', 'F3','F4'};
test_chs_L=3;
[PartA, PartB]=z_bipartition_TE(test_chs_L);

try
    time_th=10;
    features=get_features(input_directory,patient_id,PartA,PartB,channels,time_th);
    if sum(sum(features))~=0
        if sum(sum(isnan(features')))~=0
            features(sum(isnan(features'))>0,:)=[];
        end
        if size(features,1)>0
            c_d_num=[];
            for c_i=1:size(features,2)-1
                d_tmp=find(abs(features(:,c_i)) > 999);
                c_d_num=vertcat(c_d_num,d_tmp);
                clear d_tmp
            end
            if sum(c_d_num)~=0
                features(c_d_num,:)=[];
            end
            if size(features,1)>0
                time_limit=30;
                features((features(:,end) < time_limit),:)=[];
                features(:,end)=[];
                if size(features,1)>0
                    decision_all=zeros(size(features,1),size(model.model_outcome,2));
                    for model_i=1:size(model.model_outcome,2)
                        model_solo=model.model_outcome{model_i};
                        decision_all(:,model_i)= predict(model_solo,features);
                        clear model_solo
                    end
                    outcome_probability=mean(decision_all(:));
                else
                    outcome_probability=0; % good
                end
            else
                outcome_probability=0; % good
            end
        else
            outcome_probability=0; % good
        end
    else
        outcome_probability=0; % good
    end
    
    % outcome_binary
    if outcome_probability <= 0.5
        outcome_binary=0; % good
    else
        outcome_binary=1; % poor
    end
    % cpc
    if outcome_probability <= 0.5
        if outcome_probability <= 0.3
            cpc=1;
        else
            cpc=2;
        end
    else
        if outcome_probability <= 0.6
            cpc=3;
        elseif outcome_probability <= 0.7
            cpc=4;
        else
            cpc=5;
        end
    end
catch    
    outcome_probability=0; % good
    outcome_binary=0; % good
    cpc=1;
end


% prob. modify
if outcome_probability==0
    outcome_probability=mod(randn(1,1),1)/100;
elseif outcome_probability==1
    outcome_probability=1-mod(randn(1,1),1)/100;
end


end


%---------------------------------------------
function [PartA, PartB]=z_bipartition_TE(channel_length)
G = 1:channel_length;
f = 1;
for N = 2:length(G)
    % N=Nneurons;
    cases = nchoosek(G, N);
    maxM = N-1; %
    for idxD=1:size(cases,1)
        m = 0; %
        for idxC=1:maxM
            tmp = nchoosek(1:N, idxC);
            for idxS=1:size(tmp,1)

                m = m + 1;
                PartA{f,1}{idxD,m} = cases(idxD,tmp(idxS,:));
                PartB{f,1}{idxD,m} = setdiff(cases(idxD,:),PartA{f,1}{idxD,m});
            end
        end
    end
    f = f+1;
end
end