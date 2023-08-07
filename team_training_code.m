function model = team_training_code(input_directory,output_directory, verbose) % train_EEG_classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Purpose: Train EEG classifiers and obtain the models
% Inputs:
% 1. input_directory
% 2. output_directory
%
% Outputs:
% 1. model: trained model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if verbose>=1
    disp('Finding challenge data...')
end

% Find the folders
patient_ids=dir(input_directory);
patient_ids=patient_ids([patient_ids.isdir]==1);
patient_ids(1:2)=[]; % Remove "./" and "../" paths
patient_ids={patient_ids.name};
num_patients = length(patient_ids);

% Create a folder for the model if it doesn't exist
if ~isfolder(output_directory)
    mkdir(output_directory)
end
fprintf('Loading data for %d patients...\n', num_patients)

bad_id=[367	424	435	451 472	517	527	575	591	592	609 615 766	821	976	980];

channels = {'Fp1', 'Fp2', 'F3','F4'};
test_chs_L=length(channels);
[PartA, PartB]=z_bipartition_TE(test_chs_L);

j_cnt=0;
for j=1:num_patients
    
    if verbose>1
        fprintf('%d/%d \n',j,num_patients)
    end

    % Extract features
    patient_id=patient_ids{j};

    if ~isempty(find(bad_id==str2double(patient_id), 1))
        continue
    end

    time_th=70;
    current_features=get_features(input_directory,patient_id,PartA,PartB,channels,time_th);

    if size(current_features,1)>0
        if sum(sum(isnan(current_features')))~=0
            current_features(sum(isnan(current_features'))>0,:)=[];
        end
        if size(current_features,1)>0
            c_d_num=[];
            for c_i=1:size(current_features,2)-1
                d_tmp=find(abs(current_features(:,c_i)) > 999);
                c_d_num=vertcat(c_d_num,d_tmp);
                clear d_tmp
            end
            if sum(c_d_num)~=0
                current_features(c_d_num,:)=[];
            end            
            if size(current_features,1)>0                
                time_limit=30;
                current_features((current_features(:,end) < time_limit),:)=[];
                
                if size(current_features,1)>0
                    j_cnt=j_cnt+1;

                    [patient_metadata,~]=load_challenge_data(input_directory,patient_id);
                    current_outcome=get_outcome(patient_metadata);

                    current_features(:,size(current_features,2)+1)=current_outcome;
                    features_struct{j_cnt}=current_features;            
                end
            end
        end
    end

    clear current_features patient_id hos_Num
    clear patient_metadata current_outcome c_d_num
end

%% train model

f_ratio=0.5;
model_L=round(5/f_ratio); % model size
model_outcome=[];
for m_i=1:model_L
    % m_i
    features_all=[];
    for f_j=1:length(features_struct)
        current_features=features_struct{f_j};
        time_n=unique(current_features(:,end-1));
        for t_i=1:length(time_n)
            current_N=find(time_n(t_i)==current_features(:,end-1));
            test_f_n=current_N(randperm(length(current_N),round(length(current_N)*f_ratio)));
            features_all=vertcat(features_all,current_features(test_f_n,:));
            clear test_f_n current_N
        end
        clear current_features time_n
    end

    features_all(:,end-1)=[];
    model_outcome{m_i} = fitcsvm(features_all(:,1:end-1),features_all(:,end),'Standardize',true,'KernelFunction','RBF','KernelScale','auto');
    clear selected_Num selected_Num1 selected_Num2 features_all
end

% model save
model_cpc=model_outcome;
save_model(model_outcome,model_cpc,output_directory);

end

%% functions
%---------------------------------------------
function save_model(model_outcome,model_cpc,output_directory) 
% Save results.
filename = fullfile(output_directory,'model.mat');
save(filename,'model_outcome','model_cpc','-v7.3');
disp('Done.')
end

%---------------------------------------------
function outcome=get_outcome(patient_metadata)
patient_metadata=strsplit(patient_metadata,'\n');
outcome_tmp=patient_metadata(startsWith(patient_metadata,'Outcome:'));
outcome_tmp=strsplit(outcome_tmp{1},':');
if strncmp(strtrim(outcome_tmp{2}),'Good',4)
    outcome=0;
elseif strncmp(strtrim(outcome_tmp{2}),'Poor',4)
    outcome=1;
else
    keyboard
end
end

%---------------------------------------------
function [PartA, PartB]=z_bipartition_TE(channel_length)
G = 1:channel_length;
f = 1;
for N = 2:length(G)
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


