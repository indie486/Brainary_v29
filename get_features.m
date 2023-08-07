
function current_features=get_features(input_directory,patient_id,PartA,PartB,channels,time_th)

[~,recording_ids]=load_challenge_data(input_directory,patient_id);
recording_ids=unique(recording_ids);
num_recordings=length(recording_ids);

% TE delay
T_D_L=3;
block_size=60; % 60 s

time_max_f=15;
max_feature_L=time_max_f*60;

mag_threshold=500; % mag thresh
corr_threshold=0.99;

base_passband = [0.5, 30];

% Extract EEG features
channel_L=size(channels,2);
group='EEG';

% notch filter
notch_25f=[];
notch_30f=[];
notch_50f=[];
notch_60f=[];

current_features=[];

if num_recordings > 0
    
    recording_id=recording_ids{num_recordings};
    if str2double(recording_id(end-2:end)) > time_th
        clear recording_id

        for time_i=1:num_recordings
            time_max_f= 15-fix(time_i * 0.3); % feature size for each time
            if time_max_f <= 1
                break
            end
            recording_id=recording_ids{num_recordings-time_i+1}

            if str2double(recording_id(end-2:end)) <= 72 && str2double(recording_id(end-2:end)) > 30
                recording_location = fullfile(input_directory,patient_id,sprintf('%s_%s',recording_id,group));
                if exist([recording_location '.hea'],'file')>0 && exist([recording_location '.mat'],'file')>0
                    try
                        [origial_signal,fs,signal_channels]=load_recording(recording_location);
                    catch
                        continue
                    end

                    % notch filter
                    if isempty(notch_25f)
                        notch_25f=designfilt('bandstopiir','FilterOrder',2,'HalfPowerFrequency1',24,'HalfPowerFrequency2',26,'Designmethod','butter','SampleRate',fs);
                    end                    
                    if isempty(notch_30f)
                        notch_30f=designfilt('bandstopiir','FilterOrder',2,'HalfPowerFrequency1',29,'HalfPowerFrequency2',31,'Designmethod','butter','SampleRate',fs);
                    end                    
                    if isempty(notch_50f)
                        notch_50f=designfilt('bandstopiir','FilterOrder',2,'HalfPowerFrequency1',49,'HalfPowerFrequency2',51,'Designmethod','butter','SampleRate',fs);
                    end
                    if isempty(notch_60f)
                        notch_60f=designfilt('bandstopiir','FilterOrder',2,'HalfPowerFrequency1',59,'HalfPowerFrequency2',61,'Designmethod','butter','SampleRate',fs);
                    end

                    while_on=1; on_cnt=0;
                    while while_on == 1
                        while_on=0;
                        [signal_data, ~] = reduce_channels(origial_signal, channels, signal_channels);

                        % band-pass filter
                        [b,a]=butter(1,base_passband/(fs/2));
                        input_DATA = filtfilt(b,a,signal_data');
                        input_DATA=detrend(input_DATA,1);
                        input_DATA=(input_DATA-mean(input_DATA)); % DC
                        clear b a

                        % notch filter
                        input_DATA=filtfilt(notch_25f,input_DATA);
                        input_DATA=filtfilt(notch_30f,input_DATA);
                        input_DATA=filtfilt(notch_50f,input_DATA);
                        input_DATA=filtfilt(notch_60f,input_DATA);

                        %% noise check from signal
                        if size(signal_data,2) >= fs*block_size*2
                            block_duration=block_size*fs;
                            block_max_L=fix(size(signal_data,2)/block_duration);
                            if block_max_L > 0
                                block_Num=1:block_max_L;
                                d_block=1;
                                % close all
                                for block_i=1:block_max_L
                                    sig_tmp=signal_data(:,1+(block_i-1)*block_duration:block_i*block_duration)';
                                    input_DATA_tmp=input_DATA(1+(block_i-1)*block_duration:block_i*block_duration,:);
                                    if length(find(sum(abs(sig_tmp),2) < 1)) > fs || length(find(sum(abs(input_DATA_tmp),2) < 1)) > fs
                                        d_block=vertcat(d_block,block_i);
                                    elseif sum(diff(find(abs(max(input_DATA_tmp')-min(input_DATA_tmp')) < 1))==1) > size(input_DATA_tmp,1)/10
                                        d_block=vertcat(d_block,block_i);
                                    elseif ~isempty(find(max(abs(input_DATA_tmp)) > mag_threshold/2, 1))
                                        d_block=vertcat(d_block,block_i);
                                    elseif length(find(corr(sig_tmp) > corr_threshold)) > (size(sig_tmp,2)+2) && length(find(corr(input_DATA_tmp) > corr_threshold)) > (size(input_DATA_tmp,2)+2)
                                        d_block=vertcat(d_block,block_i);
                                    elseif ~isempty(find(max(abs(diff(sig_tmp))) > mag_threshold,1)) || ~isempty(find(max(abs(diff(input_DATA_tmp))) > mag_threshold/2,1))
                                        d_block=vertcat(d_block,block_i);
                                    elseif ~isempty(find(max(abs(diff(sig_tmp))) < 1,1)) || ~isempty(find(max(abs(diff(input_DATA_tmp))) < 1,1))
                                        d_block=vertcat(d_block,block_i);
                                    end
                                    clear sig_tmp corr_tmp psd f
                                end

                                if sum(d_block)~=0
                                    block_Num(d_block)=[];
                                end
                                clear d_block
                                if length(block_Num)>time_max_f
                                    block_Num=block_Num(randperm(length(block_Num),time_max_f));
                                end
                                block_L=length(block_Num);
                                if block_L > 0
                                    e_div_L=5;
                                    features=zeros(block_L,(T_D_L+channel_L*3+e_div_L+2));
                                    for block_i=1:block_L
                                        % block_i
                                        input_DATA_tmp=input_DATA(1+(block_Num(block_i)-1)*block_duration:block_Num(block_i)*block_duration,:);
                                        sig_te_out=get_TE_part(input_DATA_tmp, PartA, PartB, T_D_L,block_size);
                                        features(block_i,1:T_D_L)=sig_te_out*10; clear sig_te_out

                                        ss_NFFT = 2^(nextpow2(numel(input_DATA_tmp)));
                                        ss_X = fft(input_DATA_tmp,ss_NFFT);
                                        ss_F = fs * (linspace(0,1,ss_NFFT+1));  ss_F(end)=[];
                                        freqLim = 32;
                                        sef_ROI = z_findnear(ss_F, freqLim);
                                        goalPercentage = 95;
                                        feaSEF95 = z_sef(abs(ss_X(1:sef_ROI,:)).^2, ss_F(1:sef_ROI), goalPercentage*1e-2);
                                        features(block_i,T_D_L+1:T_D_L+channel_L)=feaSEF95(:);
                                        clear sef_ROI ss_NFFT feaSEF95

                                        winSizeSecSFS = 1; winStepSecSFS = 1;
                                        winSizeSFS = fix(fs * winSizeSecSFS);
                                        winStepSFS = fix(fs * winStepSecSFS);
                                        sfs_Nwins = fix((size(input_DATA_tmp,1)-winSizeSFS)/winStepSFS)+1; % total number of available windows
                                        sfs_windowing_ = blackman(winSizeSFS);
                                        sfs_NFFT = 2^nextpow2(winSizeSFS);
                                        sfs_F = fs * (linspace(0,1,sfs_NFFT+1)); sfs_F(end) = []; % freq. vector original
                                        for ch_i=1:size(input_DATA_tmp,2)
                                            for idxWin= 1:sfs_Nwins
                                                sig_tmp_x = input_DATA_tmp((1:winSizeSFS)+(idxWin-1)*winStepSFS,ch_i ); % extract
                                                sig_tmp_x = sig_tmp_x.*sfs_windowing_;
                                                sfs_X(:,idxWin) = fft(sig_tmp_x,sfs_NFFT)/winSizeSFS; % DFT
                                                clear sig_tmp_x
                                            end
                                            [bisp, ~] = z_bispectrums(sfs_X, sfs_F, sfs_Nwins, 1);
                                            sfs_addrs = z_findnear(sfs_F,[.5 40 47]);
                                            sfs_num = bisp(:,sfs_addrs(1):sfs_addrs(3));
                                            sfs_den = bisp(:,sfs_addrs(2):sfs_addrs(3));
                                            feaSFS(ch_i) = log(sum(sfs_num(:))/sum(sfs_den(:)));
                                            clear sfs_X bisp sfs_den sfs_num sfs_addrs
                                        end
                                        features(block_i,T_D_L+channel_L+1:T_D_L+channel_L+channel_L)=feaSFS;
                                        clear feaSFS winSizeSecSFS winSizeSFS winStepSFS sfs_Nwins
                                        clear sfs_windowing_ sfs_NFFT sfs_F

                                        freqLimL = 0.8; freqLimM = 10; freqLimH = 20;
                                        spe_ROI = z_findnear(ss_F, [freqLimL freqLimM]);
                                        Px1 = abs(ss_X( spe_ROI(1):spe_ROI(2), : )).^2;
                                        PvHgh = bsxfun( @rdivide, Px1, sum(Px1) ); % PDF
                                        spe_ROI = z_findnear(ss_F, [freqLimL freqLimH]);
                                        Px2 = abs(ss_X( spe_ROI(1):spe_ROI(2), : )).^2;
                                        PvALL = bsxfun( @rdivide, Px2, sum(Px2) ); % PDF
                                        feaReEn = z_entropy(PvALL) / log2(size(PvALL,1)); % response entropy
                                        feaStEn = z_entropy(PvHgh) / log2(size(PvHgh,1)); % state entropy
                                        features(block_i,T_D_L+channel_L+channel_L+1)=feaReEn/feaStEn;
                                        clear feaStEn feaReEn ss_F ss_X
                                        clear Px1 PvHgh spe_ROI Px2 PvALL

                                        % epoch envelope
                                        peak_th_min=20;
                                        peak_th_max=200;
                                        epoch_div_all=[0.1 0.2 0.5 1 2];
                                        % e_div_L=length(epoch_div_all);
                                        SE_div=zeros(length(epoch_div_all),1);
                                        for div_i=1:length(epoch_div_all)
                                            epoch_div=epoch_div_all(div_i);
                                            mag_max=zeros(size(input_DATA_tmp,1)/fs/epoch_div,size(input_DATA_tmp,2));
                                            for epoch_i=1:round(size(input_DATA_tmp,1)/fs/epoch_div)
                                                epoch_tmp=input_DATA_tmp(1+round(fs*(epoch_i-1)*epoch_div):round(fs*epoch_i*epoch_div),:);
                                                mag_max(epoch_i,:)=max(abs(epoch_tmp));
                                                clear epoch_tmp
                                            end
                                            SE_tmp=zeros(channel_L,1);
                                            for ch_i=1:channel_L
                                                [peak_th, ~] = z_hist_percent(mag_max(:,ch_i),0.98);
                                                if peak_th < peak_th_min
                                                    peak_th = peak_th_min;
                                                elseif peak_th > peak_th_max
                                                    peak_th = peak_th_max;
                                                end
                                                se_inp=abs(diff(round(mag_max(:,ch_i)./(round(peak_th/10)))));
                                                count_tmp=histcounts(se_inp,7,'binlimits',[0 11]);
                                                if sum(count_tmp) ~= length(se_inp)
                                                    count_tmp(end)=count_tmp(end)+(length(se_inp)-sum(count_tmp));
                                                end
                                                SE_tmp(ch_i)=(sum(count_tmp(end-2:end))/sum(count_tmp(1:2)))*100;
                                                clear count_tmp peak_th se_inp
                                            end
                                            SE_div(div_i)=max(SE_tmp);
                                            clear SE_tmp mag_max
                                        end
                                        % SE_div
                                        features(block_i,T_D_L+channel_L+channel_L+1+1:T_D_L+channel_L+channel_L+1+e_div_L)=SE_div; clear mag_max SE_div
                                        % power
                                        features(block_i,T_D_L+channel_L+channel_L+1+e_div_L+1:T_D_L+channel_L+channel_L+1+e_div_L+channel_L)=diag(cov(input_DATA_tmp)).^0.5;
                                        % data num
                                        features(block_i,T_D_L+channel_L+channel_L+1+e_div_L+channel_L+1)=str2double(recording_id(end-1:end));

                                        clear spe_ROI Px1 PvHgh spe_ROI PvALL Px2 ss_X ss_F
                                        clear sfs_F sfs_windowing_ sfs_Nwins
                                        clear winStepSFS winSizeSFS envelope_var
                                        clear input_DATA_tmp filt_inp_tmp
                                    end % block

                                    if size(features,1)~=0
                                        if sum(sum(isnan(features')))~=0
                                            features(sum(isnan(features'))>0,:)=[];
                                        end

                                        if size(features,1) > 0
                                            current_features=vertcat(current_features,features);
                                            clear features

                                            if size(current_features,1) >= max_feature_L
                                                break
                                            end
                                        end
                                    end
                                else
                                    if on_cnt==0
                                        if block_max_L > time_max_f && ~isempty(find(sum(abs(diff(input_DATA))<1)./size(input_DATA,1) > 0.8, 1))
                                            change_ch=find(sum(abs(diff(input_DATA))<1)./size(input_DATA,1) > 0.8);
                                            if ~isempty(change_ch) && length(change_ch) < (channel_L-1)
                                                if ~isempty(find(change_ch==1, 1))
                                                    channels{1}='Fz';
                                                end
                                                if ~isempty(find(change_ch==2, 1))
                                                    channels{2}='Cz';
                                                end
                                                if ~isempty(find(change_ch==3, 1))
                                                    channels{3}='Fz';
                                                end
                                                if ~isempty(find(change_ch==4, 1))
                                                    channels{4}='Cz';                                                    
                                                end
                                                while_on=1;
                                                on_cnt=1;
                                            end
                                            clear change_ch
                                        end
                                    end
                                end
                            end % block
                        end
                        clear signal_data input_DATA
                    end % while
                end
            elseif str2double(recording_id(end-2:end)) < 30
                break
            end

            clear ch_dc_noise_N dc_noise_N ch_noise_N envelope_var
            clear TE_features f_outputs corr_results features d_block
            clear max_diff_all input_DATA block_max_L block_L block_Num
            clear signal_data filtered_sig c_t_N_i recording_location sig_mag_log
        end
    end % if
end

%-----------------------------------------------------------------------------------
function [inp_data_ttmp, channels] = reduce_channels(inp_data_ttmp, channels, signal_channels)
channel_order=signal_channels(ismember(signal_channels, channels));
inp_data_ttmp=inp_data_ttmp(ismember(signal_channels, channels),:);
inp_data_ttmp=reorder_recording_channels(inp_data_ttmp, channel_order, channels);
%-----------------------------------------------------------------------------------
function reordered_signal_data=reorder_recording_channels(inp_signal_tmp, current_channels, reordered_channels)
if length(current_channels)<length(reordered_channels)
    parfor i=length(current_channels)+1:length(reordered_channels)
        current_channels{i}='';
    end
end
if sum(cellfun(@strcmp, reordered_channels, current_channels))~=length(current_channels)
    indices=[];
    parfor j=1:length(reordered_channels)
        if sum(strcmp(reordered_channels{j},current_channels))>0
            indices=[indices find(strcmp(reordered_channels{j},current_channels))];
        else
            indices=[indices nan];
        end
    end
    num_channels=length(reordered_channels);
    num_samples=size(inp_signal_tmp,2);
    reordered_signal_data=zeros(num_channels,num_samples);
    for j=1:num_channels
        if ~isnan(indices(j))
            reordered_signal_data(j,:)=inp_signal_tmp(indices(j),:);
        end
    end
else
    reordered_signal_data=inp_signal_tmp;
end
%-----------------------------------------------------------------------------------
function [rescaled_data,fs,channels]=load_recording(recording_location)
header_file=strsplit([recording_location '.hea'],'/');
header_file=header_file{end};
header=strsplit(fileread([recording_location '.hea']),'\n');
header(cellfun(@(x) isempty(x),header))=[];
header(startsWith(header,'#'))=[];
recordings_info=strsplit(header{1},' ');
fs=str2double(recordings_info{3});
num_samples=str2double(recordings_info{4});
signal_file=cell(1,length(header)-1);
gain=zeros(1,length(header)-1);
offset=zeros(1,length(header)-1);
initial_value=zeros(1,length(header)-1);
checksum=zeros(1,length(header)-1);
channels=cell(1,length(header)-1);
parfor j=2:length(header)
    header_tmp=strsplit(header{j},' ');
    signal_file{j-1}=header_tmp{1};
    gain(j-1)=str2double(header_tmp{3});
    offset(j-1)=str2double(header_tmp{5});
    initial_value(j-1)=str2double(header_tmp{6});
    checksum(j-1)=str2double(header_tmp{7});
    channels{j-1}=header_tmp{9};
end
if ~length(unique(signal_file))==1
    error('A single signal file was expected for %s',header_file)
end
load([recording_location '.mat'],'val')
num_channels=length(channels);
if num_channels~=size(val,1) || num_samples~=size(val,2)
    error('The header file %s is inconsistent with the dimensions of the signal file',header_file)
end
for j=1:num_channels
    if val(j,1)~=initial_value(j)
        error('The initial value in header file %s is inconsistent with the initial value for the channel',header_file)
    end

    if sum(val(j,:))~=checksum(j)
        error('The checksum in header file %s is inconsistent with the initial value for the channel',header_file)
    end
end
rescaled_data=zeros(num_channels,num_samples);
parfor j=1:num_channels
    rescaled_data(j,:)=(val(j,:)-offset(j))/gain(j);
end
%-----------------------------------------------------------------------------------
function [PH_cov]=get_TE_part(input_sig,PartA, PartB, T_D_L,block_size)
for band_i=1:size(input_sig,3)
    input_TE_sig=input_sig(:,:,band_i);
    parfor Delay_i = 1:T_D_L
        [PH_cov(:,Delay_i,band_i)]=z_TE_features_part(input_TE_sig, PartA, PartB, Delay_i,block_size);
    end
end
%-----------------------------------------------------------------------------------
function [PH_cov]=z_TE_features_part(input_TE_tmp, PartA, PartB, Delay,block_size)
div_NN=20;
resample_N=1:round(size(input_TE_tmp,1)/block_size/div_NN):size(input_TE_tmp,1);
input_DATA_tmp = input_TE_tmp(resample_N,:);
idxL=0;
for idx = 1:size(PartA,1)
    for idy=1:size(PartA{idx},1)
        idxL =  idxL+1;
    end
end
change_idx=zeros(idxL,1);
m = 0;
for idx = 1:size(PartA,1)
    for idy=1:size(PartA{idx},1)
        m =  m+1;
        for idz=1:size(PartA{idx},2)
            A = PartA{idx}{idy,idz};
            B = PartB{idx}{idy,idz};
            TE_AB=z_TE_v2(input_DATA_tmp, A, B, Delay);
            TE_BA=z_TE_v2(input_DATA_tmp, B, A, Delay);
            if TE_AB>TE_BA
                cov_TE{m,1}(idz,1)=TE_AB;
                change_idx(m)=0;
            else
                cov_TE{m,1}(idz,1)=TE_BA;
                change_idx(m)=1;
            end
        end
    end
end
mins_cov=zeros(idxL,1);
parfor k=1:idxL
    mins_cov(k) = min(cov_TE{k});
end
[PH_cov,~] = max(mins_cov);
%-----------------------------------------------------------------------------------
function TE = z_TE_v2(te_inp_data, snd_ch, rcv_ch, Delay)
D=Delay; K=1; L=1;
snd_L=length(snd_ch);
rcv_L=length(rcv_ch);
snd = te_inp_data(:,snd_ch);
rcv_tmp = te_inp_data(:,rcv_ch);
rcv=[zeros(1,rcv_L); rcv_tmp(1:end-1,:)];
if D > 0, sndDelayed = [nan(D,snd_L); snd(1:end-D,:)];
elseif D == 0, sndDelayed = [nan(1+D,snd_L); snd(2+D:end,:)];
end
rcvf = rcv_tmp;
S = [rcvf, z_hk(rcv, K), z_hk(sndDelayed, L)];
S(any(isnan(S),2),:)=[];
TE=z_TE_calc_v2(S, rcv_L, snd_L);
%-----------------------------------------------------------------------------------
function H = z_hk(rcv,K)
if ~isvector(rcv)
    [nr,nc] = size(rcv);
    if nc > nr
        rcv = rcv';
    end
    tmp = cell(1,nc);
    parfor idx=1:nc
        tmp{idx} = z_hk(rcv(:,idx), K );
    end
    H = cell2mat(tmp);
else
    if K >= 2
        rcv = rcv(:);
        nc = length(rcv);
        r = nan(1,K);
        r = r(:);
        nr = length(r);
        x = [ r((2:nr)'); rcv ];
        cidx = (ones(class(rcv)):nc)';
        ridx = zeros(class(r)):(nr-1);
        H = cidx(:,ones(nr,1)) + ridx(ones(nc,1),:);
        H(:) = x(H);
    elseif K==1
        H = rcv(:);
    else
        error('K must be greater than or equals to 1');
    end
end
%-----------------------------------------------------------------------------------
function [TE]= z_TE_calc_v2(S,rcv_L, snd_L)
S_rcvf=1:rcv_L;
S_rcv=(1+rcv_L):rcv_L*2;
S_snd=(1+rcv_L*2):(rcv_L*2+snd_L);
COVxx=cov(S);
COV_rcv = COVxx(S_rcv,S_rcv);
COV_rcvf_rcv = COVxx([S_rcvf S_rcv],[S_rcvf S_rcv]);
COV_rcvf_rcv_snd = COVxx([S_rcvf S_rcv S_snd],[S_rcvf S_rcv S_snd]);
COV_rcv_snd= COVxx([S_rcv S_snd],[S_rcv S_snd]);
H_rcvf_rcv = real(log((2*pi*exp(1))^length(COV_rcvf_rcv)*det(COV_rcvf_rcv))/2);
H_rcv = real(log((2*pi*exp(1))^length(COV_rcv)*det(COV_rcv))/2) ;
H_rcvf_rcv_snd = real(log((2*pi*exp(1))^length(COV_rcvf_rcv_snd)*det(COV_rcvf_rcv_snd))/2);
H_rcv_snd = real(log((2*pi*exp(1))^length(COV_rcv_snd)*det(COV_rcv_snd))/2);
TE = H_rcvf_rcv - H_rcv - H_rcvf_rcv_snd + H_rcv_snd;
%-----------------------------------------------------------------------------------
function [addr, val] = z_findnear(X, A)
a = zeros(size(A));
for k=1:length(A)
    tmp = abs(X - A(k));
    [~, a(k)] = min(tmp);
end
addr = a;
if nargout == 2
    val = X(a);
end
%-----------------------------------------------------------------------------------
function pef = z_sef( P, F, value )
if value < 0 || value > 1
    error('Check the range.');
end
RLP = bsxfun(@rdivide, P, sum(P)); % POWER NORMALIZING
CRLP = cumsum(RLP); % POWER CUMULATION
BINARY = ones(size(CRLP));
BINARY(CRLP < value ) = 0;
EDGE = [zeros(1,size(BINARY,2)); diff(BINARY)];
EDGE(1,sum(EDGE)==0)=1;
[ADDR,~] = ind2sub(size(EDGE), find(EDGE));
pef = F(ADDR);
%-----------------------------------------------------------------------------------
function z = z_entropy(inp_stream,base)
if nargin < 2
    base = 2;
end
inp_stream = inp_stream(:);
adr = inp_stream > 0;
if base == 2
    z = -1 * inp_stream(adr)'*log2(inp_stream(adr));
elseif strcmp(base,'nat')
    z = -1 * inp_stream(adr)'*log(inp_stream(adr));
else
    error('base!');
end

%-----------------------------------------------------------------------------------
function [bisp, Flet] = z_bispectrums(inp_XX, F, winSize_, winStep_)
NF = size(inp_XX,1)/2;
Flet = F(1:numel(F)/2);
[F1,F2] = meshgrid(1:NF, 1:NF);
B = zeros(NF, NF, size(inp_XX,2));
for idxWin = 1:size(inp_XX,2)
    sigWin = inp_XX(:,idxWin);
    B(:,:,idxWin) =  sigWin(F1) .* sigWin(F2) .* conj(sigWin(F1+F2-1)) ;
end
Nwins = fix((size(inp_XX,2)-winSize_)/winStep_)+1;
bisp = zeros(NF, NF, Nwins);
for idxWin= 1:Nwins
    tmp = B(:,:,(1:winSize_)+(idxWin-1)*winStep_);
    tmp2 = abs(mean(tmp,3));
    bisp(:,:,idxWin) = triu(tmp2).*fliplr(triu(ones(size(tmp2))));
end

%-----------------------------------------------------------------------------------
function [peak_th, SE_bin] = z_hist_percent(hist_sig_inp,band_thre_rate)
min_rate=1/100; % 0.1%
max_rate=99/100; % 99.9%
hist_thre_bin=linspace(min(hist_sig_inp),max(hist_sig_inp),100);
hist_thre_point=histcounts(hist_sig_inp,hist_thre_bin);
hist_thre_point_cum=cumsum(hist_thre_point);
if sum(find(hist_thre_point_cum<sum(hist_thre_point)*min_rate))==0
    min_threshold=hist_thre_bin(1);
else
    min_threshold=hist_thre_bin(find(hist_thre_point_cum<sum(hist_thre_point)*min_rate, 1, 'last' )); % min
end
if sum(find(hist_thre_point_cum>sum(hist_thre_point)*max_rate))==0
    max_threshold=hist_thre_bin(end);
else
    max_threshold=hist_thre_bin(find(hist_thre_point_cum>sum(hist_thre_point)*max_rate, 1 )); % max
end
if min_threshold>0
    SE_bin(1)=min_threshold*0.8; SE_bin(2)=max_threshold*1.2;
    hist_peak_bin=linspace(SE_bin(1),SE_bin(2),100);
else
    SE_bin(1)=min_threshold*1.2; SE_bin(2)=max_threshold*1.2;
    hist_peak_bin=linspace(SE_bin(1),SE_bin(2),100);
end
hist_peak_point=histcounts(hist_sig_inp,hist_peak_bin);
hist_peak_point_cum=cumsum(hist_peak_point);
if sum(find(hist_peak_point_cum>sum(hist_peak_point)*band_thre_rate))==0
    peak_th=hist_peak_bin(end);
else
    peak_th=hist_peak_bin(find(hist_peak_point_cum>sum(hist_peak_point)*band_thre_rate, 1 )); % max
end
%-----------------------------------------------------------------------------------


%-----------------------------------------------------------------------------------


%-----------------------------------------------------------------------------------


%-----------------------------------------------------------------------------------







