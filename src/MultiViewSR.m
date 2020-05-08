function [z, rho, N, s, pose_est] = MultiViewSR(dataFolder,dataset, params)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The implementation of the paper:                               %
% "Inferring Super-Resolution Depth from a Moving Light-Source   %
% Enhanced RGB-D Sensor: a Variational Approach"                 %    
% Lu Sang, Bjoern Haefner, Daniel Cremers                        %
%                                                                %
% The code can only be used for research purposes.               %
%                                                                %
% Computer Vision Group, TUM                                     %
% Author: Lu Sang (2019)                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    % correct the data folder name
    if dataFolder(end)~= '/'
        dataFolder = strcat(dataFolder,'/');
    end
    
    % get the data 
    load(strcat(dataFolder,'data/',dataset,'.mat'));
    scale_factor = params.SF;   
    if scale_factor == 2
        z_input = data.SF2.z;
    elseif scale_factor == 4
        z_input = data.SF4.z;
    else
        error("wrong Scale factor!");
    end
    masks = data.mask;
    % some constant
    num_image = size(data.I,4);    
    begining_level = 5;
    refFrame = params.refFrame;
    realdata = params.realdata;
    K = data.K;
    
    % some parameters setting
    params.apply_smooth_filter = realdata;
    params.harmonic_lighting = realdata;
    params.begining_level = begining_level;
    
    for level = begining_level:-1:1
 
        % create folder for pose estimation and down scale images
        DVOdepth = strcat(dataFolder,'depth');
        DVOrgb = strcat(dataFolder,'rgb');
        
        % check for if there is another folder has the same name 
        if (level == begining_level && 7==exist(DVOdepth,'dir'))
            rmdir(DVOdepth,'s');
        end
        
        if (level == begining_level && 7==exist(DVOrgb,'dir'))
            rmdir(DVOrgb, 's');
        end
            
              
        [masksd, Kd] = ImgResize(dataFolder,data,params,level);
        fileDepth = fullfile(DVOdepth, '*.png');
        depthFiles   = dir(fileDepth);

        fileRgb = fullfile(DVOrgb, '*.png');
        rgbFiles = dir(fileRgb);
        
        % initialize traj object
        Traj = ClassTraj(dataFolder,DVOrgb,DVOdepth);
               
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% pose estimation for initialization %%%%%%%%%%%%%%%%%%%%%
        if level == begining_level
           Traj.generate_assoc_file();
           A=Traj.DVOPoseEst(Kd);
           [pose_est] = permute(reshape(A(:,1:num_image),[4,4,num_image]),[2 1 3]);
           movefile(strcat(Traj.dataFolder,'traj.txt'), Traj.final_traj);
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% pose estimation fot cuurent refFrame %%%%%%%%%%%%%%%%%%%%%%%%%
        Traj.generate_assoc_file();
        Traj.refFrame = refFrame;
        Traj.update_assoc_file();
       
        if (exist(Traj.final_traj,'file'))                   
           [~, second_half_traj] = Traj.updated_traj_file();
           Traj.saveInitialTrajFile(second_half_traj);
        end
        A=Traj.DVOPoseEst(Kd);
       [pose_est_half2] = permute(reshape(A(:,2:num_image-refFrame+1),[4,4,num_image-refFrame]),[2 1 3]);
        if exist(Traj.initial_traj,'file')
           delete(Traj.initial_traj);            
        end   
        
        Traj.generate_assoc_file_backward(); 
        
       if (exist(Traj.final_traj,'file'))
           [first_half_traj, ~] = Traj.updated_traj_file();
           Traj.saveInitialTrajFile(first_half_traj);
        end
        A=Traj.DVOPoseEst(Kd);
        [pose_est_half] = permute(reshape(A,[4,4,refFrame]),[2 1 3]);
        pose_est_half = flip(pose_est_half,3);
       if exist(Traj.initial_traj,'file')
           delete(Traj.initial_traj);            
       end
       
        pose_est = cat(3,pose_est_half, pose_est_half2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if level == begining_level
            DRef = double(imread(strcat(DVOdepth,'/', depthFiles(refFrame).name)))/1000;
            DRef(DRef==0)=NaN;
            DRef = inpaint_nans(DRef);           
            z0 = DRef;
        else
            % use the refined depth from last level
            DRef = z;
            [DRef] = ImageDilateResize(DRef, mask);
            
        end
        IRef = im2double(imread(strcat(DVOrgb,'/', rgbFiles(refFrame).name))); 
        s0 = repmat([0, 0,-1, 0.02],[3,1,num_image]);      
        mask = logical(masksd(:,:,refFrame));
        
        I0= [];
       
        for i = 1:num_image
            T = pose_est(:,:,i);            
            I = im2double(imread(strcat(DVOrgb,'/', rgbFiles(i).name)));
            
            [I_new] = FindCorrespondingPix(DRef, I, inv(T), Kd, mask);
            rmse = calcRmse(IRef, I_new, mask);
            fprintf('error for origin  image warping %d to %d is %f:\n', i-1, i, rmse);
            
            mask3d = repmat(mask,1,1,size(I,3));
            I_new(~mask3d)=0;            
            I0 = cat(4, I0, I_new);
            s0(:,1:3,i) = transpose(T(1:3,1:3)*s0(:,1:3,i)');
            
        end
        inputs = InputObject(I0,z0,mask,Kd);
        inputs.s = s0; 
        if level == begining_level            
           inputs.rho = I0(:,:,:,refFrame);
        else       
            [rho] = ImageDilateResize(rho, mask);
            params.apply_smooth_filter = realdata;
            params.harmonic_lighting = 0;
            inputs.z_hs = DRef;
            inputs.rho = rho;            
        end           
            [z, rho, N, s] = MultiViewPS(inputs, params); 
            


        % save the refined depth for later pose estiamtion
        z(~mask)=DRef(~mask); 
        z0 = double(imread(strcat(DVOdepth,'/', depthFiles(refFrame).name)))/1000;
        imwrite(uint16(z*1000),strcat(DVOdepth,'/', depthFiles(refFrame).name));               
       
        %%%%%%%%%%%%% use the refined depth to estimate psoe again %%%%%%%%
        if level ~= 1
            Traj.generate_assoc_file();
            Traj.refFrame = 1;
            if (exist(Traj.final_traj,'file')) 
                movefile(Traj.final_traj, Traj.initial_traj);
            end
           A=Traj.DVOPoseEst(Kd);
           [pose_est] = permute(reshape(A(:,1:num_image),[4,4,num_image]),[2 1 3]);
           movefile(strcat(dataFolder,'traj.txt'), Traj.final_traj); 
           if (exist(Traj.initial_traj,'file')) 
                delete(Traj.initial_traj);
           end
        end
       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
    end
    figure
    subplot('Position', [0.05, 0.02, 0.6/2, 0.8/2]);
    D = z_input(:,:,refFrame);
    [nrows, ncols] = size(D);
    factor = 1/(2^(scale_factor-1));
    imShow('depth3d',D,imresize(mask,factor),diag([factor,factor,1])*K);title(sprintf('input depth (%d * %d)',nrows,ncols));
        
    subplot('Position', [0.4, 0.02, 0.6, 0.8])
    [nrows, ncols] = size(z);
    imShow('depth3d',z, mask, K); title(sprintf('refined SR depth (%d * %d)',nrows,ncols));
    drawnow;
    
    rmdir(DVOdepth,'s');
    rmdir(DVOrgb,'s');
    delete(strcat(dataFolder,'traj.txt')); 
    delete(strcat(dataFolder,'assocFile.txt'));
    delete(Traj.final_traj);

end
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [D] = ImageDilateResize(DRef, mask)
    stel = [0,0,1,0,0;0,1,1,1,0;1,1,1,1,1;0,1,1,1,0;0,0,1,0,0];
    nchannels = size(DRef,3);
    
    if nchannels > 1
        D_ = zeros(size(DRef));
        for ch = 1:nchannels
            DRef_ = DRef(:,:,ch);
            mask = (DRef_~=0);
            D1 = imdilate(DRef_,stel);
            D1(mask) = DRef_(mask);
            D_(:,:,ch) = D1;
        end
    else
        DRef(~mask) = 0;
        D_ = imdilate(DRef,stel);
        D_(mask) = DRef(mask);
        D_(D_==0) = NaN;
    end
    D = imresize(D_,2,'bilinear');
end        
    
    

