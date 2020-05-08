classdef ClassTraj < handle
    properties
        assoc_path = [];
        dataFolder;
        rgbFolder;
        depthFolder;
        refFrame = 1;
        initial_traj = []; 
        final_traj = []; 
        num_image = 20;
    end
    methods
        function obj = ClassTraj(dataFolder, rgbFolder, depthFolder)
            if nargin == 1
                rgbFolder = strcat(dataFolder, 'rgb');
                depthFolder = strcat(dataFolder, 'depth');

            elseif nargin == 0
                error('set a data folder!');
            end
            obj.dataFolder = dataFolder;
            obj.rgbFolder = rgbFolder;
            obj.depthFolder = depthFolder;
            obj.initial_traj = strcat(dataFolder,'initial_traj.txt');
            obj.final_traj = strcat(dataFolder,'final_traj.txt');
        end
        
        function generate_assoc_file(obj)
            fileDepth = fullfile(obj.depthFolder, '*.png');
            depthFiles   = dir(fileDepth);

            fileRgb = fullfile(obj.rgbFolder, '*.png');
            rgbFiles = dir(fileRgb);
            
            if length(depthFiles)~=length(rgbFiles)
                error('depth map and color images do not match!')
            end
            fid = fopen(strcat(obj.dataFolder,'/','assocFile.txt'),'w');
            for k = 1: length(depthFiles)

                rgb = strcat('rgb/',rgbFiles(k).name);
                depth = strcat('depth/',depthFiles(k).name);
                fprintf(fid, '%f %s %f %s\n', k, rgb,k, depth);
            end
            fclose(fid);
            obj.assoc_path = fullfile(obj.dataFolder, 'assocFile.txt');
        end
        
        function [first_half_traj, second_half_traj] = updated_traj_file(obj)
            [pose_est] = ReadTraj(obj.final_traj);
            T1 = pose_est(:,:,obj.refFrame);
            pose_update = [];
            for i =1:size(pose_est,3)
                T = T1\pose_est(:,:,i);
                pose_update = cat(3,pose_update,T);
            end

            first_half = pose_update(:,:,1:obj.refFrame);
            first_half = flip(first_half,3);
            first_half_traj = WriteTraj(first_half);

            second_half = pose_update(:,:,obj.refFrame:end);
            second_half_traj = WriteTraj(second_half);

        end
        
        
        function generate_assoc_file_backward(obj)
            fileDepth = fullfile(obj.depthFolder, '*.png');
            depthFiles   = dir(fileDepth);

            fileRgb = fullfile(obj.rgbFolder, '*.png');
            rgbFiles = dir(fileRgb);
            
            if length(depthFiles)~=length(rgbFiles)
                error('depth map and color images do not match!')
            end
            fid = fopen(strcat(obj.dataFolder,'/','assocFile.txt'),'w');
            for k = obj.refFrame:-1: 1
                %timestamp = k;
%                 if k < 10
%                     rgb = strcat('rgb/00',num2str(k),'.png');
%                     depth = strcat('depth/00',num2str(k), '.png');
%                 else
%                     rgb = strcat('rgb/0',num2str(k),'.png');
%                     depth = strcat('depth/0',num2str(k), '.png');
%                 end
                rgb = strcat('rgb/',rgbFiles(k).name);
                depth = strcat('depth/',depthFiles(k).name);
                fprintf(fid, '%f %s %f %s\n', k, rgb,k, depth);
            end
            fclose(fid);
        end
        
        function update_assoc_file(obj)
            if ~isempty(obj.assoc_path)
                readID = fopen(obj.assoc_path, 'r');
                tempfile = strcat(obj.dataFolder,'/','assocFiletemp.txt'); %
                writeID = fopen(tempfile,'w');

                for i =1:obj.refFrame-1
                    fgetl(readID); %skip line 1
                end

                fwrite(writeID, fread(readID));

                fclose(readID);
                fclose(writeID);
                delete(obj.assoc_path);
                movefile(tempfile, obj.assoc_path);
            else
                error('generate assoc file first!');
            end
        end
        
        function saveInitialTrajFile(obj, Traj)
            filename = strcat(obj.dataFolder,'/','initial_traj.txt');
            writeID = fopen(filename,'w');
            fprintf(writeID, '%f %f %f %f %f %f %f %f\n', Traj');
        end
        
        

        
        function [pose] = DVOPoseEst(obj,K)
            fileID = fopen(obj.assoc_path);
            if fileID ~= -1
                pose=DVO_mexMEX(obj.dataFolder, K);  
            else
                error('wrong assoc file path!');
            end
           %movefile(strcat(obj.dataFolder,'traj.txt'), obj.final_traj);
        end                      
    end
end


function [Traj] = WriteTraj(pose)
    Traj = zeros(size(pose,3),8);
    for i = 1:size(pose,3)
        T = pose(:,:,i);
        Traj(i,1)=i;
        Traj(i,2:4) = T(1:3,4)';
        quat = rotm2quat(T(1:3,1:3));
        Traj(i,5:end-1) = quat(2:end);
        Traj(i,end) = quat(1);
    end
end            

function [pose] = ReadTraj(initial_traj)
    T = importdata(initial_traj);
    pose = repmat(eye(4),[1,1,size(T,1)]);
    for i = 1:size(T,1)
        translation = T(i,2:4);
        quat = T(i,5:end-1);
        quat = [T(i,8),quat];
        pose(1:3,1:3,i) = quat2rotm(quat);
        pose(1:3,4,i) = translation';
    end
end
