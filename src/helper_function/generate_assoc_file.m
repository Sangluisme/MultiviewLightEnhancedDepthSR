function generate_assoc_file(dataFolder, depthFolder, rgbFolder)
    fileDepth = fullfile(depthFolder, '*.png');
    depthFiles   = dir(fileDepth);

    fileRgb = fullfile(rgbFolder, '*.png');
    rgbFiles = dir(fileRgb);

    fid = fopen(strcat(dataFolder,'/','assocFile.txt'),'w');
    for k = 1: length(depthFiles)
        %timestamp = k;
%         if k < 10
             rgb = strcat('rgb/',rgbFiles(k).name);
             depth = strcat('depth/',depthFiles(k).name);
%         else
%             rgb = strcat('rgb/0',num2str(k),'.png');
%             depth = strcat('depth/0',num2str(k), '.png');
%         end
        fprintf(fid, '%f %s %f %s\n', k,rgb ,k,depth );
        
    end
    fclose(fid);
end