function [masksd, Kd] = ImgResize(dataFolder, data, params, level)
    I0 = data.I;
    masks = data.mask;
    K = data.K;
    realdata = params.realdata;
    if params.SF == 2
    z0 = data.SF2.z;
    else
    z0 = data.SF4.z;
    end
    scale_factor = params.SF;        
    
    destination_rgb = strcat(dataFolder,'rgb');
    destination_depth = strcat(dataFolder,'depth');
    
    if 7~=exist(destination_rgb,'dir')
        mkdir(dataFolder,'rgb');
    end
    
    if 7~=exist(destination_depth,'dir')
        mkdir(dataFolder,'depth');
    end
    
    masksd = [];
    level_factor = 1/(2^(level-1));
    for i =1:size(I0,4)
        I = imresize(I0(:,:,:,i),level_factor);
        D = z0(:,:,i);
        imwrite(I, strcat(destination_rgb,'/',num2str(i,'%03.f'),'.png'));        
        [maskd] = imresize(masks(:,:,i), level_factor, 'nearest');
        masksd = cat(3,masksd, maskd);       
        
        if ~realdata
            background = max(D(:)); 
            mask = (D~=background);
            if (level-scale_factor+1)>0 
                [D, ~] = downscaleDepth(D, mask, level-scale_factor+1);
                else           
                [D, ~] = upscaleDepth(D, mask, abs(level-scale_factor-1));
            end
            D(~maskd) = background;
        else
            if (level-scale_factor+1)>0 
                [D] = downscaleDepthRD(D, level-scale_factor+1);
            else           
                [D] = imresize(D, size(I,1)/size(D,1));
            end
        end
            
        imwrite(uint16(D*1000), strcat(destination_depth,'/',num2str(i,'%03.f'),'.png'));
    end  
    [Kd] = downscaleK(K, level);
end
     
     
     
     
 function [Dd, maskd] = downscaleDepth(D, mask, level)
    if(level <= 1)
        % coarsest pyramid level       
        Dd = D;       
        maskd = mask;
        return;
    end
    
    %max_num = max(D(:));
    D_background = imresize(D,0.5,'nearest');
    D(~mask) = 0;
    
    if mod(size(D,1),2)==1
        % downscale depth map
        DdCountValid = (sign(D(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(D(1+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(D(0+(1:2:end-1), 1+(1:2:end))) + ...
                    sign(D(1+(1:2:end-1), 1+(1:2:end))));
        Dd = (D(0+(1:2:end-1), 0+(1:2:end)) + ...
            D(1+(1:2:end-1), 0+(1:2:end)) + ...
            D(0+(1:2:end-1), 1+(1:2:end)) + ...
            D(1+(1:2:end-1), 1+(1:2:end))) ./ DdCountValid;
        Dd(isnan(Dd)) = 0;
        
        maskdCountValid = (sign(mask(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(mask(1+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(mask(0+(1:2:end-1), 1+(1:2:end))) + ...
                    sign(mask(1+(1:2:end-1), 1+(1:2:end))));
        maskd = (mask(0+(1:2:end-1), 0+(1:2:end)) + ...
            mask(1+(1:2:end-1), 0+(1:2:end)) + ...
            mask(0+(1:2:end-1), 1+(1:2:end)) + ...
            mask(1+(1:2:end-1), 1+(1:2:end))) ./ maskdCountValid;
        maskd(isnan(maskd))=0;
    elseif mod(size(D,2),2)==1
     % downscale depth map
        DdCountValid = (sign(D(0+(1:2:end), 0+(1:2:end-1))) + ...
                    sign(D(1+(1:2:end), 0+(1:2:end-1))) + ...
                    sign(D(0+(1:2:end), 1+(1:2:end-1))) + ...
                    sign(D(1+(1:2:end), 1+(1:2:end-1))));
        Dd = (D(0+(1:2:end), 0+(1:2:end-1)) + ...
            D(1+(1:2:end), 0+(1:2:end-1)) + ...
            D(0+(1:2:end), 1+(1:2:end-1)) + ...
            D(1+(1:2:end), 1+(1:2:end-1))) ./ DdCountValid;
        Dd(isnan(Dd)) = 0;
        
        maskdCountValid = (sign(mask(0+(1:2:end), 0+(1:2:end-1))) + ...
                sign(mask(1+(1:2:end), 0+(1:2:end-1))) + ...
                sign(mask(0+(1:2:end), 1+(1:2:end-1))) + ...
                sign(mask(1+(1:2:end), 1+(1:2:end-1))));
            
        maskd = (mask(0+(1:2:end), 0+(1:2:end-1)) + ...
            mask(1+(1:2:end), 0+(1:2:end-1)) + ...
            mask(0+(1:2:end), 1+(1:2:end-1)) + ...
            mask(1+(1:2:end), 1+(1:2:end-1))) ./ maskdCountValid;
        maskd(isnan(maskd))=0;
    else
        % downscale depth map
        DdCountValid = (sign(D(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(D(1+(1:2:end), 0+(1:2:end))) + ...
                    sign(D(0+(1:2:end), 1+(1:2:end))) + ...
                    sign(D(1+(1:2:end), 1+(1:2:end))));
        Dd = (D(0+(1:2:end), 0+(1:2:end)) + ...
            D(1+(1:2:end), 0+(1:2:end)) + ...
            D(0+(1:2:end), 1+(1:2:end)) + ...
            D(1+(1:2:end), 1+(1:2:end))) ./ DdCountValid;
        Dd(isnan(Dd)) = 0;
        
        
        maskdCountValid = (sign(mask(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(mask(1+(1:2:end), 0+(1:2:end))) + ...
                    sign(mask(0+(1:2:end), 1+(1:2:end))) + ...
                    sign(mask(1+(1:2:end), 1+(1:2:end))));
        maskd = (mask(0+(1:2:end), 0+(1:2:end)) + ...
            mask(1+(1:2:end), 0+(1:2:end)) + ...
            mask(0+(1:2:end), 1+(1:2:end)) + ...
            mask(1+(1:2:end), 1+(1:2:end))) ./ maskdCountValid;       
        maskd(isnan(maskd))=0;
    end
    
    Dd(~maskd) = D_background(~maskd);
   [Dd, maskd] = downscaleDepth( Dd, maskd, level - 1);    
 end

function [Dd] = downscaleDepthRD(D, level)
    if(level <= 1)
        % coarsest pyramid level       
        Dd = D;       
        return;
    end
    
    if mod(size(D,1),2)==1
        % downscale depth map
        DdCountValid = (sign(D(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(D(1+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(D(0+(1:2:end-1), 1+(1:2:end))) + ...
                    sign(D(1+(1:2:end-1), 1+(1:2:end))));
        Dd = (D(0+(1:2:end-1), 0+(1:2:end)) + ...
            D(1+(1:2:end-1), 0+(1:2:end)) + ...
            D(0+(1:2:end-1), 1+(1:2:end)) + ...
            D(1+(1:2:end-1), 1+(1:2:end))) ./ DdCountValid;
        Dd(isnan(Dd)) = 0;
        
        
        
    elseif mod(size(D,2),2)==1
     % downscale depth map
        DdCountValid = (sign(D(0+(1:2:end), 0+(1:2:end-1))) + ...
                    sign(D(1+(1:2:end), 0+(1:2:end-1))) + ...
                    sign(D(0+(1:2:end), 1+(1:2:end-1))) + ...
                    sign(D(1+(1:2:end), 1+(1:2:end-1))));
        Dd = (D(0+(1:2:end), 0+(1:2:end-1)) + ...
            D(1+(1:2:end), 0+(1:2:end-1)) + ...
            D(0+(1:2:end), 1+(1:2:end-1)) + ...
            D(1+(1:2:end), 1+(1:2:end-1))) ./ DdCountValid;
        Dd(isnan(Dd)) = 0;
       
       
    else
        % downscale depth map
        DdCountValid = (sign(D(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(D(1+(1:2:end), 0+(1:2:end))) + ...
                    sign(D(0+(1:2:end), 1+(1:2:end))) + ...
                    sign(D(1+(1:2:end), 1+(1:2:end))));
        Dd = (D(0+(1:2:end), 0+(1:2:end)) + ...
            D(1+(1:2:end), 0+(1:2:end)) + ...
            D(0+(1:2:end), 1+(1:2:end)) + ...
            D(1+(1:2:end), 1+(1:2:end))) ./ DdCountValid;
        Dd(isnan(Dd)) = 0;
        
      
    end
     [Dd] = downscaleDepthRD( Dd, level - 1);    
 end

 function [Kd] = downscaleK(K, level)
    if(level <= 1)     
        Kd = K;      
        return;
    end
    % downscale camera intrinsics
    % this is because we interpolate in such a way, that 
    % the image is discretized at the exact pixel-values (e.g. 3,7), and
    % not at the center of each pixel (e.g. 3.5, 7.5).
    Kd = [K(1,1)/2, 0, (K(1,3)+0.5)/2-0.5;
            0, K(2,2)/2, (K(2,3)+0.5)/2-0.5;
            0, 0, 1];

    [Kd] = downscaleK(Kd, level - 1);  
 end
 
 function [maskd] = downscaleMask(mask, level)
    if(level <= 1)
        % coarsest pyramid level           
        maskd = mask;
        return;
    end

    if mod(size(mask,1),2)==1
       maskdCountValid = (sign(mask(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(mask(1+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(mask(0+(1:2:end-1), 1+(1:2:end))) + ...
                    sign(mask(1+(1:2:end-1), 1+(1:2:end))));
        maskd = (mask(0+(1:2:end-1), 0+(1:2:end)) + ...
            mask(1+(1:2:end-1), 0+(1:2:end)) + ...
            mask(0+(1:2:end-1), 1+(1:2:end)) + ...
            mask(1+(1:2:end-1), 1+(1:2:end))) ./ maskdCountValid;
        maskd(isnan(maskd))=0;
    elseif mod(size(mask,2),2)==1
        
        maskdCountValid = (sign(mask(0+(1:2:end), 0+(1:2:end-1))) + ...
                sign(mask(1+(1:2:end), 0+(1:2:end-1))) + ...
                sign(mask(0+(1:2:end), 1+(1:2:end-1))) + ...
                sign(mask(1+(1:2:end), 1+(1:2:end-1))));
            
        maskd = (mask(0+(1:2:end), 0+(1:2:end-1)) + ...
            mask(1+(1:2:end), 0+(1:2:end-1)) + ...
            mask(0+(1:2:end), 1+(1:2:end-1)) + ...
            mask(1+(1:2:end), 1+(1:2:end-1))) ./ maskdCountValid;
        maskd(isnan(maskd))=0;
    else
        % downscale depth map 
        maskdCountValid = (sign(mask(0+(1:2:end-1), 0+(1:2:end))) + ...
                    sign(mask(1+(1:2:end), 0+(1:2:end))) + ...
                    sign(mask(0+(1:2:end), 1+(1:2:end))) + ...
                    sign(mask(1+(1:2:end), 1+(1:2:end))));
        maskd = (mask(0+(1:2:end), 0+(1:2:end)) + ...
            mask(1+(1:2:end), 0+(1:2:end)) + ...
            mask(0+(1:2:end), 1+(1:2:end)) + ...
            mask(1+(1:2:end), 1+(1:2:end))) ./ maskdCountValid;       
        maskd(isnan(maskd))=0;
    end
 
   [maskd] = downscaleMask( maskd, level - 1);    
 end

function [Du,masku] = upscaleDepth(D, mask, level)
     if(level <= 1)    
        Du = D; 
        masku = mask;
        return;
     end
    
     [Du] = ImageDilateResize(D, mask);
     masku = imresize(mask,2,'nearest');
     
     [Du, masku] = upscaleDepth( Du, masku, level - 1);   
     
end

function [D] = ImageDilateResize(DRef, mask)
    stel = [0,0,1,0,0;0,1,1,1,0;1,1,1,1,1;0,1,1,1,0;0,0,1,0,0];
    nchannels = size(DRef,3);
    
    if nchannels > 1
        D_ = zeros(size(DRef));
        %mask3d = (DRef~=0);
        %DRef(~mask3d)=0;
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

    
        