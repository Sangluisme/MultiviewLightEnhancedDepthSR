classdef InputObject < handle
    properties
        s = [];
        I;
        z_ls;
        z_hs = [];
        mask;
        K;
        rho = [];
    end
    methods
        function obj = InputObject(I,z_ls,mask,K)
            obj.I = I;
            obj.z_ls = z_ls;
            obj.mask = mask;
            obj.K = K;
            num_image = size(I,4);
            scale_factor = size(I,1)/size(z_ls,1);
            if isempty(obj.s)
                obj.s = repmat([0,0,-1,0.02], [3,1,num_image]);
            end
            if isempty(obj.z_hs)
                obj.z_hs = imresize(mean(z_ls,3), scale_factor,'bicubic');
            end
            if isempty(obj.rho)
                obj.rho = 0.5*ones(size(I,1),size(I,2),size(I,3));
            end
        end
    end
end

            
            