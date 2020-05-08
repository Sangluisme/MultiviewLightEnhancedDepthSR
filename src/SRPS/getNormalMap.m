function [N_normalized, dz, zx, zy] = getNormalMap(z, zx, zy, K, xx, yy)
%variable explanation:
% z the depth as a vector
% zx = Dx*z\in Nx1, i.e. a vector;
% zy = Dy*z\in Nx1, i.e. a vector;
% K the intrinsic matrix
% xx and yy is the meshgrid as vector, where principal point is already
% taken into account, i.e. xx = xx - K(1,3) & yy = y - K(2,3)

%%
%get number of pixel in vector
nPix = size(z,1);

% get unnormalized normals
N_normalized = zeros(nPix,3);
N_normalized(:,1) = K(1,1) * zx;
N_normalized(:,2) = K(2,2) * zy;
N_normalized(:,3) = ( -z -xx .* zx - yy .* zy );

% get normalizing constant
%normal_norm = sqrt( sum( N_normalized .^ 2, 2)  );
                
%dz = max(eps,normal_norm);
dz = max(eps,sqrt((K(1,1)*zx).^2+(K(2,2)*zy).^2+(-z-xx.*zx-yy.*zy).^2));
% normalize normals
N_normalized = bsxfun(@times, N_normalized, 1./dz);

end