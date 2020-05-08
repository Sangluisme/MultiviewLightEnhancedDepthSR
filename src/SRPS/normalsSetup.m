function [xx, yy, Gx, Gy, imask] = normalsSetup(mask, K)

%generate grid,mask and substract principal point
[xx, yy] = meshgrid( 0:size(mask,2) - 1 , 0:size(mask,1) - 1 );

xx = xx(mask);
yy = yy(mask);

xx = xx - K(1,3);
yy = yy - K(2,3);


%calculate gradient matrix of valid pixels
G = make_gradient(mask);
Gx = G(1:2:end-1,:);
Gy = G(2:2:end,:);

%get position of mask
imask = find(mask>0);

end

