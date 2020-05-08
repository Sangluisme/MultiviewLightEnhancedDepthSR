function [z_out, rho_out, N_out, s] = MultiViewPS(inputs, parameters)
% Implementation of the paper "Inferring Super-Resolution Depth from a Moving Light-Source   
% Enhanced RGB-D Sensor: a Variational Approach"                    
% Lu Sang, Bjoern Haefner, Daniel Cremers        
%
%
% INPUT:
% I - RGB image sequence 
% s - lighting vector
% z0 - input low-resolution depth sequence 
% mask - binary mask for the object
% K - intrinsic matrix 
% parameters - an object include
%       % tau -tuning parameter for the depth regularizer term
%       % gamma - tuning parameter for the photometric stereo term
%       % fill_missing_z - flag for inpainting the input depth (1 or 0)
%       % apply_smooth_filter - flag for smoothing the input depth (1 or 0)
%       % tol - relative energy stoping criterion 
%       % max_iter - maximum number of iteration
%       % flag_cmg - flag for CMG precondition (1 or 0) CMG should be downloaded from http://www.cs.cmu.edu/~jkoutis/cmg.html
%       % method - energy function evaluation
%       
% rho - initialization of rho_out
% 
% OUTPUT:
% z_out - super-resolution depth map
% rho_out - super-resolution albedo image
% N_out - surface normal
%
% Author: Lu Sang (2019) based on Songyou Peng's version (August 2017)

% input set up
close all

I = inputs.I;
z0 = inputs.z_ls;
s = inputs.s;
mask = inputs.mask;
K = inputs.K;
zs = inputs.z_hs;
rho = inputs.rho;

scaling_factor = size(I,1) / size(z0,1); % Scaling factor
% load the downsampling matrix (K in the paper)
if scaling_factor == 1
    npix = size(z0,1)*size(z0,2);
    D = spdiags(ones(npix,1),0,npix,npix);
elseif (size(I,1)== 480 || size(I,1)== 960 || size(I,1)== 720 ||  size(I,1)== 360)
    load(sprintf('D_%d_%d_%d.mat',size(I,2), size(I,1), scaling_factor));
else
    [ D, ~, ~ ] = getDownsampleMat( scaling_factor, size(I,1), size(I,2));
end

%% parameter setting
gamma = 1e5;
nb_harmo = 4; % Number of Spherical Harmonics parameters
parameters.ambient_light = 0;

if(~isfield(parameters,'do_display'))
    fprintf('WARNING: if display every iteration is not provided, use a default value.')
    parameters.do_display = 1;
end
do_display = parameters.do_display;
if(~isfield(parameters,'tau'))
    fprintf('WARNING: tau not provided, use a default value.')
    parameters.tau = 1;
end
tau = parameters.tau*1e7;
if(~isfield(parameters,'apply_smooth_filter'))
    fprintf('WARNING: if use smooth filter is not specified, use a default value.\n')
    parameters.apply_smooth_filter = 1;
end
apply_smooth_filter = parameters.apply_smooth_filter;
if(~isfield(parameters,'fill_missing_z'))
    fprintf('WARNING: if fill_missing_z is not specified, use a default value.\n')
    parameters.fill_missing_z = 1;
end
fill_missing_z = parameters.fill_missing_z;
if(~isfield(parameters,'tol'))
    fprintf('WARNING: tol is not provided, use a default value.\n')
    parameters.tol = 5e-3;
end
tol = parameters.tol;
if(~isfield(parameters,'max_iter'))
    fprintf('WARNING: max_iter is not provided, use a default value.\n')
    parameters.max_iter = 30;
end
max_iter = parameters.max_iter;
if(~isfield(parameters,'flag_cmg'))
    fprintf('WARNING: if use cmg is not provided, use a default value.\n')
    parameters.flag_cmg = 0;
end
flag_cmg = parameters.flag_cmg;
if(~isfield(parameters,'method'))
    fprintf('WARNING: method is not provided, use a default value.\n')
    parameters.method = 'Cauchy';
end
method = parameters.method;
if(~isfield(parameters,'method_delta'))
    fprintf('WARNING: method_delta is not provided, use a default value.\n')
    if strcmp('Cauchy', method)
        delta = 0.04; 
    elseif strcmp('GM', method)
        delta = 0.1;
    elseif strcmp('Tukey', method)
       delta = 0.15;
    elseif strcmp('Welsh', method)
        delta = 0.15;
    elseif strcmp('Huber', method)
        delta = 0.04;
    else
        delta = 1;
    end
else
    delta = parameters.method_delta;
end


% CG Parameters
flag_pcg = 1; % 1 for PCG, 0 for Least Square
maxit_cg = 100; % Max number of iterations for the inner CG iterations
tol_cg = 1e-9; % Relative energy stopping criterion for the inner CG iterations

%some constant for harmonic lighting
y00 = 0.282095;
y10 = 0.488603;


a0 = pi;
a1 = 2*pi/3;
a2 = pi/4;

if do_display == 1
    figure(1);
    montage(I);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Pre-processing

% define the mask for the low-resolution depth
% masks = mask(1:scaling_factor:end,1:scaling_factor:end);
masks = D*mask(:);
masks = reshape(masks,size(z0,1),size(z0,2));
masks(masks<1) = 0;

% Pre-processing the input depth(s)
zs(zs==0) = NaN;
z0(z0==0) = NaN;

if(fill_missing_z)
    disp('Inpainting');
    tic
    zs = inpaint_nans(zs);
    z0 = inpaint_nans(z0); 
    disp('Done');
    toc
end 

% Smoothing
if(apply_smooth_filter)
%     disp('Filtering...');
%     max_zs = max(max(zs.*mask));
%     tic
%     zs= imguidedfilter(zs/ max_zs);
%     toc
%     zs = zs.*max_zs;
%     disp('Done')
    disp('Filtering...');
    max_zs = max(max(z0.*masks));
    tic
    z0 = imguidedfilter(z0/ max_zs);
    toc
    z0 = z0.*max_zs;
    disp('Done')
end

% initial big depth map
%z = imresize(mean(zs,3), scaling_factor,'bicubic');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Some useful stuff

[nrows,ncols,nchannels, nImg] = size(I);
% pixel index inside the big and small mask
imask = find(mask>0);
imasks = find(masks>0);
% number of pixel
npix = length(imask);
npixs = length(imasks);
% new downsampling matrix, npixs * npix
KT= D(imasks, imask);
imask = find(mask>0);
npix = length(imask);
KT= D(imasks, imask);


% For building surface normals
[xx,yy] = meshgrid(1:ncols,1:nrows);
xx = xx(imask);
xx = xx-K(1,3);
yy = yy(imask);
yy = yy-K(2,3);

G = make_gradient(mask); % Finite differences stencils
Dx = G(1:2:end-1,:);
Dy = G(2:2:end,:);
clear G


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization

% Initial guess - lighting
%s = zeros(size(I,3),9, size(I,4));s(:,3, :) = -1; % frontal directional lighting 
if parameters.harmonic_lighting
    s = light_harmonic(s,1);
end
% Initial guess - albedo
% if (~exist('rho','var'))
%     rho = 0.5.*ones(size(I(:,:,:,1)));
% end

% Vectorization
I = reshape(I,[nrows * ncols,nchannels, nImg]);
I = I(imask,:, :); % Vectorized intensities
I(isnan(I))=0;

rho = reshape(rho,[nrows * ncols,nchannels]);
rho = rho(imask,:); % Vectorized albedo

z = zs(imask);
z0s = z0(imasks);

% parameter normalization
lambda = median(abs(I(:)-median(I(:))));
lambda = delta*(max(lambda,0.005));
%lambda = delta;
tau = tau./(length(z0s)*mean(z0s,'all').^2);
gamma = gamma./(sum(mask,'all')*nchannels*mean(I,'all')^2*nImg);

fprintf('tau after normalization is %.2f, current lambda for method is %.2f\n', tau./gamma, lambda);
% Initial gradient
zx = Dx*z;
zy = Dy*z;

% Initial augmented normals
N = zeros(npix,nb_harmo); 
dz = max(eps,sqrt((K(1,1)*zx).^2+(K(2,2)*zy).^2+(-z-xx.*zx-yy.*zy).^2));
N(:,1) = K(1,1)*zx./dz;
N(:,2) = K(2,2)*zy./dz;
N(:,3) = (-z-xx.*zx-yy.*zy)./dz;
N(:,4) = 1;

E = [NaN];
iter = 1;

T_total = 0;
if do_display
    fprintf('Starting algorithm...\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while 1
	%% ambient lighting estimation
        t_start = tic;
        for i = 1 : nImg 
            A = 0;
            b = 0;
            for ch = 1:nchannels
                r = bsxfun(@times,rho(:,ch),N(:,1:nb_harmo)*s(ch,1:nb_harmo,i)')-I(:,ch,i);
                A_ch = bsxfun(@times,rho(:,ch),N(:,1:nb_harmo));
                b_ch = I(:,ch, i);
                % weight calculation
                if strcmp('Cauchy', method)
                    w_l = lambda^2./(lambda^2+r.^2);
                elseif strcmp('GM', method)
                    w_l = lambda^2./((lambda^2+r.^2).^2);
                elseif strcmp('Tukey', method)
                    w_l = (1-(r.^2./lambda^2)).^2;
                    w_l(abs(r)>lambda)=0;
                elseif strcmp('L1', method)
                    w_l = 1./(max(abs(r),tol));
                elseif strcmp('L2', method)
                    w_l = ones(npix,1);
                elseif strcmp('Welsh', method)
                    w_l = exp(-r.^2./(lambda^2));
                elseif strcmp('Huber', method)
                    w_l = lambda./ abs(r);
                    w_l(abs(r)<lambda)=1;
               else
               end
               w_l = sparse(1:npix,1:npix,w_l(:));
                A = A + A_ch'*w_l*A_ch;
                b = b + A_ch'*w_l*b_ch;
            end

                s_now = solve_framework(s(ch,1:nb_harmo, i)', sparse(A), b,...
                                        flag_pcg, tol_cg, maxit_cg, flag_cmg);

                s(:,1:nb_harmo, i) = repmat(transpose(s_now),[3,1]); 
        end
        s(:,nb_harmo+1:end, :) = 0;
        t_light_cur = toc(t_start);   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Estimate rho
	t_start = tic;	
	for ch = 1 : nchannels
		r = bsxfun(@times,rho(:,ch),N(:,1:nb_harmo)*squeeze(s(ch,1:nb_harmo,:)))-squeeze(I(:,ch,:)); % NPIX x NIMGS
        b = I(:,ch,:);
		b = b(:);
        A_ = N(:,1:nb_harmo)*squeeze(s(ch,1:nb_harmo,:));
        A_ = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_(:));  
        
       if strcmp('Cauchy', method)
            w_rho = lambda^2./(lambda^2+r.^2);         
        elseif strcmp('GM', method)
            w_rho = lambda^2./((lambda^2+r.^2).^2);
        elseif strcmp('Tukey', method)
            w_rho = (1-(r.^2./lambda^2)).^2;
            w_rho(abs(r)>lambda)=0;
        elseif strcmp('L1', method)
            w_rho = 1./(max(abs(r),tol));
        elseif strcmp('L2', method)
            w_rho = ones(nImg*npix,1);
        elseif strcmp('Welsh', method)
            w_rho = exp(-r.^2./(lambda^2));
        elseif strcmp('Huber', method)
            w_rho = lambda./ abs(r);
            w_rho(abs(r)<lambda)=1;
       else
        end
        w_rho = sparse(1:nImg*npix,1:nImg*npix,w_rho(:));
        
        A = A_'*w_rho*A_;
        b = A_'*w_rho*b;
     
		rho(:, ch) = solve_framework(rho(:,ch), A, b,...
							         flag_pcg, tol_cg, maxit_cg, flag_cmg);
	end
	t_albedo_cur = toc(t_start);

	% Plot estimated albedo
    rho_plot = reproduceImage(rho, imask, nrows, ncols, nchannels);
   
    if do_display
        figure(2);imagesc(max(0,min(1,rho_plot)));title('estimated albedo');
        axis off;
        drawnow
    end
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


	%% Depth refinement
	A = [];
	B = [];
    w_z = [];
	t_start = tic;
    for ch = 1:nchannels
        r = bsxfun(@times,rho(:,ch),N(:,1:nb_harmo)*squeeze(s(ch,1:nb_harmo,:)))-squeeze(I(:,ch,:)); % NPIX x NIMGS
        
        if strcmp('Cauchy', method)
            w_ = lambda^2./(lambda^2+r.^2);           
        elseif strcmp('GM', method)
            w_ = lambda^2./((lambda^2+r.^2).^2);
        elseif strcmp('Tukey', method)
            w_ = (1-(r.^2./lambda^2)).^2;
            w_(abs(r)>lambda)=0;
        elseif strcmp('L1', method)
            w_ = 1./(max(abs(r),tol));
        elseif strcmp('L2', method)
            w_ = ones(nImg*npix,1);
        elseif strcmp('Welsh', method)
            w_ = exp(-r.^2./(lambda^2));
        elseif strcmp('Huber', method)
            w_ = lambda./ abs(r);
            w_(abs(r)<lambda)=1;
        end
         

		B_ch = squeeze(I(:,ch,:)) ...
               - bsxfun(@times,rho(:,ch),N(:,4)*squeeze(s(ch,4,:))');
        
		A_ch_1 = bsxfun(@times,rho(:,ch)./dz,bsxfun(@minus,transpose(K(1,1)*squeeze(s(ch,1,:))),bsxfun(@times,xx,transpose(squeeze(s(ch,3,:))))));
    	A_ch_2 = bsxfun(@times,rho(:,ch)./dz,bsxfun(@minus,transpose(K(2,2)*squeeze(s(ch,2,:))),bsxfun(@times,yy,transpose(squeeze(s(ch,3,:))))));
        A_ch_3 = bsxfun(@times,rho(:,ch)./dz,transpose(squeeze(s(ch,3,:))));
           
        A_ch_1 = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_ch_1(:));
        A_ch_2 = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_ch_2(:));
        A_ch_3 = sparse(1:nImg*npix,repmat(transpose(1:npix),[nImg 1]),A_ch_3(:));
        
		A_ch = A_ch_1*Dx+A_ch_2*Dy - A_ch_3;
 
        B = [B; B_ch(:)];
 		A = [A; A_ch];       
        w_z = [w_z;w_(:)];

    end
	
    w_z = sparse(1:nImg*npix*nchannels,1:nImg*npix*nchannels,w_z(:));
    A_ = tau .* (KT' * KT) + gamma .* (A'*w_z* A);
    B_ = tau .* (KT' * z0s) + gamma .* (A'*w_z* B);

	z = solve_framework(z, A_, B_,...
                        flag_pcg, tol_cg, maxit_cg, flag_cmg);
    
    z_new = z;
	t_depth_cur = toc(t_start);                    
	
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	% Energy
    residual = A*z-B;
    if strcmp('Cauchy', method)
        E_cur = lambda^2*log((residual./lambda).^2+1)/2;
    elseif strcmp('GM', method)
        E_cur = residual.^2./(2*(1+residual.^2));
    elseif strcmp('Tukey', method)
        mask_E = (abs(residual)> lambda);
        E_cur = lambda^2/6*(1-(1-(residual/lambda).^2).^3);
        E_cur(mask_E) = lambda^2/6;
    elseif strcmp('L1', method)
        E_cur = abs(residual);
    elseif strcmp('L2', method)
        E_cur = residual.^2/2;
    elseif strcmp('Welsh', method)
        E_cur = lambda.^2/2*(1-exp(-(residual./lambda).^2));
    elseif strcmp('Huber', method)
        mask_E = (abs(residual)>lambda);
        E_cur = residual.^2/2;     
        E_cur(mask_E) = lambda*(abs(residual(mask_E))-lambda/2);
    else
    end
    E = [E, gamma*sum(E_cur)+tau*sum((KT*z - z0s).^2)];
    E_depth = sum((KT*z - z0s).^2);
    E_PS = sum(E_cur);
	% Relative residual
	rel_res = abs(E(end)-E(end-1))./abs(E(end));
	 
	% Update Normal map
	zx = Dx * z;
	zy = Dy * z;
    dz = max(eps,sqrt((K(1,1)*zx).^2+(K(2,2)*zy).^2+(-z-xx.*zx-yy.*zy).^2));
    N(:,1) = K(1,1)*zx./dz;
    N(:,2) = K(2,2)*zy./dz;
    N(:,3) = (-z-xx.*zx-yy.*zy)./dz;
    N(:,4) = 1;

    % Runtime 
    T_total = T_total + t_light_cur+t_albedo_cur+t_depth_cur;

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% Plot and Prints

    Ndx = zeros(size(mask));Ndx(imask) = N(:,1);
    Ndy = zeros(size(mask));Ndy(imask) = N(:,2);
    Ndz = zeros(size(mask));Ndz(imask) = -N(:,3);
    N_out = cat(3,Ndx,Ndy,-Ndz);
    outs = z0;
    out = reproduceImage(z_new, imask, nrows, ncols, 1);
   
    fprintf('[%d] light : %.2f, albedo : %.2f, depth: %.2f, total time: %.2f\n',...
 			iter, t_light_cur, t_albedo_cur, t_depth_cur, t_light_cur+t_albedo_cur+t_depth_cur);
	fprintf('[%d] E : %.2f,E_Ps: %.2f, E_depth : %.2f, R : %f\n\n',...
			iter, E(end), E_PS, E_depth, rel_res);		
    if do_display 
        
        % visualize depth
    	figure(3);
        subplot('Position', [0.05, 0.02, 0.6/2, 0.8/2]);
        imShow('depth3d',outs,masks,diag([1./scaling_factor, 1./scaling_factor,1])*K);title(sprintf('input depth (%d * %d)',nrows/scaling_factor,ncols/scaling_factor));
        
        subplot('Position', [0.4, 0.02, 0.6, 0.8])
        imShow('depth3d',out, mask, K); title(sprintf('refined SR depth (%d * %d)',nrows,ncols));
    	drawnow;
        
    	% Plot Normal map
        figure(4);
        imShow('normals', N_out);
        title('Refined Normal Map')
        axis off
        drawnow

        % Plot the energy
        figure(5); plot(E);title('Energy');
        drawnow
    end 

	iter = iter + 1;

	% Test CV
	if(rel_res<tol || iter>max_iter || E(end) > E(end-1))
        z_out = out;
        z_out(z_out==0) = NaN;    
        rho_out = rho_plot;             
        disp('Done! Enjoy the result.');
		break;
    end
end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [L] = light_harmonic(s,harmo_order)
    y00 = 0.282095;
    y10 = 0.488603;
    y21 = 1.092548;
    y20 = 0.315392;
    y22 = 0.546774;

    a0 = pi;
    a1 = 2*pi/3;
    a2 = pi/4;
    
    nImg = size(s,3);
    
    % normalize   
    L = zeros(3,9,nImg);
    for i = 1 : nImg
        normal_factor = norm(s(1,:,i));
        x = s(1,1,i)./normal_factor;
        y = s(1,2,i)./normal_factor;
        z = s(1,3,i)./normal_factor;

        L(:,4,i) = a0*y00;
        L(:,1,i) = a1*y10*x;
        L(:,2,i) = a1*y10*y;
        L(:,3,i) = a1*y10*z;
        L(:,5,i) = a2*y21*x*y;
        L(:,6,i) = a2*y21*x*z;      
        L(:,7,i) = a2*y21*y*z;
        L(:,8,i) = a2*y22*(x^2-y^2);
        L(:,9,i) = a2*y20*(3*z^2-1);
    end
    
    if harmo_order == 1
        L(:,5:end,:) = 0;
    end
end

function [spherical_harmonics, nb_harmo] = normals2SphericalHarmonics(normals, harmo_order)
%normals2SphericalHarmonics is a function which calculates the spherical
%harmonics based on the normals and the corresponding spherical harmonics
%order.
%INPUT:
%       normals is a nx3 matrix, each column represents [nx,ny,nz]
%       harmo_order = {0, 1, 2, ...} and describes the spherical harmonics
%       order
%OUTPUT:
%       spherical_harmonics is of size nxnb_harmo
%       nb_harmo = {1, 4, 9, ...} and describes the dimension of
%       approximation of the spherical harmonics
%
%OPTIONAL OUTPUT:
%       J       is the Jacobian matrix J(zx,zy,z)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
    y00 = 0.282095;
    y10 = 0.488603;
    y21 = 1.092548;
    y20 = 0.315392;
    y22 = 0.546774;

    nb_harmo = harmo_order^2 + 2*harmo_order + 1;

    spherical_harmonics = zeros(size(normals,1), nb_harmo);

    normalize_factor = vecnorm(normals');
    normals = normals./normalize_factor';

    if ( harmo_order == 0)
      spherical_harmonics(:) = 1;
      return;
    end

    if ( harmo_order == 1 || harmo_order == 2)
      spherical_harmonics(:,1:3) = y10*normals;
      spherical_harmonics(:,4) = y00;
    end

    if (harmo_order == 2)
      spherical_harmonics(:,5) = y21*normals(:,1)      .* normals(:,2);
      spherical_harmonics(:,6) = y21*normals(:,1)      .* normals(:,3);
      spherical_harmonics(:,7) = y21*normals(:,2)      .* normals(:,3);
      spherical_harmonics(:,8) = y22*normals(:,1) .^ 2 -  normals(:,2) .^ 2;
      spherical_harmonics(:,9) = y20*( 3*normals(:,3) .^ 2 - 1);
    end

    if ( harmo_order > 2)
      error('Error in normals2SphericalHarmonics(): Unknown order of spherical harmonics %d; Not yet implemented', nb_harmo);
    end

end


