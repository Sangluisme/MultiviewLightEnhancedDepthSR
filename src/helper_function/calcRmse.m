%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% calcRmse%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function rmse = calcRmse(I, In, mask)
% by haefner
% I : original signal
% In: noisy signal(ie. original signal + noise signal)
% mask which values to take into account
% rmse = sqrt(sum((I(mask) - In(mask)).^2)/numel(mask))


if (~exist('mask','var'))
  mask3d = true(size(I));
else
%     for i=1:4
%         mask = imerode(mask, [0,1,0;1,1,1;0,1,0]);
%     end
    mask3d = repmat(mask,1,1,size(I,3));
end

% calculate rmse
rmse = sqrt(sum((I(mask3d) - In(mask3d)).^2)/sum(mask3d(:)));

end
