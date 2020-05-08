function [img] = reproduceImage(vec_in, imask, nRows, nCols, nChannels)

img = zeros(nRows, nCols, nChannels);
for ci = 1:nChannels
  img_ch = zeros(nRows, nCols, 1);
  img_ch(imask) = vec_in(:,ci);
  img(:,:,ci) = img_ch;
end

end
