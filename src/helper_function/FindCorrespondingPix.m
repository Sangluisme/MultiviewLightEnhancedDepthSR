function [I_new] = FindCorrespondingPix(DRef, I, T, K, mask)
    
    DRef(~mask)=0;
    
    R = T(1:3, 1:3);
    t = T(1:3,4);
    
    RKInv = R * K^-1;
     
    scale_factor = size(I,1)/size(DRef,1);  
    [cols, rows] = size(DRef);
    
    
    [m, n] = meshgrid(1:cols,1:rows);
    D_ = DRef';
    p = [n(:),m(:),ones(length(n(:)),1)].*D_(:);
    
    p_trans = K * (RKInv * p' + t);
    nImg = transpose(p_trans(1,:)./p_trans(3,:));
    mImg = transpose(p_trans(2,:)./p_trans(3,:));
    
%     nImg(imask) = -10;
%     mImg(imask) = -10;
%     
    nImg = reshape(nImg, size(DRef,2),size(DRef,1))';
    mImg = reshape(mImg, size(DRef,2),size(DRef,1))';
    
    nImg(nImg<=0)=-10;
    nImg(nImg>size(DRef,2))=-10;
    
    mImg(mImg<=0)=-10;
    mImg(mImg>size(DRef,1))=-10;
    
    nImg(mImg<=0)=-10;
    nImg(mImg>size(DRef,1))=-10;
    
    mImg(nImg<=0)=-10;
    mImg(nImg>size(DRef,2))=-10;
    
    I_new = zeros(size(I));
   
    
    for i=1:size(I,3)
        I_ = I(:,:,i);        
        I_new(:,:,i) = interp2(I_,nImg,mImg);
        
    end
   I_new(isnan(I_new))=0;
end
    
    
  