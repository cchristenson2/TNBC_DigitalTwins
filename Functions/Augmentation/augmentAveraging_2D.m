%{
Averages maps in order to fill in gaps between time points, no parameter estimation required

Inputs:
    N - cell maps to fill gaps between
    t - time of cell maps
    depth - number of averages to add

Outputs:
    N_aug - full time course with true maps, and averaged maps inbetween
    t_spacing - time of each map in the augmented cell data

%}


function [N_aug, t_spacing] = augmentAveraging_2D(tumor)

    depth = 8;

    N_aug = tumor.N;
    t_spacing = tumor.t_scan;
    
    mask = sum(tumor.N,3);
    mask(mask>0) = 1;
    mask = logical(mask);

    %% Create new midpoint for every depth, 1 = one average, 2 = two averages ...
    for j = 1:depth
        
        nt = numel(t_spacing); %Number of images in current dataset
        
        curr = 0;
        for i = 1:nt-1
            
            N_mid = (N_aug(:,:,i+curr)+N_aug(:,:,i+1+curr))./2;
            
            if(j==1)
                N_mid = imgaussfilt(N_mid, 0.5);
            end
            
            t_mid = (t_spacing(i+curr)+t_spacing(i+1+curr))/2;
            
            N_aug = cat(3, N_aug(:,:,1:i+curr), N_mid, N_aug(:,:,i+1+curr:end));
            t_spacing = [t_spacing(1:i+curr), t_mid, t_spacing(i+1+curr:end)];
            
            curr = curr+1;
        end
    end

    for i = 1:size(N_aug,3)
        temp = N_aug(:,:,i);
        temp(~mask) = 0;
        N_aug(:,:,i) = temp;
    end
    
end