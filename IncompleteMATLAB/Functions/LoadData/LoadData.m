%{
Description: Function for loading in processed MRI
Coarse resolution is reduced by factor of 2 in-plane
2D selection, automatically selects the central axial slice of MRI

Inputs:
    - File location

Named inputs:
    - 'Resolution'; 'coarse' (default) or 'full'
    - 'Dimension'; '3' (default) or '2'

Output:
    - Tumor structure:
        - N: cell data
        - h, dz: scan dimensions (in-plane and slice)
        - t_scan: times of MRI acquisitions (ns x 1 vector)
        - t_trx: times for scheduled treatments (nt x 1 vector)
        - d_trx: doses for scheduled treatments (nt x 1 vector)
        - AUC: concentration map from DCE MRI
        - Mask: breast mask
        - Tissues: tissue segmentation masks (required for mechanical coupling)
        - bcs: boundary condition maps

        Optional params:
        - t_surgery: time of surgery
        - t_trx_pac: time for paclitaxel administrations
        - CF: correction factor for coarse images

Last updated: 3/15/2024

%}

function tumor = LoadData(location, varargin)


    tumor = struct;
    
    %Name-value pair set up
    defaultResolution = 'coarse';
    validResolutions = {'coarse','full'};
    defaultDimension = 2;
    validDimensions = [2,3];

    %Parse inputs
    p = inputParser;
    p.addRequired('location',@ischar);
    p.addParameter('resolution', defaultResolution, @(x) any(strcmp(validResolutions,x)));
    p.addParameter('dimension',defaultDimension,@(x) any(x == validDimensions));

    parse(p, location, varargin{:});
    
    location = p.Results.location;
    resolution = p.Results.resolution;
    dimension = p.Results.dimension;

    %Load in data at location
    data = load(location);

    %Get full cell maps in 3D with requested resolution
    if(strcmp(resolution,'full'))
        names = fields(data.full_res_dat);
        idx = contains(names,'NTC');
        names(~idx) = [];
        names = sort(names);
        N = zeros([size(data.full_res_dat.NTC1), numel(names)]);
        for i = 1:numel(names)
            N(:,:,:,i) = data.full_res_dat.("NTC" + num2str(i));
        end

        AUC = data.full_res_dat.AUC;
        Mask = data.full_res_dat.BreastMask;
        Tissues = data.full_res_dat.Tissues;

        tumor.h = data.schedule_info.imagedims(1);
        tumor.dz = data.schedule_info.imagedims(3);
    else
        names = fields(data.coarse_res_dat);
        idx = contains(names,'NTC');
        names(~idx) = [];
        names = sort(names);
        N = zeros([size(data.coarse_res_dat.NTC1), numel(names)]);
        for i = 1:numel(names)
            N(:,:,:,i) = data.coarse_res_dat.("NTC" + num2str(i));
        end
        tumor.CF = data.coarse_res_dat.CF;

        AUC = data.coarse_res_dat.AUC;
        Mask = data.coarse_res_dat.BreastMask;
        Tissues = data.coarse_res_dat.Tissues;

        tumor.h = data.schedule_info.imagedims(1)*data.coarse_res_dat.CF;
        tumor.dz = data.schedule_info.imagedims(3);
    end
    
    %Normalize voxels to carrying capacity
    theta = 0.7405 * ((data.schedule_info.imagedims(1))^2 * data.schedule_info.imagedims(3)) / 4.189e-6; %This technically should be specific to the resolution but the data I have does not adhere to this
    N = N./theta;
    N(N>1) = 0.99999;

    %Get patient schedule for standard of care regimen
    t_all = cumsum(data.schedule_info.times);
    tumor.t_scan = t_all(data.schedule_info.schedule=='S');
    tumor.t_trx  = t_all(data.schedule_info.schedule=='A');
    tumor.d_trx  = ones(1,numel(tumor.t_trx));

    try
        t_pac_all = t_all(end) + cumsum(data.schedule_info.times_pac);
        tumor.t_surgery = t_pac_all(data.schedule_info.schedule_pac=='X');
        tumor.t_trx_pac = t_pac_all(data.schedule_info.schedule_pac=='P');
        tumor.d_trx_pac = ones(1,numel(tumor.t_trx_pac));
    catch
        disp('No paclitaxel or surgery information in tumor file.');
    end

    %Crop to 2D if requested
    %Save into struct, build boundaries based on mask
    if(dimension == 2)
        sz = size(N,3);
        slice = round(sz/2);

        tumor.N = squeeze(N(:,:,slice,:));
        tumor.AUC = AUC(:,:,slice);
        tumor.Mask = Mask(:,:,slice);
        tumor.Tissues = Tissues(:,:,slice);

        tumor.bcs = buildBoundaries_2D(tumor.Mask);
    else
        tumor.N = N;
        tumor.AUC = AUC;
        tumor.Mask = Mask;
        tumor.Tissues = Tissues;

        tumor.bcs = buildBoundaries_3D(tumor.Mask);
    end

    try
        tumor.pcr_status = data.pcr;
    catch
        disp('No PCR information in tumor file.');
    end

end