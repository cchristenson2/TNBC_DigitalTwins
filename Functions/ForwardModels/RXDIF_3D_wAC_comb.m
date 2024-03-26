%{ 
Forward model for the reaction diffusion equation with AC treatment + Paclitaxel in 3D
dN(t)/dt = D*d^2N(t)/h^2 + kp*N(t)(1-N(t)) - dose(t)*alpha*(exp(-beta1*t) - exp(-beta2*t))*C(t)*N(t) - dose_pac(t)*alpha_pac*exp(-beta3*t)*C(t)*N(t)
N is a volume fraction

Input:
    - tumor: Struct needed for cell data, timing and dosing of drugs
    - params: Struct containing all current parameter values and associated operators (if valid)
    - dt: time step
    
Named inputs:
    - 'initial': index of initial condition; '1' (default) to number of visits in tumor
    - 'tspan': times to output solution; V2 scan time by default, can be any array of doubles

Output:
    - N_sim; Cell map at desired times
    - drug_TC; Drug time course from 0 to t, for drugs A/C and potentially Paclitaxel

Last Updated: 3/15/2024
%}

function [N_sim, drug_TC] = RXDIF_3D_wAC_comb(tumor, params, dt, varargin)
    %Name-value pair set up
    defaultStart = 1; %Starts at V1 cell count as default
    validStarts = 1:size(tumor.N,3);
    defaultEndTime = numel(tumor.t_scan); %Simulates to last scan as default

    %Parse inputs
    p = inputParser;
    p.addRequired('tumor',@isstruct);
    p.addRequired('params',@isstruct);
    p.addRequired('dt',@(x) isa(x,'double'));
    p.addParameter('initial', defaultStart, @(x) any(x == validStarts));
    p.addParameter('tspan',tumor.t_scan(defaultEndTime), @(x) isa(x,'double'));

    p.parse(tumor, ROM, params, dt, varargin{:});

    tumor  = p.Results.tumor;
    params = p.Results.dt;
    dt     = p.Results.dt;
    N0     = tumor.N(:,:,:,p.Results.initial);
    t      = p.Results.tspan;
    clear p;

    %Set up simulation
    nt = t(end)/dt + 1;
    t_ind = t./dt + 1;

    N = zeros(size(N0), nt);
    N(:,:,:,1) = N0;

    %Remove treatments where the dose is 0
    idx = find(tumor.d_trx == 0);
    tumor.t_trx(idx) = [];
    tumor.d_trx(idx) = [];
    
    %Combination AC therapy variables
    nt_trx = tumor.t_trx./dt; %Indices of treatment times
    trx_on = 0; %Starts off, turns on at first treatment delivery
    trx_cnt = 1;
    delivs = [];

    %Paclitaxel variables, if exists
    names = fieldnames(tumor);
    if(~isempty(contains(names,'t_trx_pac')))
        idx = find(tumor.d_trx_pac == 0);
        tumor.t_trx_pac(idx) = [];
        tumor.d_trx_pac(idx) = [];

        nt_trx_pac = tumor.t_trx_pac./dt;
        trx_on_pac = 0;
        trx_cnt_pac = 1;
        delivs_pac = [];

        b3 = params.beta3;

        pac_exists = 1;
        n_drugs = 3;
        if(t(end) < tumor.t_trx_pac(1))
            pac_exists = 0;
        end
    else
        pac_exists = 0;
        n_drugs = 2;
    end

    b1 = params.beta1;
    b2 = params.beta2;
    
    drug_TC = zeros(n_drugs,nt);
    
    %Time stepping
    for k = 2:nt
        %Get time since last A/C treatment
        try            
            if(k - 1/dt >= nt_trx(trx_cnt)) %If current time is at or above the next delivery
                if(trx_on == 0)
                    delivs = 0; %add first treatment
                    trx_on = 1;
                else
                    delivs = delivs + dt; %Add dt to old treatment
                    delivs = [delivs, 0]; %Add new treatment to list
                end
                trx_cnt = trx_cnt + 1; %Move index to look for next treatment next time
            else %No new delivery, so add dt to the current ones
                if(trx_on == 1)
                    delivs = delivs + dt;
                end
            end
        catch %No more deliveries, keep adding dt to all the old ones
            delivs = delivs + dt;
        end

        %Get time since last paclitaxel if it exists
        if(pac_exists==1)
            try            
                if(k - 1/dt >= nt_trx_pac(trx_cnt_pac)) %If current time is at or above the next delivery
                    if(trx_on_pac == 0)
                        delivs_pac = 0; %add first treatment
                        trx_on_pac = 1;
                    else
                        delivs_pac = delivs_pac + dt; %Add dt to old treatment
                        delivs_pac = [delivs_pac, 0]; %Add new treatment to list
                    end
                    trx_cnt_pac = trx_cnt_pac + 1; %Move index to look for next treatment next time
                else %No new delivery, so add dt to the current ones
                    if(trx_on_pac == 1)
                        delivs_pac = delivs_pac + dt;
                    end
                end
            catch %No more deliveries, keep adding dt to all the old ones
                delivs_pac = delivs_pac + dt;
            end
        end
        
        %Space stepping
        for z = 1:sz
            for y = 1:sy
                for x = 1:sx
                    boundary = tumor.bcs(y,x,z,:);
                    
                    %FDM in Y direction
                    if(boundary(1)==0)
                        inv_y = params.d*(N(y+1,x,z,k-1)-2*N(y,x,z,k-1)+N(y-1,x,z,k-1))/(tumor.h^2);
    
                    elseif(boundary(1)==1)
                        inv_y = params.d*(-2*N(y,x,z,k-1)+2*N(y-1,x,z,k-1))/(tumor.h^2);
    
                    elseif(boundary(1)==-1)
                        inv_y = params.d*(-2*N(y,x,z,k-1)+2*N(y+1,x,z,k-1))/(tumor.h^2);
                    else
                        inv_y = 0;
                    end
    
                    %FDM in X direction
                    if(boundary(2)==0)
                        inv_x = params.d*(N(y,x+1,z,k-1)-2*N(y,x,z,k-1)+N(y,x-1,z,k-1))/(tumor.h^2);
    
                    elseif(boundary(2)==1)
                        inv_x = params.d*(-2*N(y,x,z,k-1)+2*N(y,x-1,z,k-1))/(tumor.h^2);
    
                    elseif(boundary(2)==-1)
                        inv_x = params.d*(-2*N(y,x,z,k-1)+2*N(y,x+1,z,k-1))/(tumor.h^2);
                    else
                        inv_x = 0;
                    end

                    %FDM in Z direction
                    if(boundary(3)==0)
                        inv_z = params.d*(N(y,x,z+1,k-1)-2*N(y,x,z,k-1)+N(y,x,z-1,k-1))/(tumor.dz^2);
    
                    elseif(boundary(3)==1)
                        inv_z = params.d*(-2*N(y,x,z,k-1)+2*N(y,x,z-1,k-1))/(tumor.dz^2);
    
                    elseif(boundary(3)==-1)
                        inv_z = params.d*(-2*N(y,x,z,k-1)+2*N(y,x,z+1,k-1))/(tumor.dz^2);
                    else
                        inv_z = 0;
                    end
    
                    invasion = inv_y + inv_x + inv_z;
                    prolif   = N(y,x,z,k-1)*params.k(y,x,z,k-1)*(1-N(y,x,z,k-1));
    
                    %Treatment calculation                
                    %Solve treatment effects A/C chemo
                    drug1 = 0;
                    drug2 = 0;
                    if(~isempty(delivs))
                        for n = 1:numel(delivs)
                            drug1 = drug1 + tumor.d_trx(n)*exp(-b1*delivs(n));
                            drug2 = drug2 + tumor.d_trx(n)*exp(-b2*delivs(n));
                        end
                        treat = params.alpha*(drug1+drug2)*tumor.AUC(y,x,z)*N(y,x,z,k-1);
                    end
                    drug_TC(1,k) = drug1;
                    drug_TC(2,k) = drug2;
                    %Solve treatment effects paclitaxel
                    if(pac_exists==1)
                        drug3 = 0;
                        if(~isempty(delivs_pac))
                            for n = 1:numel(delivs_pac)
                                drug3 = drug3 + tumor.d_trx_pac(n)*exp(-b3*delivs_pac(n));
                            end
                            treat = treat + params.alpha_pac*drug3*tumor.AUC(y,x,z)*N(y,x,z,k-1);
                        end
                        drug_TC(3,k) = drug3;
                    end

                    N(y,x,z,k) = N(y,x,z,k-1) + dt*(invasion + prolif - treat);
                end
            end
        end
    end

    N_sim = N(:,:,:,t_ind);
end