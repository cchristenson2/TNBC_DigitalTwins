
clear all; clc; close all;

path = pwd();
addpath(genpath(pwd));

%% Load data

    data_loc = [path, '\Data\PatientData_ungrouped\'];
    files = dir(data_loc); files([1,2]) = [];
    tumor = LoadData([data_loc,files(1).name],'resolution','coarse','dimension',3); % Do not try with full 3D yet, operators cannot be built

%% Set up problem

    Problem = struct;
    Problem.param_bounds = [1e-6, 1e-3; 1e-3, 0.1; 1e-3, 1.0; 0.35, 0.85; 1.0, 5.5];
    %Diffusivity, Proliferation, Alpha, Beta-A, Beta-C
    
    %Make distribution objects for global parameters
    priors = struct;
    priors.d_dist = makedist('Normal','mu',5e-4,'sigma',2.5e-4);
    priors.d_dist = truncate(priors.d_dist,Problem.param_bounds(1,1),Problem.param_bounds(1,2));
    
    priors.alpha_dist = makedist('Uniform','lower',Problem.param_bounds(3,1),'upper',Problem.param_bounds(3,2));
    
    priors.betaA_dist = makedist('Normal','mu',0.60,'sigma',0.0625);
    priors.betaA_dist = truncate(priors.betaA_dist,Problem.param_bounds(4,1),Problem.param_bounds(4,2));
    
    priors.betaC_dist = makedist('Normal','mu',3.25,'sigma',0.5625);
    priors.betaC_dist = truncate(priors.betaC_dist,Problem.param_bounds(5,1),Problem.param_bounds(5,2));
    
    Problem.priors = priors;
    clear priors;

    %Measured data from V1 and V2 represents starting point for calibration.
    %Treatment schedule must be saved
    Problem.tumor = tumor;
    if(size(tumor.N,4)==1)
        Problem.tumor.N(:,:,end) = []; %Remove v3 from measured data
    else
        Problem.tumor.N(:,:,:,end) = []; %Remove v3 from measured data
    end
    Problem.tumor.t_scan(end) = []; %Remove timing for v3

    %Save future data for V3 separately
    %Can be added onto tumor for assimilation or used to compare predictions
    Problem.futureTumor = tumor;
    if(size(tumor.N,4)==1)
        Problem.futureTumor.N(:,:,1:end-1) = [];
    else
        Problem.futureTumor.N(:,:,:,1:end-1) = [];
    end
    Problem.futureTumor.t_scan(1:end-1) = [];

    clear tumor;

    %Set model to be used by full problem
    Problem.model = @ RXDIF_3D_wAC_comb;

%% Construct ROM
    ROM = constructROM(Problem);


%% Calibrate with MCMC ROM






%% Optimize therapy




