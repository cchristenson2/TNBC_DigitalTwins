%{
Description: 

Input:
    - Problem structure

Named inputs:
    - 'LibSize'; 2 (default), or integer for number of library components for each parameter
    - 'samples'; [] (default) or array of parameter samples to build snapshots

Output:
    - ROM structure
        - V: projection matrix for the snapshots
        - Library: operator library for the given model and fmt for naming

Last updated:

%}

function ROM = constructROM(Problem, varargin)

    %Name-value pair set up
    defaultLibSize = 2;
    defaultAugmentation = 'averaged';
    validAugmentation = {'averaged','sampled'};

    %Parse inputs
    p = inputParser;
    p.addRequired('Problem',@isstruct);
    p.addOptional('Samples',[],@(x) isa(x,'double'));
    p.addParameter('LibSize', defaultLibSize, @(x) isa(x,'double'));
    p.addParameter('Augmentation',defaultAugmentation, @(x) any(strcmp(validAugmentation,x)));

    parse(p, Problem, varargin{:});
    
    Problem = p.Results.Problem;
    LibSize = p.Results.LibSize;
    Samples = p.Results.Samples;
    Augmentation = p.Results.Augmentation;

    %Get snapshots for basis construction
    temp = size(Problem.tumor.N,4);
    if(temp == 1), dimension = 2; else, dimension = 3; end
    if(strcmp(Augmentation,'averaged'))
        if(dimension == 2) %2D augmentation
            Snapshots = augmentAveraging_2D(Problem.tumor);
        else %3D augmentation
            Snapshots = augmentAveraging_3D(Problem.tumor);
        end
    elseif(strcmp(Augmentation,'sampled'))
        %Use parameter samples for snapshot generation !!!Not written!!!
        if(dimension == 2) %2D generation
            Snapshots = sampleData_2D(Problem, Samples);
        else %3D generation
            Snapshots = sampleData_3D(Problem, Samples);
        end
    end

    %Construct the basis
    [V,r] = getProjectionMatrix(Snapshots,0);

    %Depending on the model, construct the library for the relevant bounds
    %Currently only set up for RXDIF with combined alpha for A/C therapy
    fmt = '%.6f';
    if(strcmp(func2str(Problem.model),'RXDIF_3D_wAC_comb') || strcmp(func2str(Problem.model),'RXDIF_2D_wAC_comb'))
        Library = struct;

        Library.D = buildReduced_D_lib( Problem.param_bounds(1,:) , V, fmt, LibSize, Problem.tumor);
        [P1,P2] = buildReduced_LocalKp_lib( Problem.param_bounds(2,:) , V, fmt, LibSize);
        Library.P1 = P1; Library.P2 = P2;
        Library.T = buildReduced_T_lib( Problem.param_bounds(3,:) , V, fmt, LibSize, Problem.tumor.AUC);

        reducedModel = @ OperatorRXDIF_wAC_comb;

    else
        error('Model not currently set up for building ROM');
    end

    ROM.V = V;
    ROM.r = r;
    ROM.Library = Library;
    ROM.fmt = fmt;
    ROM.reducedModel = reducedModel;

end