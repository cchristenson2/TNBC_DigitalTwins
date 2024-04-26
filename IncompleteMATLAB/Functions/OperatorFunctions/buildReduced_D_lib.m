%{ 
Builds ROM library for a set of parameter bounds, and grid sizes
Operator type determined by input

Inputs:
    - bounds; bounds for the parameters of interest
        n x 2 vector at a minimum; giving upper and lower bounds for n parameters
    - N; Cell map for sizing, 2D or 3D cell map 
    - fmt; name formating
    - number of operators to build total, including the bounds (2 minimum)
    - Concentration map, 2D or 3D

Outputs:
    - lib; library for all operators in the bounds vector, and each parameter

Contributors: Chase Christenson
%}

function [lib] = buildReduced_D_lib(bounds,V,fmt,num,tumor)

    lib = struct;
    
    if(num>=2)
        curr_vec = linspace(bounds(1),bounds(2),num);
    else
        disp('Number of operators to build must be 2 or higher');
        lib = NaN;
        return;
    end
    
    for i = 1:numel(curr_vec)
        
        curr = curr_vec(i);
        
        name = ['val',replace(num2str(curr,fmt),'.','_')];

        lib.(name) = V' * assembleA(tumor.Mask, curr, tumor.h, tumor.dz, tumor.bcs) * V;

    end
    
end