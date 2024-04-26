%{ 
Builds ROM library for a set of parameter bounds, and grid sizes
Operator type determined by input

Library split first into modes contained in V, then repeated num times

Inputs:
    - bounds; bounds for the parameters of interest
        1 x 2 vector; giving upper and lower bounds for kp
    - N; Cell map for sizing, 2D or 3D cell map 
    - fmt; name formating
    - number of operators to build total, including the bounds (2 minimum)
    - param; which operator to build for, either B or H

Outputs:
    - lib; library for all operators in the bounds vector, and each parameter

Contributors: Chase Christenson
%}

function [P1, P2] = buildReduced_LocalKp_lib(bounds,V,fmt,num)

    P1 = struct;
    
    [n,k] = size(V);
    
    %Solve for coefficient bounds given modes, and kp bounds
    coeff_bounds = zeros(k,2);
    for z = 1:k
        test = zeros(n,1);
        test(V(:,z)>=0) = bounds(2);
        test(V(:,z)<0) = bounds(1);
        
        temp(1) = (V(:,z)' * test);

        test = zeros(n,1);
        test(V(:,z)>=0) = bounds(1);
        test(V(:,z)<0) = bounds(2);
        
        temp(2) = (V(:,z)' * test);
        
        coeff_bounds(z,:) = sort(temp,'ascend');
    end
    
    %Cycle through each mode for linear operator
    for z = 1:k
        if(num>=2)
            curr_vec = linspace(coeff_bounds(z,1), coeff_bounds(z,2), num);
        else
            error('Number of operators to build must be 2 or higher');
        end
        %Cycle through each value between the bounds
        for i = 1:numel(curr_vec)
            
            curr = curr_vec(i);
            curr_map = V(:,z) * curr;
            
            temp_op = zeros(k,k);
            for j = 1:n
                temp_op = temp_op + V(j,:)' * curr_map(j) * V(j,:);
            end

            name = ['val',replace(replace(num2str(curr,fmt),'.','_'),'-','n')];

            P1.("Mode" + num2str(z)).(name) = temp_op;
        end
    end
    P1.coeff_bounds = coeff_bounds;

    %Cycle through each mode for quadratic operator
    for z = 1:k
        if(num>=2)
            curr_vec = linspace(coeff_bounds(z,1), coeff_bounds(z,2), num);
        else
            error('Number of operators to build must be 2 or higher');
        end
        %Cycle through each value between the bounds
        for i = 1:numel(curr_vec)
            
            curr = curr_vec(i);
            curr_map = V(:,z) * curr;
            
            temp_op = zeros(k,k^2);
            for j = 1:n
                temp_op = temp_op + V(j,:)' * curr_map(j) * kron(V(j,:),V(j,:));
            end

            name = ['val',replace(replace(num2str(curr,fmt),'.','_'),'-','n')];

            P2.("Mode" + num2str(z)).(name) = temp_op;
        end
    end
    P2.coeff_bounds = coeff_bounds;
    
end