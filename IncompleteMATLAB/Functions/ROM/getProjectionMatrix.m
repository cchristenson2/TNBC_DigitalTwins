 %{ 
Get projection matrix for a given set of snapshots

Inputs:
    - N; Snapshot matrix, (Sy X Sx X nt, or Sy X Sx X Sz X nt)
    - k; desired rank

Outputs:
    - V; projection matrix 
    - k; rank of selected projection basis
        - returns input k unless input is 0

Contributors: Chase Christenson
%}

function [V, k] = getProjectionMatrix(N,k)
    [~,~,temp1,temp2] = size(N);
    %Reshape 2D or 3D time course into vectors over time
    if(temp2==1) %2D
        N_vert = zeros(numel(N(:,:,1)), temp1);
        for i = 1:temp1
            N_vert(:,i) = reshape(N(:,:,i),[],1);
        end
    else %3D
        N_vert = zeros(numel(N(:,:,:,1)), temp2);
        for i = 1:temp2
            N_vert(:,i) = reshape(N(:,:,:,i),[],1);
        end
    end

    %Compute SVD of snapshot matrix
    if(k==0)
        k_mid = 20; %Base target rank if not specified
    else
        k_mid = k;
    end
    [U,S,~] = svds(N_vert, k_mid);
    
    %Compute cummulative energy and select first k columns based off decay
    sings = S(S>0);
    cummulative_e = cumsum(sings);
    max_e = max(cummulative_e);
    if(k == 0)
        k = numel(cummulative_e(cummulative_e < (max_e*0.995))) + 1;
    end

    %Set projection basis
    V = U(:,1:k);
end