%{ 
Build breast boundary condition map in 2D

Inputs:
    - mask; 2D Breast mask (0s outside of breast and 1s inside)

Outputs:
    - bcs
        - 3D boundary map for all (y,x) pairs
        - Boundary vector bcs(y,x,:) = [y boundary type, x boundary type]
            Value = -1 if boundary is behind, left, or below
            Value = 0 if no boundary is present
            Value = 1 if boundary is in front, right or above
            Value = 2 if outside of mask

Last updated: 9/14/2023
%}
function [bcs] = buildBoundaries_2D(mask)
    [sy,sx] = size(mask);
    bcs = zeros(sy,sx,2); %[y,x,[y type, x type]]
    for y = 1:sy
        for x = 1:sx
            boundary = [0,0]; %[y x], exists on -1 to 1, 2 = outside of mask
            
            %% Y boundary conditions
            %Edge of grid
            if(y == 1)
                boundary(1)=-1;
            elseif(y==sy)
                boundary(1)=1;
            end
            %Upwards boundary check
            if(boundary(1)==0)
                try
                    test = mask(y-1,x);
                    if(test==0)
                       boundary(1) = -1; 
                    end
                catch
                    boundary(1) = -1; 
                end
            end
            %Downwards boundary check
            if(boundary(1)==0)
                try %positive y-check
                    test = mask(y+1,x);
                    if(test==0)
                       boundary(1) = 1; 
                    end
                catch
                    boundary(1) = 1; 
                end
            end
            
            %% X boundary conditions
            %Edge of grid
            if(x == 1)
                boundary(2)=-1;
            elseif(x==sx)
                boundary(2)=1;
            end
            %Left boundary check
            if(boundary(2)==0)
                try
                    test = mask(y,x-1);
                    if(test==0)
                       boundary(2) = -1; 
                    end
                catch
                    boundary(2) = -1; 
                end
            end
            %Right boundary check
            if(boundary(2)==0)
                try %positive y-check
                    test = mask(y,x+1);
                    if(test==0)
                       boundary(2) = 1; 
                    end
                catch
                    boundary(2) = 1; 
                end
            end
            
            %Remove everything outside of mask
            if(mask(y,x)==0)
                boundary(1) = 2;
                boundary(2) = 2;
            end
            
            bcs(y,x,:) = boundary;
            
        end
    end

end