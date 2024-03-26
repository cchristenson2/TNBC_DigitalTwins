%{ 
Build breast boundary condition map in 3D

Inputs:
    - mask; 3D Breast mask

Outputs:
    - bcs
        - 4D boundary map for all (y,x,z) pairs
        - Boundary vector bcs(y,x,z,:) = [y boundary type, x boundary type, z boundary type]
            Value = -1 if boundary is behind, left, or below
            Value = 0 if no boundary is present
            Value = 1 if boundary is in front, right or above
            Value = 2 if outside of mask

Last updated: 9/14/2023
%}

function [bcs] = buildBoundaries_3D(mask)
    [sy,sx,sz] = size(mask);
    bcs = zeros(sy,sx,sz,3); %[y,x,z,[y type, x type, z type]]
    for z = 1:sz
        for y = 1:sy
            for x = 1:sx
                boundary = [0,0,0]; %[y x z], exists on -1 to 2

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
                        test = mask(y-1,x,z);
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
                        test = mask(y+1,x,z);
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
                        test = mask(y,x-1,z);
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
                        test = mask(y,x+1,z);
                        if(test==0)
                           boundary(2) = 1; 
                        end
                    catch
                        boundary(2) = 1; 
                    end
                end
                
                %% Z boundary conditions
                %Edge of grid
                if(z == 1)
                    boundary(3)=-1;
                elseif(z==sz)
                    boundary(3)=1;
                end
                %Under boundary check
                if(boundary(3)==0)
                    try
                        test = mask(y,x,z-1);
                        if(test==0)
                           boundary(3) = -1; 
                        end
                    catch
                        boundary(3) = -1; 
                    end
                end
                %Above boundary check
                if(boundary(3)==0)
                    try %positive y-check
                        test = mask(y,x,z+1);
                        if(test==0)
                           boundary(3) = 1; 
                        end
                    catch
                        boundary(3) = 1; 
                    end
                end
                
                %Remove everything outside of mask
                if(mask(y,x,z)==0)
                    boundary(1) = 2;
                    boundary(2) = 2;
                    boundary(3) = 2;
                end
            
                bcs(y,x,z,:) = boundary;

            end
        end
    end

end