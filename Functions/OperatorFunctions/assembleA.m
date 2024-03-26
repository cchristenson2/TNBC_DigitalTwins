%{
Build operator for diffusivity with boundary conditions around the breast

dN/dt = AN = grad(d*grad(N))

Input:
    - N; Cell map for sizing, 2D or 3D cell map 
    - d; diffusivity (spatially varying or scalar value)
    - h, dz; Grid spacing, in-plane (h), slice (dz)
    - bcs; Map containing the breast boundaries

Output:
    - A; Laplacian operator for a given domain, and diffusivity map

Contributors: Chase Christenson, Graham Pash
%}
function [A] = assembleA(N, d, h, dz, bcs)
%     disp(size(d));
    if(size(d,1) == 1)
        d = d.*ones(numel(N),1);
        scalar = 1;
    else
        d = d(:);
        scalar = 0;
    end
    
    L = sparse(assembleL(N, d, h, dz));
    if(scalar == 0)
        div = sparse(assembleDiv(N, d, h, dz, bcs));
    else
        div = 0;
    end
    
    A = L+div;
    
    A = applyBC(A, bcs, h, dz, d);
    
%     A = L+div;
end

%Build laplacian for d*lap(N)
function [L] = assembleL(N, d, h, dz)
    [sy,sx,sz] = size(N);
    
    if(sz==1) %2D build
        n = sy*sx;
        e = d(:);
        Y = spdiags([e -2*e e]./(h^2), [-1 0 1],n,n);
        X = spdiags([e -2*e e]./(h^2), [-sy 0 sy],n,n);
        L = sparse(X+Y);
        
    else %3D build
        n = sy*sx*sz;
        e = d(:);
        Y = spdiags([e -2*e e]./(h^2), [-1 0 1],n,n);
        X = spdiags([e -2*e e]./(h^2), [-sy 0 sy],n,n);
        Z = spdiags([e -2*e e]./(dz^2), [-1*(sy*sx) 0 (sy*sx)],n,n);
        L = sparse(X+Y+Z);
    end
end

%Build divergence for div(d) * div(N) with boundaries included
%Boundaries have to be built now since d is included in this operator
function [div] = assembleDiv(N, d, h, dz, bcs)
    [sy,sx,sz] = size(N);
    
    if(sz==1) %2D build
        n = sy*sx;
        div = zeros(n,n);
        
        for x = 1:sx
            for y = 1:sy
                i = y + (x-1)*sy;
                % Y Direction
                if(bcs(y,x,1)==0) %Only on interior nodes
                    div(i,i+1) = (d(i+1)-d(i-1))/(4*(h^2));
                    div(i,i-1) = (d(i-1)-d(i+1))/(4*(h^2));
                end
                % X Direction
                if(bcs(y,x,2)==0) %Only on interior nodes
                    div(i,i+sy) = (d(i+sy)-d(i-sy))/(4*(h^2));
                    div(i,i-sy) = (d(i-sy)-d(i+sy))/(4*(h^2));
                end
            end
        end
        
        div = sparse(div);
        
        
    else %3D build
        n = sy*sx*sz;
        div = zeros(n,n);
        
        for z = 1:sz
            for x = 1:sx
                for y = 1:sy
                    i = y + (x-1)*sy + (z-1)*sx*sy;
                    
                    % Y Direction
                    if(bcs(y,x,z,1)==0) %Only on interior nodes
                        div(i,i+1) = (d(i+1) - d(i-1))/(4*(h^2));
                        div(i,i-1) = (d(i-1) - d(i+1))/(4*(h^2));
                    end
                    
                    % X Direction
                    if(bcs(y,x,z,2)==0) %Only on interior nodes
                        div(i,i+sy) = (d(i+sy)-d(i-sy))/(4*(h^2));
                        div(i,i-sy) = (d(i-sy)-d(i+sy))/(4*(h^2));
                    end

                    % Z Direction
                    if(bcs(y,x,z,3)==0) %Only on interior nodes
                        div(i,i+(sy*sx)) = (d(i+(sy*sx))-d(i-(sy*sx)))/(4*dz^2);
                        div(i,i-(sy*sx)) = (d(i-(sy*sx))-d(i+(sy*sx)))/(4*dz^2);
                    end
                    
                end
            end
        end
        
        div = sparse(div);
    end
end


function [L] = applyBC(L, bcs, h, dz, d)
% Apply Neumann BC discrete laplacian
    [sy,sx,sz,~] = size(bcs);
    n_3D = sy*sx*sz;
    n_2D = sy*sx;
    
    if(sz<=2) %2D operator
        for x = 1:sx
            for y = 1:sy
%                 disp(['(y,x) = (',num2str(y),',',num2str(x),')']);
                i = y + (x-1)*sy;
                %Remove outside of mask contributions
                if(bcs(y,x,1)==2 || bcs(y,x,2)==2)
                    L(i,:) = 0;
                else

                    if(bcs(y,x,1)==-1) %Check if top wall
                        L(i,i+1) = 2*d(i)/(h^2);
                        if(i-1>=1)
                            L(i,i-1) = 0;
                        end
                    elseif(bcs(y,x,1)==1) %Check bottom wall
                        L(i,i-1) = 2*d(i)/(h^2);
                        if(i+1<=n_2D)
                            L(i,i+1) = 0;
                        end
                    end

                    if(bcs(y,x,2)==-1) %Check if left wall
                        L(i,i+sy) = 2*d(i)/(h^2);
                        if(i-sy>=1)
                            L(i,i-sy) = 0;
                        end
                    elseif(bcs(y,x,2)==1) %Check if right wall
                        L(i,i-sy) = 2*d(i)/(h^2);
                        if(i+sy<=n_2D)
                            L(i,i+sy) = 0;
                        end
                    end
                end
            end
        end
    else %3D operator
        for z = 1:sz
            for x = 1:sx
                for y = 1:sy
        %             disp(['(y,x) = (',num2str(y),',',num2str(x),')']);
                    i = y + (x-1)*sy + (z-1)*sx*sy;
%                     disp([x, y, z]);
                    %Remove outside of mask contributions
                    if(bcs(y,x,z,1)==2 || bcs(y,x,z,2)==2 || bcs(y,x,z,3)==2)
                        L(i,:) = 0;
                    else
                        if(bcs(y,x,z,1)==-1) %Check if top wall
                            L(i,i+1) = 2*d(i)/(h^2);
                            if(i-1>=1)
                                L(i,i-1) = 0;
                            end
                        elseif(bcs(y,x,z,1)==1) %Check bottom wall
                            L(i,i-1) = 2*d(i)/(h^2);
                            if(i+1<=n_3D)
                                L(i,i+1) = 0;
                            end
                        end

                        if(bcs(y,x,z,2)==-1) %Check if left wall
                            L(i,i+sy) = 2*d(i)/(h^2);
                            if(i-sy>=1)
                                L(i,i-sy) = 0;
                            end
                        elseif(bcs(y,x,z,2)==1) %Check if right wall
                            L(i,i-sy) = 2*d(i)/(h^2);
                            if(i+sy<=n_3D)
                                L(i,i+sy) = 0;
                            end
                        end
                        
                        if(bcs(y,x,z,3)==-1) %Check if under wall
                            L(i,i+sy*sx) = 2*d(i)/(dz^2);
%                             disp(size(L));
                            if(i-(sy*sx)>=1)
                                L(i,i-(sy*sx)) = 0;
                            end
                        elseif(bcs(y,x,z,3)==1) %Check if above wall
                            L(i,i-sy*sx) = 2*d(i)/(dz^2);
                            if(i+(sy*sx)<=n_3D)
                                L(i,i+(sy*sx)) = 0;
                            end
                        end
                        
%                         disp(size(L));
                    end
                end
            end
        end
    end
end
