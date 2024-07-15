img = double(mat2gray(imread('071318.jpg')));

mask_size=[5 5];
sigma=3;
%zuo paper claims no noticeable accuracy difference within sigma range 1-4

Gaussian_filter = fspecial('gaussian',mask_size,sigma);
%https://www.mathworks.com/help/images/ref/fspecial.html

A=conv2(img, Gaussian_filter, 'same');

Gx = [-1, 0, 1; -2, 0, 2; -1, 0, 1];
Gy = [1, 2, 1; 0, 0, 0; -1, -2, -1];
%Sobel operators/filters in horizontal and vertical direction commonly used in
%image filtering; while Zuo recommended against these filters they seemed
%to work well in this project
%https://en.wikipedia.org/wiki/Sobel_operator

SobelX = conv2(A, Gx, 'same');
SobelY = conv2(A, Gy, 'same');

gradient_direction = (atan2(SobelY, SobelX))*180/pi;
gradient_magnitude = sqrt((SobelX.^2) + (SobelY.^2));

Threshold_High = min(min(gradient_magnitude))+max(max(gradient_magnitude))*.09;
Threshold_Low = Threshold_High/2;
%Thresholding values suggested by Zuo; high=90% of max intensity,
%low=high/2

rows=size(A,1);
columns=size(A,2);

for i=1:rows
    for j=1:columns
        if (gradient_direction(i,j)<0) 
            gradient_direction(i,j)=360+gradient_direction(i,j);
        end
    end
end
edge_direction=zeros(rows,columns);
%adjust to make gradient directions all positive
%https://www.mathworks.com/matlabcentral/fileexchange/46859-canny-edge-detection


for i = 1  : rows
    for j = 1 : columns
        if ((gradient_direction(i,j)>=0)&&(gradient_direction(i,j)<22.5)||(gradient_direction(i,j)>=157.5)&&(gradient_direction(i,j)<202.5)||(gradient_direction(i,j)>=337.5)&&(gradient_direction(i,j)<=360))
            edge_direction(i,j)=0;
        elseif((gradient_direction(i,j)>=22.5)&&(gradient_direction(i,j)<67.5)||(gradient_direction(i,j)>=202.5)&&(gradient_direction(i,j)< 247.5))
            edge_direction(i,j)=45;
        elseif((gradient_direction(i,j)>=67.5&&gradient_direction(i,j)<112.5)||(gradient_direction(i,j)>=247.5&&gradient_direction(i,j)<292.5))
            edge_direction(i,j)=90;
        elseif((gradient_direction(i,j)>=112.5&&gradient_direction(i,j)<157.5)||(gradient_direction(i,j)>=292.5&&gradient_direction(i,j)<337.5))
            edge_direction(i,j)=135;
        end
    end
end
%adjust directions to nearest 0,45,90,135 degrees
%https://www.mathworks.com/matlabcentral/fileexchange/46859-canny-edge-detection

edge1 = zeros (rows, columns);
for i=2:rows-1
    for j=2:columns-1

        if (edge_direction(i,j)==0)
            edge1(i,j)=(gradient_magnitude(i,j)==max([gradient_magnitude(i,j),gradient_magnitude(i,j+1),gradient_magnitude(i,j-1)]));

        elseif (edge_direction(i,j)==45)
            edge1(i,j)=(gradient_magnitude(i,j)==max([gradient_magnitude(i,j),gradient_magnitude(i+1,j-1),gradient_magnitude(i-1,j+1)]));
        elseif (edge_direction(i,j)==90)
            edge1(i,j)=(gradient_magnitude(i,j)==max([gradient_magnitude(i,j),gradient_magnitude(i+1,j),gradient_magnitude(i-1,j)]));
        elseif (edge_direction(i,j)==135)
            edge1(i,j)=(gradient_magnitude(i,j)==max([gradient_magnitude(i,j),gradient_magnitude(i+1,j+1),gradient_magnitude(i-1,j-1)]));
        end
    end
end
edge2 = edge1.*gradient_magnitude;
%non-maximum supression
%https://www.mathworks.com/matlabcentral/fileexchange/46859-canny-edge-detection

Threshold_Low = Threshold_Low * max(max(edge2));
Threshold_High = Threshold_High * max(max(edge2));
%recalibrate threshold to further supress non-maximum values

edge_binary = zeros (rows, columns);

for i = 1  : rows
    for j = 1 : columns
        if (edge2(i, j) < Threshold_Low)
            edge_binary(i, j) = 0;
        elseif (edge2(i, j) > Threshold_High)
            edge_binary(i, j) = 1;
        elseif(edge2(i+1,j)>Threshold_High||edge2(i-1,j)>Threshold_High||edge2(i,j+1)>Threshold_High||edge2(i,j-1)>Threshold_High||edge2(i-1,j-1)>Threshold_High||edge2(i-1,j+1)>Threshold_High||edge2(i+1,j+1)>Threshold_High||edge2(i+1,j-1)>Threshold_High)
            edge_binary(i,j)=1;
        end
    end
end
%binarization of final edge
%https://www.mathworks.com/matlabcentral/fileexchange/46859-canny-edge-detection

edge_binary(1600:2054,:)=[];
edge_binary(:,1:300)=[];

edge_binary(602,526)=1;
%cutting off additional noise from needle edge/outside noise; can be
%adjusted to specific image needs, add edge points est. where there is a
%hole

ind = find(edge_binary',1,'last');
[x0,y0] = ind2sub(size(edge_binary'),ind);

ind = find(edge_binary',1,'first');
[xmax,ymax] = ind2sub(size(edge_binary'),ind);

imshow(imcomplement(edge_binary));

hold on
axis equal
title('Captive Bubble Air-Liquid Interface')
xlabel('x')
ylabel('y')
plot([x0 x0],[y0 ymax],'k')

hline_step=50;
n=round((y0-ymax)/hline_step,-1);

x=zeros(n,1);
y=x;



for i = 1:n-1
	y(i) = y0-i*hline_step;
    x(i) = find(edge_binary(y(i),:),1,'first');
    plot([x(i) x0], [y(i) y(i)],'b');
end

for i = 1:n-2
    xg = (x(i+1)+x(i))/2;           % middle point
    yg = (y(i+1)+y(i))/2;           
    yg_p = (x(i+1)-x(i))/(y(i+1)-y(i))*(xg-x0) + yg;% perpendicular
    plot([xg x0],[yg yg_p],'g');
end

  for i=1:n-3
     
  plot(x(i+1),y(i+1),'.r');
  end
   
x=nonzeros(x');
x=reshape(x,size(x,1),1);
y=nonzeros(y');
y=reshape(y,size(y,1),1);

plot(x,y,'r')

hold off

%syms x %creates variable x without a value
%Cx=solve(slope_2 * (x - midX_2) + midY_2==slope * (x - midX) + midY,x);
%Cy=slope_2 * (Cx - midX_2) + midY_2;
%find intersection of two perpendiculars by setting them equal and solving
%for x
%https://www.mathworks.com/help/symbolic/solve.html

%RP12=eval(sqrt((Cx-midX)^2+(Cy-midY)^2)); %length of RP12
%RP23=eval(sqrt((Cx-midX_2)^2+(Cy-midY_2)^2)); %length of RP23
%RP=double((RP12+RP23)/2); %length of RP

%ratio1=(Cy-my1)/(Cx-mx1); 
%alpha1=90-(atand(ratio1)); %find leftmost angle in triangle formed by RP, R2;
%subtract angle from 90 to get theta/remaining angle in triangle

%ratio2=(Cy-my2)/(Cx-mx2);
%alpha2=90-(atand(ratio2));

%theta=eval((alpha1+alpha2)/2);

%RPx=x2+(RP*cosd(90-theta));
%RPy=(m+n)+(RP*sind(90-theta));
%plot([x2 RPx],[(m+n) RPy],'r'); %RP
%plot([RPx RPx],[RPy m+n],'r'); %vertical line from RP to m+n

%syms RT phi beta
%V=solve(-R3*cosd(beta)==-R2+RT*(1-cosd(phi))*sind(theta), RT*sind(phi)==R3*sind(beta), (m+n)==m+RT*(1-cosd(phi))*cosd(theta));
%V.RT=eval(V.RT);
%V.phi=eval(V.phi);
%V.beta=eval(V.beta);

%rho=1000;
%g=9.8;
%z=(m+n)-x1;
%rho*g*z=-delta_P+ST(1/RP(z)+1/RT(z));
