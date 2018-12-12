#Bilinear interpolation between grid points.
#X = 2 by 2 matrix including the 4 values around the point that needs to be interpolated
#ix,iy = dx,dy respectively to the point
#dX = the interpolated value that is inbetween all the points in the matrix X
#OBS! The grid is (y,x) and not (x,y) why the formula is slightly different from the formula on https://en.wikipedia.org/wiki/Bilinear_interpolation

#def interpolate(X,ix,iy):
#    dX = X[0,0]*(1.-ix)*(1.-iy) + X[1,0]*ix*(1.-iy) + X[0,1]*(1.-ix)*iy + X[1,1]*ix*iy

#    return dX

#Modification May 2018 - the interpolation was wrong (indices)
#http://multivis.net/lecture/bilinear.html

def interpolate(X,ix,iy):
   A = X[0,0] - X[1,0]
   B = X[1,1] - X[1,0]
   C = X[1,0] - X[0,0] - X[1,1] + X[0,1]
   D = X[1,0]
   dX = A*iy + B*ix + C*ix*iy + D

   return dX

