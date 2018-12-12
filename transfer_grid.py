#Marielle B Fournier, June 2018
#Subroutine for transferring the grid to have the cold pool in the center of the domain

import numpy as np
import copy as cp

def transfer_grid(grid,delx,dely,dimx,dimy):

   grid_out = np.zeros((2,dimy+1,dimx+1))
   grid_in = grid

   if delx > 0:
       grid_out[1, :, delx:] = grid[1, :, 0:dimx - delx + 1]
       grid_out[1, :, 0:delx] = grid[1, :, dimx - delx + 1:]
   elif delx < 0:
       grid_out[1, :, 0:delx] = grid[1, :, (-1) * delx:]
       grid_out[1, :, delx:] = grid[1, :, 0:(-1) * delx]
   else:
       grid_out = grid_in

   grid_new = cp.deepcopy(grid_out)

   if dely > 0:
       grid_out[0, dely:, :] = grid_new[0, 0:dimy - dely + 1, :]
       grid_out[0, 0:dely, :] = grid_new[0, dimy - dely + 1:, :]
   elif dely < 0:
       grid_out[0, 0:dely, :] = grid_new[0, (-1) * dely:, :]
       grid_out[0, dely:, :] = grid_new[0, 0:(-1) * dely, :]
   else:
       grid_out = grid_new

   return grid_out

def value_trans(nmb,value,delx,dely,dimx,dimy):

   value_out = np.zeros((dimy+1,dimx+1))
   value_in = value[nmb,:,:]

   if delx > 0:
       value_out[:, delx:] = value_in[:, 0:dimx - delx + 1]
       value_out[:, 0:delx] = value_in[:, dimx - delx + 1:]
   elif delx < 0:
       value_out[:, 0:delx] = value_in[:, (-1) * delx:]
       value_out[:, delx:] = value_in[:, 0:(-1) * delx]
   else:
       value_out = value_in

   value_new = cp.deepcopy(value_out)

   if dely > 0:
       value_out[dely:, :] = value_new[0:dimy - dely + 1, :]
       value_out[0:dely, :] = value_new[dimy - dely + 1:, :]
   elif dely < 0:
       value_out[0:dely, :] = value_new[(-1) * dely:, :]
       value_out[dely:, :] = value_new[0:(-1) * dely, :]
   else:
       value_out = value_new

   #if delx > 0:
   #    value_out[:,delx+1:] = value[nmb,:,0:dimx-delx]
   #    value_out[:,0:delx+1] = value[nmb,:,dimx-delx:]
   #elif delx < 0:
   #    value_out[:,0:dimx+delx+2] = value[nmb,:,-delx-1:]
   #    value_out[:,dimx+delx+2:] = value[nmb,:,0:(-1)*delx-1]
   #else:
   #    value_out = value[nmb,:,:]

   #if dely > 0:
   #    value_out[dely+1:,:] = value[nmb,0:dimy-dely,:]
   #    value_out[0:dely+1,:] = value[nmb,dimy-dely:,:]
   #elif dely < 0:
   #    value_out[0:dimy+dely+2,:] = value[nmb,-dely-1:,:]
   #    value_out[dimy+dely+2:,:] = value[nmb,0:(-1)*dely-1,:]
   #else:
   #    value_out = value[nmb,:,:]

   return value_out

def wind_trans(wind,delx,dely,dimx,dimy):

   wind_out = np.zeros((dimy+1,dimx+1))
   wind_in = wind

   if delx > 0:
       wind_out[:, delx:] = wind_in[:, 0:dimx - delx + 1]
       wind_out[:, 0:delx] = wind_in[:, dimx - delx + 1:]
   elif delx < 0:
       wind_out[:, 0:delx] = wind_in[:, (-1) * delx:]
       wind_out[:, delx:] = wind_in[:, 0:(-1) * delx]
   else:
       wind_out = wind_in

   wind_new = cp.deepcopy(wind_out)

   if dely > 0:
       wind_out[dely:, :] = wind_new[0:dimy - dely + 1, :]
       wind_out[0:dely, :] = wind_new[dimy - dely + 1:, :]
   elif dely < 0:
       wind_out[0:dely, :] = wind_new[(-1) * dely:, :]
       wind_out[dely:, :] = wind_new[0:(-1) * dely, :]
   else:
       wind_out = wind_new

   return wind_out

