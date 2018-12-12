# Marielle B Fournier, July 2018
# Subroutine for calculating the radial velocity for each cold pool at multiple time step.
# Each cold pool is identified from the identified rain events using Haerter, Moseleys
# Iterative rain cell tracking software. The radial velocity is the component of the
# total horizontal velocity pointing in the direction of the center of mass (COM) = grid
# point of maximum divergence, of the cold pool. Positive is defined to be away from the cold pool.
# The first and second derivative is also calculated using bilinear interpolation.
# For every grid point, the points 1 ps closer and away from the center along the radial
# axis are found and their values are interpolate by the four surrounding grid points
# Finite differences are calculated from the two interpolated points.
# Including the calculation for finding cold pool boundaries


def calc_vr(in_path, out_file):

   from netCDF4 import Dataset
   import numpy as np
   import copy as cp
   import time as tid
   from find_tracks import trackID
   from find_tracks import track_time
   from find_tracks import COM_x
   from find_tracks import COM_y
   from find_tracks import tracktime_arr
   from find_tracks import IDmerge
   from transfer_grid import transfer_grid
   from transfer_grid import value_trans
   from transfer_grid import wind_trans
   from calc_maxdvr_first import maxdvr_first
   from calc_maxdvr_more import maxdvr_more
   import vr
   import dvr


   # find number of time steps and domain size
   data_in_wind2D = Dataset(in_path + in_file + 'wind2D.nc', 'r', format='NETCDF4')  # + '3D/'

   time_steps = data_in_wind2D.variables['time'].size
   dim_x = 1024  # 320
   dim_y = 1024  # 320
   dim_z = data_in_wind2D.variables['zt'].size

   # first and last time step
   t0 = 0
   t1 = time_steps
   dt = 300  # temporal resolution is 300 seconds = 5 minutes
   dxdy = 200  # horizontal resolution is 200 meters

   # generate output file
   data_out = Dataset(out_file, 'w', format='NETCDF4')
   xt = data_out.createDimension('xt', dim_x)
   yt = data_out.createDimension('yt', dim_y)
   zt = data_out.createDimension('zt', dim_z)
   time = data_out.createDimension('time', time_steps)
   cp_nmb = data_out.createDimension('cp_nmb', None)
   onex = data_out.createDimension('onex', 1)
   oney = data_out.createDimension('oney', 1)

   # createVariable makes a variable of name 'time', format 'f8' and dimension
   # of the variable 'time'
   time_out = data_out.createVariable('time', 'f8', ('time',))
   xt_out = data_out.createVariable('xt', 'f8', ('xt',))
   yt_out = data_out.createVariable('yt', 'f8', ('yt',))
   cp_nmb_out = data_out.createVariable('cp_nmb', 'f8', ('cp_nmb',))
   cp_ID_out = data_out.createVariable('cp_ID', 'f4', ('cp_nmb', 'oney', 'onex', 'time'))
   vrad_out = data_out.createVariable('vr', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
   #dvrdr_out = data_out.createVariable('dvrdr', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
   cp_edge_out = data_out.createVariable('cp_edge', 'f4', ('yt', 'xt', 'time'))

   # Create variables to use for the identification of the same cold pools
   maxcp = 400
   info_out = np.zeros(shape=(time_steps, maxcp, 3, 2))
   r_save = np.zeros(shape=(time_steps, maxcp))

   time_in = np.array(data_in_wind2D.variables['time'])
   # just because the variables xt and yt as missing!!
   xt_in = np.linspace(1, dim_x, dim_x)
   yt_in = np.linspace(1, dim_y, dim_y)
   zt_in = np.array(data_in_wind2D.variables['zt'])

   # attributes long_name and unit
   time_out.long_name = 'Time'
   time_out.units = 's'
   time_out[:] = time_in[t0:t1]

   xt_out.long_name = 'Longitudinal position of cell centers'
   xt_out.units = 'm'
   xt_out[:] = xt_in

   yt_out.long_name = 'Lateral position of cell centers'
   yt_out.units = 'm'
   yt_out[:] = yt_in

   cp_ID_out.long_name = 'ID of the individual cold pool for the whole time series'
   cp_ID_out.units = 'Number'

   vrad_out.long_name = 'Radial velocity'
   vrad_out.units = 'm s-1'

   #dvrdr_out.long_name = 'Radial derivative of the radial velocity (second order accuracy)'
   #dvrdr_out.units = 's-1'

   start_time = tid.time()
   print ('starting time is', start_time)

   # Creating a grid that resembles the grid of the model
   x = np.linspace(1, dim_x, dim_x).astype(int)
   y = np.linspace(1, dim_y, dim_y).astype(int)
   grid = np.array(np.meshgrid(y, x, indexing='ij'))
   startup = 10
   initial_grid = grid[:,dim_y/2-startup:dim_y/2+startup,dim_x/2-startup:dim_x/2+startup]
   ingr_xy = np.size(initial_grid,axis=1)
   ingr_x = np.linspace(dim_x/2-startup+1, dim_x/2+startup, ingr_xy).astype(int)
   ingr_y = np.linspace(dim_y/2-startup+1, dim_y/2+startup, ingr_xy).astype(int)

   cp_ID_list = np.zeros(shape=(time_steps, maxcp))
   xylist = np.zeros(shape=(1, 6))

   # Looping over time
   for ts in range(t0, t1):

       print ('t=', ts, '/', time_steps)

       trackID_in = list()
       tracktime_in = list()

       for tt in range(0, len(trackID)):
           if track_time[tt] - 1 == ts:
               trackID_in.append(trackID[tt])
               tracktime_in.append(track_time[tt])

       if not trackID_in:  # If no cold pools are found, continue to next time step
           cp_first = True
           continue

       if cp_first:
           ts_first = ts
           cp_first = False

       trackIDs = np.array(trackID_in)
       trackIDs_initial = cp.deepcopy(trackIDs)

       if ts == ts_first:
           pass
       else:
           for n in range(0, len(trackIDs_last)):
               if trackIDs_last[n] not in trackIDs:
                   # If the track is terminated by a merging event, it should be terminated as a cold pool as well
                   if np.isin(trackIDs_last[n], IDmerge):
                       pass
                   else:
                       vrad_one = cp.deepcopy(vrad_last[n,:,:])
                       vrad_one_mean = np.mean(vrad_one[cpedge_last == trackIDs_last[n]])
                       if vrad_one_mean > 1.:
                           trackIDs = np.append(trackIDs,trackIDs_last[n])

       # The amount of cold pools identified in the time step determines the size of the variables
       cp_ID = np.zeros(shape=(len(trackIDs), 1, 1))
       vrad = np.zeros(shape=(len(trackIDs), dim_y, dim_x))
       dvrdr = np.zeros(shape=(len(trackIDs), dim_y, dim_x))
       cp_edge = np.zeros(shape=(dim_y, dim_x))
       info = np.zeros(shape=(len(trackIDs), 3, 2))
       cp_nmb_in = np.linspace(1, len(trackIDs), len(trackIDs))
       cp_nmb_out[:] = cp_nmb_in
       COMx_save = np.zeros(shape=(len(trackIDs)))
       COMy_save = np.zeros(shape=(len(trackIDs)))

       COMx_list = []
       COMy_list = []
       for k in range(0, len(trackID)):
           if (track_time[k] - 1 == ts) and ~np.isin(tracktime_arr[k], 0): #Don't exclude the COM-point if it is the first for that cold pool
               COMx_list.append(COM_x[k])
               COMy_list.append(COM_y[k])
       COMx_list = np.array(COMx_list)
       COMy_list = np.array(COMy_list)
       for nmb in range(0, len(trackIDs)):
           if trackIDs[nmb] not in trackIDs_initial and (ts != ts_first):
               for n in range(0, len(trackIDs_last)):
                   if trackIDs_last[n] == trackIDs[nmb]:
                       COMx_list = np.append(COMx_list, COMx_last[n])
                       COMy_list = np.append(COMy_list, COMy_last[n])

       print ('Amount of cold pools:', len(trackIDs))
       print ('Calculating vr and dvrdr')

       for nmb in range(0, len(trackIDs)):

           wind2D_in = np.array(data_in_wind2D.variables['wind2D'][ts, :, :, 0])
           wind2D_dir_in = np.array(data_in_wind2D.variables['wind2D_dir'][ts, :, :, 0])

           cp_ID[nmb, :, :] = trackIDs[nmb]
           cp_ID_list[ts, nmb] = trackIDs[nmb]

           COMx_in = list()
           COMy_in = list()

           if trackIDs[nmb] not in trackIDs_initial and (ts != ts_first):
               for n in range(0, len(trackIDs_last)):
                   if trackIDs_last[n] == trackIDs[nmb]:
                       COMx_in.append(COMx_last[n])
                       COMy_in.append(COMy_last[n])
           else:
               for k in range(0, len(trackID)):
                   if (trackID[k] == trackIDs[nmb]) and (track_time[k] - 1 == ts):
                       COMx_in.append(COM_x[k])
                       COMy_in.append(COM_y[k])

           COMx = np.array(COMx_in)
           COMy = np.array(COMy_in)
           COM = np.array([COMy[0], COMx[0]])  # The COM of the time step, COM is updated with the rain COM
           COMx_save[nmb] = COMx
           COMy_save[nmb] = COMy
           # The COM is defined with a grid starting with 1, ending at 320 (domainsize)

           # Transfer grid such that the cold pool is in the middle of the domain
           delx = np.int(np.round(dim_x/2. - COM[1]))
           dely = np.int(np.round(dim_y/2. - COM[0]))
           # Set COM in the middle
           COM_trans = np.array([dim_y / 2. + (COMy[0] - np.round(COMy[0])), dim_x / 2. + (COMx[0] - np.round(COMx[0]))])

           #grid_trans = transfer_grid(grid, delx, dely)
           dimx = dim_x - 1  # Indexwise
           dimy = dim_y - 1  # Indexwise
           wind2D_trans = wind_trans(wind2D_in, delx, dely, dimx, dimy)
           winddir_trans = wind_trans(wind2D_dir_in, delx, dely, dimx, dimy)

           # Remember that the grid in the model is (y,x), why the first value of the position vector for the wind
           # must be the sine value and the second must be the cosine. Think triangle.
           wind2D_pos = np.array(
               [np.multiply(wind2D_trans, np.sin(winddir_trans)), np.multiply(wind2D_trans, np.cos(winddir_trans))])

           # CALCULATE VR AND DVR
           if (ts == ts_first) or (trackIDs[nmb] not in trackIDs_last):
               # Calculate vr using cython script
               vrad[nmb, dim_y/2-startup:dim_y/2+startup, dim_x/2-startup:dim_x/2+startup] = vr.vr_calc(ingr_xy, ingr_xy, initial_grid, COM_trans, wind2D_pos[:, dim_y/2-startup:dim_y/2+startup, dim_x/2-startup:dim_x/2+startup])
               #print ('vr calculated for cp', str(trackIDs[nmb]))

               # Calculate dvr using cython script
               dvrdr[nmb, dim_y/2-startup:dim_y/2+startup, dim_x/2-startup:dim_x/2+startup] = dvr.dvr_calc(ingr_x, ingr_y, initial_grid, ingr_xy, ingr_xy, COM_trans, nmb, vrad)
               #print ('dvr calculated for cp ', str(trackIDs[nmb]))
           else:
               for i in range(0, len(trackIDs_last)):
                   if trackIDs_last[i] == trackIDs[nmb]:
                       search_up_old = info_out[i, 1, :]
                       search_down_old = info_out[i, 2, :]
                       new_length = len(np.linspace(search_down_old[0], search_up_old[0], (search_up_old[0] - search_down_old[0] + 1)))
                       if new_length % 2 == 0:
                           new_length = new_length + 1
                       new_length = new_length / 2 - 1 / 2
                       vrad_edge = np.max(vrad_last[i, :, :][cpedge_last == trackIDs[nmb]])
                       distance = np.abs(vrad_edge) * dt
                       travel = distance / dxdy
                       grid_travel = np.round(travel) + 12
                       new_up = dimy/2 + new_length + grid_travel
                       new_down = dimy/2 - new_length - grid_travel
                       smallgrid_yx = np.linspace(new_down, new_up, (new_up - new_down + 1)).astype(int)
                       smallgrid = np.array(np.meshgrid(smallgrid_yx, smallgrid_yx, indexing='ij'))
                       smallgridlen = len(smallgrid_yx)
                       # Calculate vr using cython script
                       vrad[nmb, np.min(smallgrid_yx)-1:np.max(smallgrid_yx), np.min(smallgrid_yx)-1:np.max(smallgrid_yx)] = vr.vr_calc(smallgridlen, smallgridlen, smallgrid, COM_trans, wind2D_pos[:, np.min(smallgrid_yx)-1:np.max(smallgrid_yx), np.min(smallgrid_yx)-1:np.max(smallgrid_yx)])
                       #print ('vr calculated for cp', str(trackIDs[nmb]))
                       # Calculate dvr using cython script
                       dvrdr[nmb, np.min(smallgrid_yx)-1:np.max(smallgrid_yx), np.min(smallgrid_yx)-1:np.max(smallgrid_yx)] = dvr.dvr_calc(smallgrid_yx, smallgrid_yx, smallgrid, smallgridlen, smallgridlen, COM_trans, nmb, vrad)
                       #print ('dvr calculated for cp ', str(trackIDs[nmb]))

           # Transforming back to the original grid
           delx = -delx
           dely = -dely
           vrad[nmb, :, :] = value_trans(nmb, vrad, delx, dely, dimx, dimy)
           dvrdr[nmb, :, :] = value_trans(nmb, dvrdr, delx, dely, dimx, dimy)

           #print('Calculating cp edge for cp', str(trackIDs[nmb]))
           # Finding cold pool boundaries
           travelmore = 2

           if ts == ts_first:

                cp_edge, info, xylist, r_save = maxdvr_first(vrad, nmb, COM, dim_x, dim_y, dt, dxdy, travelmore, info, dvrdr, r_save, ts, cp_ID_list, cp_edge, xylist, startup)
                print('dvr_max calculated for cp ', str(trackIDs[nmb]))
                if ts == ts_first:
                    xylist = xylist[1:, :]
           else:
               cpID_last = cp_ID_list[ts - 1, :]

               if cp_ID_list[ts, nmb] not in cpID_last:
                   cp_edge, info, xylist, r_save = maxdvr_first(vrad, nmb, COM, dim_x, dim_y, dt, dxdy, travelmore,
                                                                info, dvrdr, r_save, ts, cp_ID_list, cp_edge, xylist,
                                                                startup)
                   print('dvr_max calculated for cp ', str(trackIDs[nmb]))
               else:
                   cp_edge, info, xylist, r_save = maxdvr_more(trackID, track_time, cpID_last, info_out, vrad, nmb, COM,
                                                               dt, dxdy, travelmore, dvrdr, r_save, ts, cp_ID_list,
                                                               cp_edge, xylist, vrad_last, info, COMy_list, COMx_list,
                                                               dimx, dimy, travel)
                   print('dvr_max calculated for cp ', str(trackIDs[nmb]))
           #info_out[ts, nmb, :, :] = info[nmb, :, :]

       info_out = cp.deepcopy(info)
       vrad_last = cp.deepcopy(vrad)
       COMx_last = cp.deepcopy(COMx_save)
       COMy_last = cp.deepcopy(COMy_save)
       cpedge_last = cp.deepcopy(cp_edge)
       trackIDs_last = cp.deepcopy(trackIDs)
       vrad_out[:, :, :, ts] = vrad[:, :, :]
       #dvrdr_out[:, :, :, ts] = dvrdr[:, :, :]
       cp_edge_out[:, :, ts] = cp_edge[:, :]
       cp_ID_out[:,:,:,ts] = cp_ID[:,:,:]

       del vrad,dvrdr,cp_edge,cp_ID

   data_in_wind2D.close()

   data_out.close()

   elapsed_time = (tid.time() - start_time)/(60*60)
   print ('elapsed time is', elapsed_time)

   return xylist;



# Main program

from netCDF4 import Dataset
import numpy as np
import os
import sys

in_path = '/conv1/fournier/UCLA_lind/'  # '/conv1/fournier/test1plus4K/rain_begin/'
out_path = in_path + 'nc_output/largerslices/'  # 'minsizelarger/nc_output/'
in_file = 'lind_p2K.out.vol.'  # 'test1plus4K.out.vol.'
out_file = out_path + in_file + 'edge.nc'
print(in_path)
print(out_file)

program = calc_vr(in_path, out_file)
np.savetxt('xylist.txt',program,delimiter=',')

