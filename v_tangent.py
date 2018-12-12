#Marielle B Fournier, October 2018
#Script to calculate tangential velocity from the radial velocity and full 2D wind field

def calc_vt(in_path, out_file):

    from netCDF4 import Dataset
    import numpy as np
    import copy as cp
    from find_tracks import trackID
    from find_tracks import track_time
    from find_tracks import COM_x
    from find_tracks import COM_y

    # find number of time steps and domain size
    data_in_wind2D = Dataset(in_path + in_file + 'wind2D_rainbegin.nc', 'r', format='NETCDF4')
    data_in_vr = Dataset(out_path + 'test_changeinterpolation/' + in_file + 'edge_rainbegin.nc', 'r', format='NETCDF4')
    xylist = '/conv1/fournier/pycharm-2018.2.1/projects/test_changeinterpolation/xylist.txt'  # (time, yt, xt, cp, COMy, COMx)

    time_steps = data_in_vr.variables['time'].size
    dim_x = 320
    dim_y = 320

    # first and last time step
    t0 = 0
    t1 = time_steps
    dt = 300  # temporal resolution is 300 seconds = 5 minutes
    dxdy = 200  # horizontal resolution is 200 meters

    # generate output file
    data_out = Dataset(out_file, 'w', format='NETCDF4')
    xt = data_out.createDimension('xt', dim_x)
    yt = data_out.createDimension('yt', dim_y)
    time = data_out.createDimension('time', time_steps)
    cp_nmb = data_out.createDimension('cp_nmb', None)

    # createVariable makes a variable of name 'time', format 'f8' and dimension
    # of the variable 'time'
    time_out = data_out.createVariable('time', 'f8', ('time',))
    xt_out = data_out.createVariable('xt', 'f8', ('xt',))
    yt_out = data_out.createVariable('yt', 'f8', ('yt',))
    cp_ID_out = data_out.createVariable('cp_ID', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
    vt_out = data_out.createVariable('vt', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
    vrscp_out = data_out.createVariable('vrscp', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
    vrtheta_out = data_out.createVariable('vrtheta', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))

    time_in = np.array(data_in_vr.variables['time'])
    # just because the variables xt and yt as missing!!
    xt_in = np.linspace(1, dim_x, dim_x)
    yt_in = np.linspace(1, dim_y, dim_y)
    cp_ID_in = data_in_vr.variables['cp_ID']

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
    cp_ID_out[:] = cp_ID_in[:]

    vt_out.long_name = 'Tangential velocity'
    vt_out.units = 'm s-1'

    vrscp_out.long_name = 'Radial velocity - scalar projection'
    vrscp_out.units = 'm s-1'

    vrtheta_out.long_name = 'Radial velocity - cosinus calc'
    vrtheta_out.units = 'm s-1'

    # Creating a grid that resembles the grid of the model
    x = np.linspace(1, dim_x, dim_x).astype(int)
    y = np.linspace(1, dim_y, dim_y).astype(int)
    grid = np.array(np.meshgrid(y, x, indexing='ij'))

    xyfile = open(xylist,"r")
    lines = xyfile.readlines()

    xytime = []
    xyID = []
    COMx = []
    COMy = []

    for x in lines:
        xytime.append(x.split(',')[0])
        xyID.append(x.split(',')[3])
        COMy.append(x.split(',')[4])
        COMx.append(x.split(',')[5])

    xytime = [float(i) for i in xytime]
    xyID = [float(i) for i in xyID]
    xytime = [int(i) for i in xytime]
    xyID = [int(i) for i in xyID]
    COMy = [float(i) for i in COMy]
    COMx = [float(i) for i in COMx]
    xytime = np.array(xytime)
    xyID = np.array(xyID)
    COMx = np.array(COMx)
    COMy = np.array(COMy)

    # Looping over time
    for ts in range(t0, t1):

        print ('t=', ts, '/', time_steps)

        cpID_in = np.array(data_in_vr.variables['cp_ID'][:, 0, 0, ts])
        if cpID_in[0] > 260:
            continue
        cpID_in = cpID_in[cpID_in < 260]

        COMtx = np.unique(COMx[xytime == ts])
        COMty = np.unique(COMy[xytime == ts])
        cpIDt = np.unique(xyID[xytime == ts])
        vt = np.zeros(shape=(len(cpID_in), dim_y, dim_x))
        vrscp = np.zeros(shape=(len(cpID_in), dim_y, dim_x))
        vrtheta = np.zeros(shape=(len(cpID_in), dim_y, dim_x))

        wind2D_in = np.array(data_in_wind2D.variables['wind2D'][ts, :, :])
        wind2Ddir_in = np.array(data_in_wind2D.variables['wind2D_dir'][ts, :, :])
        wind2D_pos = np.array(
               [np.multiply(wind2D_in, np.sin(wind2Ddir_in)), np.multiply(wind2D_in, np.cos(wind2Ddir_in))])

        for nmb in range(0,len(cpID_in)):
            COM = np.array([COMty[cpIDt == cpID_in[nmb]], COMtx[cpIDt == cpID_in[nmb]]])

            vec_rad = np.reshape([grid[k,i,j] - COM[k] for k in range(2) for i in range(dim_y) for j in range(dim_x)], (2,dim_y,dim_x))
            #vec_rad_dir = np.arctan2(vec_rad[1,:,:], vec_rad[0,:,:])
            #vec_rad_dir[vec_rad_dir < 0] = (np.add(2*np.pi,vec_rad_dir[vec_rad_dir < 0])) #changing the range to be 0 to 2pi (0 to 360 degrees) - if the sign is less than 0, add 2pi
            abdot = wind2D_pos[0,:,:] * vec_rad[0,:,:] + wind2D_pos[1,:,:] * vec_rad[1,:,:]
            blensq = np.sqrt(vec_rad[0,:,:] * vec_rad[0,:,:] + vec_rad[1,:,:] * vec_rad[1,:,:])
            vrscp[nmb, :, :] = abdot.astype(float) / blensq.astype(float)
            alensq = np.sqrt(wind2D_pos[0,:,:] * wind2D_pos[0,:,:] + wind2D_pos[1,:,:] * wind2D_pos[1,:,:])
            vrtheta[nmb, :, :] = alensq*np.cos(wind2Ddir_in)
            vt[nmb, :, :] = alensq*np.sin(wind2Ddir_in)

            #bbdot = vec_rad[0,:,:] * vec_rad[0,:,:] + vec_rad[1,:,:] * vec_rad[1,:,:]
            #abdotdivbbdot = abdot / bbdot
            #vrproj = abdotdivbbdot * vec_rad
            #vec_rejec = wind2D_pos - vrproj
            #vt[nmb, :, :] = np.sqrt(vec_rejec[0,:,:] * vec_rejec[0,:,:] + vec_rejec[1,:,:] * vec_rejec[1,:,:])

        vt_out[:, :, :, ts] = vt[:, :, :]
        vrscp_out[:, :, :, ts] = vrscp[:, :, :]
        vrtheta_out[:, :, :, ts] = vrtheta[:, :, :]

    data_in_wind2D.close()
    data_in_vr.close()

    data_out.close()

    return()

# Main program

from netCDF4 import Dataset
import numpy as np
import os
import sys

in_path = '/conv1/fournier/test1plus4K/rain_begin/'
out_path = in_path + 'nc_output/'
in_file = 'test1plus4K.out.vol.'
out_file = out_path + in_file + 'vt_rainbegin.nc'
print(in_path)
print(out_file)

program = calc_vt(in_path, out_file)