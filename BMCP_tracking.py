def calc_vr(in_path, out_file):

    from netCDF4 import Dataset
    import numpy as np
    import copy as cp
    import time as tid
    from transfer_grid import transfer_grid
    from transfer_grid import value_trans
    from transfer_grid import wind_trans
    from calc_maxdvr_first import maxdvr_first
    from calc_maxdvr_more import maxdvr_more

    # find number of time steps and domain size
    data_in = Dataset(in_path + '1200' + in_file, 'r', format='NETCDF4')

    time_steps = 1
    dim_x = data_in.groups[u'fields'].dimensions['nx'].size
    dim_y = data_in.groups[u'fields'].dimensions['ny'].size
    dim_z = data_in.groups[u'fields'].dimensions['nz'].size

    # generate output file
    data_out = Dataset(out_file, 'w', format='NETCDF4')
    xt = data_out.createDimension('xt', dim_x)
    yt = data_out.createDimension('yt', dim_y)
    zt = data_out.createDimension('zt', dim_z)
    time = data_out.createDimension('time', time_steps)
    cp_nmb = data_out.createDimension('cp_nmb', None)

    # createVariable makes a variable of name 'time', format 'f8' and dimension
    # of the variable 'time'
    xt_out = data_out.createVariable('xt', 'f8', ('xt',))
    yt_out = data_out.createVariable('yt', 'f8', ('yt',))
    cp_ID_out = data_out.createVariable('cp_ID', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
    vrad_out = data_out.createVariable('vr', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
    dvrdr_out = data_out.createVariable('dvrdr', 'f4', ('cp_nmb', 'yt', 'xt', 'time'))
    cp_edge_out = data_out.createVariable('cp_edge', 'f4', ('yt', 'xt', 'time'))

    xt_in = np.linspace(1, dim_x, dim_x)
    yt_in = np.linspace(1, dim_y, dim_y)
    zt_in = np.linspace(1, dim_z, dim_z)

    # attributes long_name and unit
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

    dvrdr_out.long_name = 'Radial derivative of the radial velocity (fourth order accuracy)'
    dvrdr_out.units = 's-1'

    # Creating a grid that resembles the grid of the model
    x = np.linspace(1, dim_x, dim_x).astype(int)
    y = np.linspace(1, dim_y, dim_y).astype(int)
    grid = np.array(np.meshgrid(y, x, indexing='ij'))

    trackIDs = np.array([1])
    xylist = np.zeros(shape=(1, 6))
    maxcp = 1
    r_save = np.zeros(shape=(time_steps, maxcp))

    cp_ID = np.zeros(shape=(len(trackIDs), dim_y, dim_x))
    vrad = np.zeros(shape=(len(trackIDs), dim_y, dim_x))
    dvrdr = np.zeros(shape=(len(trackIDs), dim_y, dim_x))
    cp_edge = np.zeros(shape=(dim_y, dim_x))
    info = np.zeros(shape=(len(trackIDs), 3, 2))
    cp_nmb_in = np.linspace(1, len(trackIDs), len(trackIDs))
    COMx_save = np.zeros(shape=(len(trackIDs)))
    COMy_save = np.zeros(shape=(len(trackIDs)))

    u_in = np.array(data_in.groups[u'fields'].variables['v'][:, :, 0])
    v_in = np.array(data_in.groups[u'fields'].variables['u'][:, :, 0])
    wind2D = np.sqrt(np.square(u_in) + np.square(v_in))
    wind2D_dir_rad = np.arctan2(v_in,u_in)
    wind2D_dir_rad[wind2D_dir_rad < 0] = (np.add(2*np.pi,wind2D_dir_rad[wind2D_dir_rad < 0]))

    COM = np.array([100.001, 100.001])
    nmb = 0
    dt = 1
    dxdy = 1
    ts = 0
    cp_ID_list = np.zeros(shape=(time_steps, maxcp))
    cp_ID_list[ts, nmb] = trackIDs[nmb]

    wind2D_pos = np.array(
               [np.multiply(wind2D, np.sin(wind2D_dir_rad)), np.multiply(wind2D, np.cos(wind2D_dir_rad))])

    # Calculate vr using cython script
    import vr
    vrad[nmb, :, :] = vr.vr_calc(dim_x, dim_y, grid, COM, wind2D_pos)
    print ('vr calculated for cp', str(trackIDs[nmb]))

    # Calculate dvr using cython script
    import dvr
    dvrdr[nmb, :, :] = dvr.dvr_calc(x, y, grid, dim_x, dim_y, COM, nmb, vrad)
    print ('dvr calculated for cp ', str(trackIDs[nmb]))

    travelmore = 80
    cp_edge, info, xylist, r_save = maxdvr_first(vrad, nmb, COM, dim_x, dim_y, dt, dxdy, travelmore, info, dvrdr, r_save, ts, cp_ID_list, cp_edge, xylist)

    vrad_out[:, :, :, ts] = vrad[:, :, :]
    dvrdr_out[:, :, :, ts] = dvrdr[:, :, :]
    cp_edge_out[:, :, ts] = cp_edge[:, :]
    cp_ID_out[:,:,:,ts] = cp_ID[:,:,:]

    data_in.close()

    data_out.close()

# Main program

from netCDF4 import Dataset
import numpy as np

in_path = '/conv1/fournier/'
out_path = in_path  # + 'nc_output/'
in_file = '_k120.nc'
out_file = out_path + 'k120_edge.nc'
print(in_path)
print(out_file)

program = calc_vr(in_path, out_file)