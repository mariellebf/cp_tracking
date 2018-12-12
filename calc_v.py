# Marielle B Fournier, August 2018
# Subroutine for calculating the magnitude and direction of the resultant 2D wind field at multiple layers
def calc_v(in_path,out_file):

    from netCDF4 import Dataset
    import numpy as np
    import sys
    import os #to use path from bashrc
    import matplotlib
    import matplotlib.pyplot as plt

    # find out number of timesteps and domain size:
    data_in_u = Dataset(in_path+in_file+'.out.vol.u.nc', 'r', format='NETCDF4')
    data_in_v = Dataset(in_path+in_file+'.out.vol.v.nc', 'r', format='NETCDF4')

    time_steps = data_in_u.variables['time'].size
    dim_z = 75
    dim_x = 1024
    dim_y = 1024

    #first and last time step
    t0 = 0
    t1 = time_steps

    #generate output file
    data_out = Dataset(out_file,'w',format='NETCDF4')
    xt = data_out.createDimension('xt',dim_x)
    yt = data_out.createDimension('yt',dim_y)
    zt = data_out.createDimension('zt',dim_z)
    time = data_out.createDimension('time', None)

    # createVariable makes a variable of name 'time', format 'f8' and dimension
    # of the variable 'time'
    time_out = data_out.createVariable('time','f8',('time',))
    xt_out = data_out.createVariable('xt','f8',('xt',))
    yt_out = data_out.createVariable('yt','f8',('yt',))
    zt_out = data_out.createVariable('zt','f8',('zt',))

    time_in = numpy.array(data_in_u.variables['time'])
    # just because the variables xt and yt as missing!!
    xt_in = numpy.linspace(1,dim_x,dim_x)
    yt_in = numpy.linspace(1,dim_y,dim_y)
    zt_in = numpy.array(data_in_u.variables['zt'])
    #zt_in = zt_initial[1:12]

    # create output variables
    wind2D_out = data_out.createVariable('wind2D','f4',('time','yt','xt','zt'))
    wind2D_dir_out = data_out.createVariable('wind2D_dir','f4',('time','yt','xt','zt'))
    #vec_rad_dir_out = data_out.createVariable('vec_rad_dir','f4',('time','yt','xt','zt'))

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

    zt_out.long_name = 'Vertical displacement of cell centers'
    zt_out.units = 'm'
    zt_out[:] = zt_in

    wind2D_out.long_name = 'Horizontal winds'
    wind2D_out.units = 'm s-1'

    wind2D_dir_out.long_name = 'Horizontal wind direction'
    wind2D_dir_out.units = 'degrees'

    # loop over time
    for ts in range(t0,t1):

        print ('t=',ts,'/',time_steps)

        u_new = np.zeros(shape=(dim_y, dim_x, dim_z))
        v_new = np.zeros(shape=(dim_y, dim_x, dim_z))
        wind2D = np.zeros(shape=(dim_y, dim_x, dim_z))
        wind2D_dir = np.zeros(shape=(dim_y, dim_x, dim_z))

        for zz in range(1,len(zt_in)):
            u_in = np.array(data_in_u.variables['u'][ts,:,:,zz])
            v_in = np.array(data_in_v.variables['v'][ts,:,:,zz])

            #u and v are staggered half a grid point up-grid of the thermodynamic variables
            #For this purpose we would like to have the resultant wind vector at the same grid point
            #as the thermodynamic variables, why interpolation must be done (average between the value at the gridpoint before and after the current value)

            for xx in range(0,dim_x):
                if xx == 0:
                    u_new[:,xx,zz-1] = (u_in[:,-1] + u_in[:,xx]) / 2
                else:
                    u_new[:,xx,zz-1] = (u_in[:,xx-1] + u_in[:,xx]) / 2

            for yy in range(0,dim_y):
                if yy == 0:
                    v_new[yy,:,zz-1] = (v_in[-1,:] + v_in[yy,:]) / 2
                else:
                    v_new[yy,:,zz-1] = (v_in[yy-1,:] + v_in[yy,:]) / 2

        #vec_rad_dir = np.zeros(shape=(dim_y,dim_x))
        #point = [160,160]
        #x = np.linspace(1,dim_x,dim_x)
        #y = np.linspace(1,dim_y,dim_y)
        #grid = np.array(np.meshgrid(y, x))#,indexing='ij'))
        #for ite_x in range(0,dim_x):
        #        for ite_y in range(0,dim_y):
        #            vec_rad = np.subtract(grid[:,ite_y, ite_x],point)
        #            vec_rad_dir[ite_y,ite_x] = np.arctan2(vec_rad[1],vec_rad[0])


        wind2D = np.sqrt(np.square(u_new) + np.square(v_new))
        wind2D_dir_rad = np.arctan2(v_new,u_new) #using arctan2 makes the range of the output -pi to pi instead of -pi/2 to pi/2 (mapping full domain) and hence takes the sign of the wind into account
        wind2D_dir_rad[wind2D_dir_rad < 0] = (np.add(2*np.pi,wind2D_dir_rad[wind2D_dir_rad < 0])) #changing the range to be 0 to 2pi (0 to 360 degrees) - if the sign is less than 0, add 2pi

        #vec_rad_dir[vec_rad_dir < 0] = (np.add(2*np.pi,vec_rad_dir[vec_rad_dir < 0]))
        #vec_rad_dir_out[ts,:,:] = vec_rad_dir[:,:]
        wind2D_out[ts,:,:] = wind2D[:,:]
        wind2D_dir_out[ts,:,:] = wind2D_dir_rad[:,:]

        #reshaping arrays for scatterplot
        #wind2D_xdata = np.reshape(wind2D,102400)
        #wind2Ddir_ydata = np.reshape(wind2D_dir_rad,102400)

        #plt.scatter(wind2D_xdata,wind2Ddir_ydata)
        #plt.savefig(in_path+'rain_begin/plots/scatter_mag_dir_'+str(ts)+'.png')
        #plt.close()

    data_in_u.close()
    data_in_v.close()

    data_out.close()
    return;

# Main program

from netCDF4 import Dataset
import numpy
import os
import sys

in_path = '/conv1/fournier/UCLA_lind/'
out_path = in_path
in_file = 'lind_p2K'
out_file = out_path+in_file+'.out.vol.wind2D.nc'
print(in_path)
print(out_file)

calc_v(in_path,out_file)
