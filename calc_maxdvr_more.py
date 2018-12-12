# Marielle B Fournier July 2018
# Subroutine for calculating the maximum derivative of the radial velocity at further time steps

import numpy as np
import copy as cp

def maxdvr_more(trackID, track_time, cpID_last, info_out, vrad, nmb, COM, dt, dxdy, travelmore, dvrdr_in, r_save, ts, cp_ID_list, cp_edge, xylist, vrad_last, info, COMy_list_old, COMx_list_old, dimx, dimy, travel):

   cpID = cp_ID_list[ts, nmb]
   cpID_last = cpID_last[np.nonzero(cpID_last)]
   cpID_list = cp_ID_list[ts,:]
   cpID_list = cpID_list[np.nonzero(cpID_list)]
   #info_in = np.array(info_out[ts - 1, :, :, :])
   info_in = np.array(info_out)
   vr_in = vrad_last
   vr_now = vrad

   COMy_list = COMy_list_old[(COMy_list_old != COM[0]) | (COMx_list_old != COM[1])]
   COMx_list = COMx_list_old[(COMx_list_old != COM[1]) | (COMy_list_old != COM[0])]

   for i in range(0, len(cpID_last)):
       if cpID_last[i] == cpID:

           # Making a list of the the cold pools current "life length".
           cp_life = list()
           for k in range(0, len(trackID)):
               if (trackID[k] == cpID) and (track_time[k] - 1 < ts):
                   cp_life.append(track_time[k])
           cp_life = np.array(cp_life)
           cp_len = ts - (np.amin(cp_life) - 1)
           # If the cold pool is older than 2 timesteps, the search area doesn't need to be expanded as much
           travelmore = 3
           if ts > 90:  #120
               travelmore = 5
           elif cp_len > 2:
               travelmore = 2

           COM_round = np.round(COM) - 1
           search_up_old = info_in[i, 1, :]
           search_down_old = info_in[i, 2, :]
           search_y_old = np.linspace(search_down_old[0], search_up_old[0], (search_up_old[0] - search_down_old[0] + 1))
           new_length = len(search_y_old)
           if new_length % 2 == 0:
               new_length = new_length + 1
           new_length = new_length / 2 - 1 / 2
           search_up_new = COM_round + new_length
           search_down_new = COM_round - new_length

           #up_vr = cp.deepcopy(search_up_new)  # Need correct indices to find mean vr
           #down_vr = cp.deepcopy(search_down_new)
           #up_vr[up_vr > dimx] = np.subtract(up_vr[up_vr > dimx], dimx+1)
           #up_vr[up_vr < 0] = np.add(up_vr[up_vr < 0], dimx+1)
           #down_vr[down_vr > dimx] = np.subtract(down_vr[down_vr > dimx], dimx+1)
           #down_vr[down_vr < 0] = np.add(down_vr[down_vr < 0], dimx+1)
           #vr_mean = np.mean(np.absolute(np.array([vr_in[i, int(up_vr[0]), int(up_vr[1])], vr_in[i, int(up_vr[0]), int(down_vr[1])],
           #    vr_in[i, int(down_vr[0]), int(up_vr[1])], vr_in[i, int(down_vr[0]), int(down_vr[1])]])))  # The mean of the four outer corners of the domain - should it be of the whole outer square?
           #distance = np.abs(vr_mean) * dt
           #travel = distance / dxdy
           grid_travel = np.round(travel) + travelmore
           search_up = search_up_new + grid_travel  # From the outer boundaries of the last domain
           search_down = search_down_new - grid_travel  # From the outer boundaries of the last domain
           info[nmb, :, :] = np.array([COM_round, search_up, search_down])
           search_y = np.linspace(search_down[0], search_up[0], (search_up[0] - search_down[0] + 1))
           search_x = np.linspace(search_down[1], search_up[1], (search_up[1] - search_down[1] + 1))
           if len(search_x) > dimx:
               search_x = np.linspace(COM_round[1]-dimx/2, COM_round[1]+dimx/2, dimx)
               search_y = np.linspace(COM_round[0]-dimy/2, COM_round[0]+dimy/2, dimy)
           search_area = np.array(np.meshgrid(search_y, search_x, indexing='ij'))
           search_grid = cp.deepcopy(search_area) + 1
           search_area[search_area > dimx] = np.subtract(search_area[search_area > dimx], dimx+1)
           search_area[search_area < 0] = np.add(search_area[search_area < 0], dimx+1)
           dvrdr_search = np.zeros(shape=(len(search_area[0]), len(search_area[1])))
           vr_search = np.zeros(shape=(len(search_area[0]), len(search_area[1])))
           vr_before = np.zeros(shape=(len(search_area[0]), len(search_area[1])))
           for yy in range(0, len(search_area[0])):
               for xx in range(0, len(search_area[1])):
                   dvrdr_search[yy, xx] = dvrdr_in[nmb, int(search_area[0, yy, xx]), int(search_area[1, yy, xx])]
                   vr_search[yy, xx] = vr_now[nmb, int(search_area[0, yy, xx]), int(search_area[1, yy, xx])]
                   vr_before[yy, xx] = vr_in[i, int(search_area[0, yy, xx]), int(search_area[1, yy, xx])]

           dvrdr_new = cp.deepcopy(dvrdr_search)
           vr_new = cp.deepcopy(vr_search)
           vr_old = cp.deepcopy(vr_before)
           dy = search_grid[0, :, :] - COM[0]  # (info_in[i,0,0]+1) #The saved COM is the index COM, why one must be added (np.flip(search_grid[0, :, :], 0) - COM[0])
           dx = search_grid[1, :, :] - COM[1]  # (info_in[i,0,1]+1)
           dr = np.round(np.sqrt(dy ** 2 + dx ** 2))
           dr[dr == 0] = 1
           phi = np.arctan2(dy, dx)
           phi[phi < 0] = (np.add(2 * np.pi, phi[phi < 0]))
           phi_slice = (np.linspace(0, 360, 17) * np.pi) / 180  # bigger domain - 8 slices in each quadrant range(0,360,33) -> 11.25 degrees. 9 slices in each quadrant = 10 degrees in each slice
           for j in range(1, int(np.max(dr)) + 1):
               for ij in range(0, len(phi_slice) - 1):
                   mean_dvrdr = np.mean(dvrdr_search[(phi > phi_slice[ij]) & (phi < phi_slice[ij + 1]) & (dr == j)])
                   dvrdr_new[(phi > phi_slice[ij]) & (phi < phi_slice[ij + 1]) & (dr == j)] = mean_dvrdr
                   mean_vr = np.mean(vr_search[(phi > phi_slice[ij]) & (phi < phi_slice[ij + 1]) & (dr == j)])
                   vr_new[(phi > phi_slice[ij]) & (phi < phi_slice[ij + 1]) & (dr == j)] = mean_vr
                   mean_old_vr = np.mean(vr_before[(phi > phi_slice[ij]) & (phi < phi_slice[ij + 1]) & (dr == j)])
                   vr_old[(phi > phi_slice[ij]) & (phi < phi_slice[ij + 1]) & (dr == j)] = mean_old_vr

           dvrdr_max_logical = np.zeros(shape=(len(dvrdr_new[0]), len(dvrdr_new[1])))

           phi_4slice = (np.linspace(0, 360, 5) * np.pi) / 180
           r_bad_1 = 0
           r_bad_2 = 0
           r_bad_3 = 0
           r_bad_4 = 0

           badCOMx = []
           badCOMy = []
           for k in range(0,len(COMx_list)):
               if np.size(search_area[:, (search_area[0,:,0] == np.round(COMy_list[k])), (search_area[1,0,:] == np.round(COMx_list[k]))]) > 0:
                   badCOMx.append(COMx_list[k])
                   badCOMy.append(COMy_list[k])

           if len(badCOMx) == 0:
               pass
           else:
               for kk in range(0,len(phi_4slice) - 1):
                   search_slice = search_area[:, (phi > phi_4slice[kk]) & (phi < phi_4slice[kk + 1])]
                   search_slice_grid = search_grid[:, (phi > phi_4slice[kk]) & (phi < phi_4slice[kk+1])]
                   dr_slice =  dr[(phi > phi_4slice[kk]) & (phi < phi_4slice[kk + 1])]
                   for k in range(0,len(badCOMy)):
                       if search_slice[:, (search_slice[0,:] == np.round(badCOMy[k])) & (search_slice[1,:] == np.round(badCOMx[k]))].size > 0:
                           badCOMygrid = search_slice_grid[:, (search_slice[0,:] == np.round(badCOMy[k])) & (search_slice[1,:] == np.round(badCOMx[k]))][0][0]
                           badCOMxgrid = search_slice_grid[:, (search_slice[0,:] == np.round(badCOMy[k])) & (search_slice[1,:] == np.round(badCOMx[k]))][1][0]
                           if kk == 0:
                               r_bad_1 = np.append(r_bad_1, dr_slice[(search_slice_grid[0,:] >= np.round(badCOMygrid)) & (search_slice_grid[1,:] >= np.round(badCOMxgrid))])
                           elif kk == 1:
                               r_bad_2 = np.append(r_bad_2, dr_slice[(search_slice_grid[0,:] >= np.round(badCOMygrid)) & (search_slice_grid[1,:] <= np.round(badCOMxgrid))])
                           elif kk == 2:
                               r_bad_3 = np.append(r_bad_3, dr_slice[(search_slice_grid[0,:] <= np.round(badCOMygrid)) & (search_slice_grid[1,:] <= np.round(badCOMxgrid))])
                           elif kk == 3:
                               r_bad_4 = np.append(r_bad_4, dr_slice[(search_slice_grid[0,:] <= np.round(badCOMygrid)) & (search_slice_grid[1,:] >= np.round(badCOMxgrid))])

           r_bad_1 = np.unique(r_bad_1)
           r_bad_2 = np.unique(r_bad_2)
           r_bad_3 = np.unique(r_bad_3)
           r_bad_4 = np.unique(r_bad_4)


           for ii in range(0, len(phi_slice) - 1):
               if phi_slice[ii + 1] < phi_4slice[1]:
                   r_cpbad = cp.deepcopy(r_bad_1)
               elif phi_slice[ii + 1] < phi_4slice[2]:
                   r_cpbad = cp.deepcopy(r_bad_2)
               elif phi_slice[ii + 1] < phi_4slice[3]:
                   r_cpbad = cp.deepcopy(r_bad_3)
               else:
                   r_cpbad = cp.deepcopy(r_bad_4)

               if len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])]) == 0:
                   continue

               max_dr = 3
               minr = np.amin(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])])
               maxr = np.amax(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])])
               problem = False

               if len(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (~np.isin(dr,r_cpbad))]) == 0:
                   max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == minr)])
                   r_old = minr
                   dvrdr_max_logical[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dvrdr_new == -max_dvrdr)] = cpID
                   continue
               if ii == 0:
                   if len(vr_old[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts - 1, i])]) == 0:
                       travel_r = 1
                   else:
                       travel_r = np.round((np.unique(vr_old[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts - 1, i])]) * dt) / dxdy)
                   r_new = r_save[ts - 1, i] + travel_r  # Remembering where the cold pool edge was last time step including the travelled part in one time step
                   while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr)]) == 0:
                       max_dr = max_dr + 1
                   while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr) & (~np.isin(dr,r_cpbad))]) == 0:
                       r_new = r_new - 1
                       if r_new <= minr:
                           break
                   if r_new <= minr:
                       max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                   else:
                       max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr) & (~np.isin(dr,r_cpbad))])
                   r_save[ts, nmb] = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
                   if np.isin(r_save[ts, nmb] + 1, r_cpbad):
                       pass
                   else:
                       # If the radial velocity one radial step outwards is larger than the radial velocity at the found edge, do the search again further out till this is no longer true
                       while np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts, nmb])]) < np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts, nmb] + 1) & (~np.isin(dr,r_cpbad))]):
                           r_new = r_new + 2
                           # If this moves r_new so far out that there are no values here, move it inwards again
                           if len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr)]) == 0:
                               r_new = r_new - 1
                           while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr) & (~np.isin(dr,r_cpbad))]) == 0:
                               r_new = r_new - 1
                               if r_new <= minr:
                                   break
                           if r_new <= minr:
                               max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                           else:
                               max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr) & (~np.isin(dr,r_cpbad))])
                           r_save[ts, nmb] = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
               elif (ii > 1 and len(dvrdr_new[(phi > phi_slice[ii - 1]) & (phi < phi_slice[ii])]) == 0):  # If the last slice was empty, but there exist other earlier slices, use their r value
                   while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr)]) == 0:
                       max_dr = max_dr + 1
                   while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_cpbad))]) == 0:
                       r_old = r_old - 1
                       if r_old <= minr:
                           break
                   if r_old <= minr:
                       max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                   else:
                       max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_cpbad))])
               elif (ii > 0 and len(dvrdr_new[(phi > phi_slice[ii - 1]) & (phi < phi_slice[ii])]) == 0):  # If the first slice was empty, treat the second slice as the first slice
                   if len(vr_old[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts - 1, i])]) == 0:
                       travel_r = 1
                   else:
                       travel_r = np.round((np.unique(vr_old[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts - 1, i])]) * dt) / dxdy)
                   r_new = r_save[ts - 1, i] + travel_r  # Remembering where the cold pool edge was last time step including the travelled part in one time step
                   while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr) & (~np.isin(dr,r_cpbad))]) == 0:
                       r_new = r_new - 1
                       if r_new <= minr:
                           break
                   if r_new <= minr:
                       max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                   else:
                       max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr) & (~np.isin(dr,r_cpbad))])
                   if np.any(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == 1)] == -max_dvrdr):
                       dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == 1)] = 100
                       max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])])
               else:
                   while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr)]) == 0:
                       max_dr = max_dr + 1  # expand max_dr if there are no values of dvrdr in the allowed range
                   while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_cpbad))]) == 0:
                       r_old = r_old - 1
                       if r_old <= minr:
                           break
                   if r_old <= minr:
                       max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                   else:
                       max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_cpbad))])
               r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
               r_now = cp.deepcopy(r_old)

               r_bad = []
               if np.isin(r_now + 1, r_cpbad):
                   pass
               else:
                   # If the radial velocity one-three radial step outwards is larger than the radial velocity at the found edge, do the search again further out till this is no longer true
                   while (np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == r_now)]) < np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == r_now + 1)])):
                       if np.isin(r_now, r_cpbad):
                           r_now = np.min(r_cpbad[1:])
                           max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == r_now)])
                           r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dvrdr_new == -max_dvrdr)])
                           break
                       elif r_now == minr:
                           break
                       r_bad = np.append(r_bad,r_now)
                       r_now = r_now + 2
                       # If this moves r_new so far out that there are no values here, move it inwards again
                       if len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr > r_now - max_dr) & (dr < r_now + max_dr)]) == 0:
                           r_now = r_now - 1
                       while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_now - max_dr) & (dr < r_now + max_dr) & (~np.isin(dr,r_bad)) & (~np.isin(dr,r_cpbad))]) == 0:
                           r_now = r_now - 1
                           if r_now <= minr:
                               break
                       if r_now <= minr:
                           max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                       else:
                           max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr > r_now - max_dr) & (dr < r_now + max_dr) & (~np.isin(dr,r_bad)) & (~np.isin(dr,r_cpbad))])
                       r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dvrdr_new == -max_dvrdr)])
                       r_now = cp.deepcopy(r_old)

               if cp_len > 2:
                   problem = False
                   # If all the radial velocities in the slice are below zero, don't go through this loop
                   if np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr < r_old)] < 0) or (cp_len >= 3) or np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])] < 0):
                       pass
                   else: # If the edge is found where vr is smaller than zero or all of the values are NaN, then re-do the search further towards the cold pool center
                       r_bad = []
                       while np.any(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)] < 0) or all(np.isnan(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)])):
                           if (r_old <= minr) or (r_old - max_dr <= minr) or (r_old >= maxr):
                               problem = True
                               break
                           r_bad = np.append(r_bad,r_old)
                           r_old = r_old - max_dr
                           # If this moves r so far in that there are no values here or all the values are nans (vr below zero), move it outwards again
                           while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False)]) == 0:
                               r_old = r_old + 1
                               if (r_old >= maxr):
                                   problem = True
                                   break
                           while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False) & (~np.isin(dr,r_cpbad))]) == 0:
                               r_old = r_old - 1
                               if r_old <= minr:
                                   break
                           if r_old <= minr:
                               max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                           else:
                               max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False) & (~np.isin(dr,r_cpbad))])
                           r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
                       if r_old <= 2: #If the edge is found at r smaller or equal to 2, check if the radial velocity is above zero and search another time
                           if np.isin(r_old + 1, r_cpbad):
                               pass
                           else:
                               while np.any(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)] < 0):
                                   if np.isin(r_old + 1, r_cpbad):
                                       break
                                   else:
                                       r_old = r_old + 1
                                       if (r_old >= maxr):
                                           problem = True
                                           break
                               while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_cpbad))]) == 0:
                                   r_old = r_old - 1
                                   if r_old <= minr:
                                       break
                               if r_old <= minr:
                                   max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                               else:
                                   max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_cpbad))])
                               r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
                       # If all of the values of vr in the slice is below zero, don't go through this loop
                       if np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr < r_old)] < 0):
                           pass
                       elif r_old > 6:  # Checking further inside the cold pool, if the mean of the velocities on the inside of the edge is less than zero, then redo the search further in
                           r_bad = []
                           while (np.mean([np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old - 1) & (np.isnan(vr_new) == False)]),np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old - 2) & (np.isnan(vr_new) == False)]), np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old - 3) & (np.isnan(vr_new) == False)])]) < -0.01) and (r_old > 3) or (np.any(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)] < 0)):
                               r_bad = np.append(r_bad,r_old)
                               r_old = r_old - max_dr
                               while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad))]) == 0 or all(np.isnan(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)])):
                                   r_old = r_old + 1
                               while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False) & (~np.isin(dr,r_cpbad))]) == 0:
                                   r_old = r_old - 1
                                   if r_old <= minr:
                                       break
                               if r_old <= minr:
                                   max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                               else:
                                   max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False) & (~np.isin(dr,r_cpbad))])
                               r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
                               if np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr < r_old)] < 0):
                                   break

               # Checking even further inside the cold pool, if it has had a long life, the risk of finding another cold pools edge is greater
               if np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])] < 0):
                   pass
               elif cp_len >= 3:
                   problem = False
                   if cp_len >= 8 and r_old > 20:
                       r_vec = np.array([int(r_old - 1), int(r_old - 2), int(r_old - 3), int(r_old - 4), int(r_old - 5), int(r_old - 6),int(r_old - 7), int(r_old - 8), int(r_old - 9), int(r_old - 10), int(r_old - 11), int(r_old - 12)])
                   elif r_old >= 8:
                       r_vec = np.array([int(r_old - 1), int(r_old - 2), int(r_old - 3), int(r_old - 4), int(r_old - 5), int(r_old - 6),int(r_old - 7)])
                   else:
                       r_vec = np.array([int(r_old - 1), int(r_old - 2), int(r_old - 3), int(r_old - 4), int(r_old - 5)])
                   vr_inside = np.array([])
                   for j in r_vec:
                       if j not in dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr < r_old)]:
                           index = np.where(r_vec == j)
                           r_vec = np.delete(r_vec, index)
                           continue
                       if len(np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == j)])) > 1: #The length will only be more than 1 if the values are NaN
                           vr_inside = np.append(vr_inside, np.nan)
                       else:
                           vr_inside = np.append(vr_inside,np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == j)]))
                   if len(vr_inside[(np.isnan(vr_inside) == False) & (vr_inside < 0)]) == 0:
                       pass
                   else:
                       rmin_neg = min(r_vec[(np.isnan(vr_inside) == False) & (vr_inside < 0)])
                       # Don't use the minimum r value if it's equal to 3 or less or if all of the values of vr at r less than this value are below zero
                       if rmin_neg <= 3 or np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr <= rmin_neg)] < 0) or np.all(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])] < 0):
                           pass
                       elif (rmin_neg <= minr) or (rmin_neg - max_dr <= minr): # Don't use the minimum r value if it is less than the minimum possible r value
                           pass
                       else:
                           r_old = rmin_neg
                           r_bad = []
                           # If the value of vr at r_old is below zero or all the values are NaN, move even further inwards
                           while np.any(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)] < 0) or all(np.isnan(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)])):
                               if (r_old <= minr) or (r_old - max_dr <= minr) or (r_old >= maxr):
                                   problem = True
                                   break
                               r_bad = np.append(r_bad,r_old)
                               vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_old)] = np.nan
                               r_old = r_old - max_dr
                               # If this moves the r value so far in that there are no values, move it outwards again
                               while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False)]) == 0 or r_old <= 3:
                                   r_old = r_old + 1
                                   if (r_old >= maxr):
                                       problem = True
                                       break
                               while len(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False) & (~np.isin(dr,r_cpbad))]) == 0:
                                   r_old = r_old - 1
                                   if r_old <= minr:
                                       break
                               if r_old <= minr:
                                   max_dvrdr = np.unique(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == minr)])
                               else:
                                   max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr) & (~np.isin(dr,r_bad)) & (np.isnan(vr_new) == False) & (~np.isin(dr,r_cpbad))])
                               r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])

               if problem:
                   dvrdr_max_logical[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)] = -999
               else:
                   dvrdr_max_logical[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dvrdr_new == -max_dvrdr)] = cpID

           for yy in range(0, len(search_area[0])):
               for xx in range(0, len(search_area[1])):
                   if dvrdr_max_logical[yy, xx] == cpID:
                       cp_edge[int(search_area[0, yy, xx]), int(search_area[1, yy, xx])] = dvrdr_max_logical[yy, xx]
                       xylist = np.vstack((xylist, [ts, search_grid[0, yy, xx], search_grid[1, yy, xx], cpID, COM[0], COM[1]]))
                   if search_area[0, yy, xx] == COM_round[0] and search_area[1, yy, xx] == COM_round[1]:
                       cp_edge[int(search_area[0, yy, xx]), int(search_area[1, yy, xx])] = -50

   return cp_edge, info, xylist, r_save

