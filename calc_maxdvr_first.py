# Marielle B Fournier July 2018
# Subroutine for calculating the maximum derivative of the radial velocity at the first time step
# the cp is identified

import numpy as np
import copy as cp

def maxdvr_first(vr_in, nmb, COM, dimx, dimy, dt, dxdy, travelmore, info, dvrdr_in, r_save, ts, cp_ID_list, cp_edge, xylist, startup):

  cpID = cp_ID_list[ts, nmb]

  dimxi = dimx - 1  # Indexwise
  dimyi = dimy - 1  # Indexwise

  # The upper/lower x/y values will be used as index for the radial velocity, why they must be subtracted with one because python indexing starts at 0.
  #upper_y = int(np.ceil(COM[0])) - 1
  #lower_y = int(np.floor(COM[0])) - 1
  #upper_x = int(np.ceil(COM[1])) - 1
  #lower_x = int(np.floor(COM[1])) - 1
  # Accounting for circular boundary conditions
  #if upper_x > dimxi:
  #    upper_x = 0
  #if upper_y > dimyi:
  #    upper_y = 0
  # Finding mean distance (and amount of grid boxes) travelled
  #vr_mean = np.mean(np.absolute(np.array([vr_in[nmb, upper_y, upper_x], vr_in[nmb, upper_y, lower_x], vr_in[nmb, lower_y, upper_x], vr_in[nmb, lower_y, lower_x]])))
  #distance = vr_mean * dt
  #travel = distance / dxdy
  #grid_travel = np.round(travel) + travelmore  # Search distance is more grid boxes than what the cold pool travelled, uncertainty reasons
  # Finding indices of the involved grid boxes accounting for periodic bc
  COM_round = np.round(COM) - 1 # Still indices, remember python starts at 0
  search_up = COM_round + startup-1  # grid_travel
  search_down = COM_round - startup+1  # grid_travel
  info[nmb, :, :] = np.array([COM_round, search_up, search_down])  # Saving the indices for use in further iterations for identifying the same cold pool in multiple time steps
  search_y = np.linspace(search_down[0], search_up[0], (search_up[0] - search_down[0] + 1))  # the area to be searched in the y-direction, + 1 to include first and last point
  search_x = np.linspace(search_down[1], search_up[1], (search_up[1] - search_down[1] + 1))
  search_area = np.array(np.meshgrid(search_y, search_x, indexing='ij'))
  search_grid = cp.deepcopy(search_area) + 1  # saving for later use when dr has to be calculated, why we must add one, as it will no longer be the index but the actual point
  search_area[search_area > dimxi] = np.subtract(search_area[search_area > dimxi], dimx)
  search_area[search_area < 0] = np.add(search_area[search_area < 0], dimx)
  dvrdr_search = np.zeros(shape=(len(search_area[0]), len(search_area[1])))
  vr_search = np.zeros(shape=(len(search_area[0]), len(search_area[1])))
  for yy in range(0, len(search_area[0])):
      for xx in range(0, len(search_area[1])):
          dvrdr_search[yy, xx] = dvrdr_in[nmb, int(search_area[0, yy, xx]), int(search_area[1, yy, xx])]
          vr_search[yy, xx] = vr_in[nmb, int(search_area[0, yy, xx]), int(search_area[1, yy, xx])]
  # Saving the dvrdr field in the search area for later use
  dvrdr_new = cp.deepcopy(dvrdr_search)
  vr_new = cp.deepcopy(vr_search)

  # Coordinate transformation
  dy = search_grid[0, :, :] - COM[0] # Flipped such that y grows in the upward direction - dy is from COM to the grid point (np.flip(search_grid[0, :, :], 0) - COM[0])
  dx = search_grid[1, :, :] - COM[1]
  dr = np.round(np.sqrt(dy ** 2 + dx ** 2))
  dr[dr == 0] = 1
  phi = np.arctan2(dy, dx)
  phi[phi < 0] = (np.add(2 * np.pi, phi[phi < 0]))  # change range to 0 to 2pi
  phi_slice = (np.linspace(0, 360, 17) * np.pi) / 180  # Slices that will be averaged over, 5 slices in each quadrant -> range(0,360,21)

  # Averaging dvrdr in the radial direction in all the slices
  for j in range(1, int(np.max(dr)) + 1):
      for i in range(0, len(phi_slice) - 1):
          mean_dvrdr = np.mean(dvrdr_search[(phi > phi_slice[i]) & (phi < phi_slice[i + 1]) & (dr == j)])
          dvrdr_new[(phi > phi_slice[i]) & (phi < phi_slice[i + 1]) & (dr == j)] = mean_dvrdr
          mean_vr = np.mean(vr_search[(phi > phi_slice[i]) & (phi < phi_slice[i + 1]) & (dr == j)])
          vr_new[(phi > phi_slice[i]) & (phi < phi_slice[i + 1]) & (dr == j)] = mean_vr

  # Finding the maximum negative dvrdr in the averaged field and saving it
  # with the cold pool number representing the edge of the cold pool
  dvrdr_max_logical = np.zeros(shape=(len(dvrdr_new[0]), len(dvrdr_new[1])))

  for ii in range(0, len(phi_slice) - 1):

      max_dr = 2  # Maximum distance between cold pool edges in neighbouring slices

      if len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1])]) == 0:
          continue

      if (ii == 0) or (ii > 0 and len(dvrdr_new[(phi > phi_slice[ii - 1]) & (phi < phi_slice[ii])]) == 0):
          max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])])
          if np.any(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == 1)] == -max_dvrdr):  # If the maximum is found 1 r away, it is probably a problem with the vr
              dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == 1)] = 100
              max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1])])

          if ii == 0:
              r_save[ts, nmb] = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
              r_new = cp.deepcopy(r_save[ts, nmb])
              while np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_save[ts, nmb])]) < np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr == r_save[ts, nmb] + 1)]):
                  r_new = r_new + 2
                  if len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr)]) == 0:
                      r_new = r_new - 1
                  max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_new - max_dr) & (dr < r_new + max_dr)])
                  r_save[ts, nmb] = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
      else:
          while len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr)]) == 0:
              max_dr = max_dr + 1
          max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr > r_old - max_dr) & (dr < r_old + max_dr)])

      r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
      r_now = cp.deepcopy(r_old)

      # If the radial velocity one radial step outwards is larger than the radial velocity at the found edge, do the search again further out till this is no longer true
      while np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_now)]) < np.unique(vr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr == r_now + 1)]):
          r_now = r_now + 2
          if len(dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dr > r_now - max_dr) & (dr < r_now + max_dr)]) == 0:
              r_now = r_now - 1
              max_dvrdr = np.max(-dvrdr_new[(phi > phi_slice[ii]) & (phi < phi_slice[ii+1]) & (dr > r_now - max_dr) & (dr < r_now + max_dr)])

      r_old = np.unique(dr[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)])
      dvrdr_max_logical[(phi > phi_slice[ii]) & (phi < phi_slice[ii + 1]) & (dvrdr_new == -max_dvrdr)] = cpID

  # Inserting the found cool pool edge into the full area
  for yy in range(0, len(search_area[0])):
      for xx in range(0, len(search_area[1])):
          if dvrdr_max_logical[yy, xx] == cpID:
              cp_edge[int(search_area[0, yy, xx]), int(search_area[1, yy, xx])] = dvrdr_max_logical[yy, xx]
              xylist = np.vstack((xylist, [ts, search_grid[0, yy, xx], search_grid[1, yy, xx], cpID, COM[0], COM[1]]))
          if search_area[0, yy, xx] == COM_round[0] and search_area[1, yy, xx] == COM_round[1]:
              cp_edge[int(search_area[0, yy, xx]), int(search_area[1, yy, xx])] = -50

  return cp_edge, info, xylist, r_save

