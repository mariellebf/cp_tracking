# Marielle B Fournier, March 2018
# This subroutine searches the irt_tracks_output_nospace.txt file to get the track ID, time step and
# center of mass.

import numpy as np

input_dir = '/conv1/fournier/UCLA_lind/'   # '/conv1/fournier/test1plus4K/rain_begin/minsizelarger/'

head_file = open(input_dir+"irt_tracks_head.txt","r")
lines = head_file.readlines()

ID = []
duration = []
beginning = []
end = []

for x in lines:
    ID.append(x.split(' ')[0])
    duration.append(x.split(' ')[2])
    beginning.append(x.split(' ')[3])
    end.append(x.split(' ')[4])

ID = [int(i) for i in ID]
duration = [int(i) for i in duration]
beginning = [int(i) for i in beginning]
end = [int(i) for i in end]
ID = np.array(ID)
duration = np.array(duration)
beginning = np.array(beginning)
end = np.array(end)

# Don't include cold pools of shorter duration than two
ID = ID[duration > 2]
beginning = beginning[duration > 2]
end = end[duration > 2]
duration = duration[duration > 2]
# Don't include cold pools which result from a splitting event
ID = ID[beginning != 1]
duration = duration[beginning != 1]
end = end[beginning != 1]
beginning = beginning[beginning != 1]
# Cold pools that end because of a merging event
IDmerge = ID[end == 1]

COM_file = open(input_dir+"irt_tracks_nohead.txt","r")
lines = COM_file.readlines()

trackID = []
track_time = []
COM_x = []
COM_y = []

for x in lines:
    trackID.append(x.split(' ')[0])
    track_time.append(x.split(' ')[1])
    COM_x.append(x.split(' ')[11])
    COM_y.append(x.split(' ')[12])

trackID = [int(i) for i in trackID]
track_time = [int(i) for i in track_time]
COM_x = [float(i) for i in COM_x]
COM_y = [float(i) for i in COM_y]
trackID_arr = np.array(trackID)
tracktime_arr = np.array(track_time)

trackIDs = np.unique(trackID)
badtrack = []

for i in range(0,len(trackIDs)):
    if len(tracktime_arr[trackID_arr == trackIDs[i]]) <= 2:
        badtrack = np.append(badtrack,trackIDs[i])
    tracktime_arr[trackID_arr == trackIDs[i]] = tracktime_arr[trackID_arr == trackIDs[i]] - tracktime_arr[trackID_arr == trackIDs[i]][0]

track_time = np.array(track_time)
trackID = np.array(trackID)
COM_y = np.array(COM_y)
COM_x = np.array(COM_x)
track_time = track_time[~np.isin(trackID,badtrack)]
COM_x = COM_x[~np.isin(trackID,badtrack)]
COM_y = COM_y[~np.isin(trackID,badtrack)]
tracktime_arr = tracktime_arr[~np.isin(trackID,badtrack)]
trackID = trackID[~np.isin(trackID,badtrack)]
track_time = track_time[np.isin(trackID,ID)]
COM_x = COM_x[np.isin(trackID,ID)]
COM_y = COM_y[np.isin(trackID,ID)]
tracktime_arr = tracktime_arr[np.isin(trackID,ID)]
trackID = trackID[np.isin(trackID,ID)]



