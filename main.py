import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import casadi as ca

from track import * 
from PID import *

from optimize_ import *



''' Initialize the environment '''

## Set the directory
dir = './'# os.path.expanduser('~') + '/Desktop/IterativeTrackOptimization/'
center_dir = dir + 'track/center_traj_with_boundary.txt' 
inner_dir = dir + 'track/innerwall.txt'
outer_dir = dir + 'track/outerwall.txt'

## Read trajectory information
center = np.loadtxt(center_dir, delimiter=",", dtype = float)
inner = np.loadtxt(inner_dir, delimiter=",", dtype = float)
outer = np.loadtxt(outer_dir, delimiter=",", dtype = float)
s=0
rounds = 1
Nsim = 2
for i in range(0,Nsim):
    # idx = np.random.randint(0,len(center))
    
    idx = 0
    N = 98
    new_center = center[idx:]
    new_center = np.concatenate((new_center,center[:idx]),axis=0)
    track = Track(new_center,inner,outer)
    curvature_info = track.get_curvature_steps(N)
    s_values = [info[0] for info in curvature_info]
    
    model = KinematicBicycle(track, N=N )
    ego_history, ego_sim_state, egost_list, = run_pid_warmstart(track, model, t=0.0)
    # print(ego_history)
    # ego_sim_state = model.vehicle_state.initialize()
    # compose_history = lambda state_history, input_history: (np.array(state_history), np.array(input_history))
    # ego_history = compose_history(np.ones((N,3)),np.ones((N,2)))

    # if ego_sim_state.v <=0 : 
    #     ego_sim_state.v =0.5
        
    # if i>= 1:
    #     ego_sim_state.v = speed[-1]
    #     print(ego_sim_state.v)
    mpcc_ego_controller = Optimize(model, track, opt_params)
    
    mpcc_ego_controller.set_warm_start(*ego_history)

    q,u,states  = mpcc_ego_controller.solve_optimization(ego_sim_state)
    q = np.insert(q, 3, s_values, axis=1)
    #print("q",q)
   
    
    # if optimized_s_values[-1] >= track.track_length:
    #     idx_s = np.where(optimized_s_values >= track.track_length)[0][0]
    # else:
    #     idx_s = -1
        
    # if optimized_s_values[-1] >= track.track_length * 2:
    #     idx_s2 = np.where(optimized_s_values >= track.track_length * 2)[0][0]
    # else:
    #     idx_s2 = -1
    round_indices = [0]
    for i in range(1,rounds+1):
        if q[-1,3] >= track.track_length * i:
            idx = np.where(q[:,3] >= track.track_length * i)[0][0]
            round_indices.append(idx)
    # round_indices.append(len(q))
    #print("round",round_indices)
    
    
    for i in range(len(round_indices)-1):
    
    # for i in range(6):
        start, end = round_indices[i], round_indices[i+1]
        if start !=end and end <= len(q):   
            round_q= q[start:end]
            #print("tq",round_q,round_q.shape)
            v_value = round_q[:,2].flatten()
            v_value = np.where(v_value==0,np.inf, v_value)
            start_s = round_q[0,3]
            end_s = round_q[-1,3]
            step_s = (end_s - start_s) / N
            s_values = np.linspace(start_s,end_s,len(v_value))    
            if s_values[-1] != end_s:
                s_values = np.append(s_values,end_s)
            
            print("v",v_value,v_value.shape)
            print(f"Round {i+1} v_value shape:", v_value.shape)
            print(f"Round {i+1} s_values shape:", s_values.shape)
            opt_t = np.trapz(1/v_value, s_values)
            print(f"Round {i+1} opt",opt_t)
            
        
        if start != end and len(q[start:end,2]) > 0:
            points = states[start:end,:2].reshape(-1, 1, 2)
            speed = q[start:end,2]
            
            ## Save the optimized trajectory
            
            traj = np.zeros((len(speed),6))
            traj[:,:2] = states[start:end,:2]
            dx = traj[2, 0] - traj[1, 0]
            dy = traj[2, 1] - traj[1, 1]
            angle = np.arctan2(dy, dx)
            ds_values = np.diff(s_values)
            ds_values = np.insert(ds_values, 0, s_values[0])
            print("s",ds_values,ds_values.shape)
            #t_values = ds_values / v_value[:-1]
            
            t_values = np.cumsum(ds_values / v_value[:])
            print("Real",t_values, t_values.shape)
            
            # Estimate the first row's values
            traj[0, 0] = traj[1, 0] - np.cos(angle) * np.linalg.norm(traj[2, :2] - traj[1, :2])
            traj[0, 1] = traj[1, 1] - np.sin(angle) * np.linalg.norm(traj[2, :2] - traj[1, :2])
            traj[:,2] = states[start:end,2]
            traj[:,3] = q[start:end,2]
            traj = np.hstack((traj, round_q, t_values.reshape(-1, 1)))
            np.savetxt(f'./data/optimized_traj_round{i+1}.txt',traj, delimiter=",")
            
            fig = plt.figure()
            ax = plt.gca()
            ax.axis('equal')
            plt.plot(track.x, track.y,'--k')
            plt.plot(inner[:,0],inner[:,1], 'k')
            plt.plot(outer[:,0],outer[:,1], 'k')
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(min(speed), max(speed))    
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(speed)
            lc.set_linewidth(5)
            line = ax.add_collection(lc)
            fig.colorbar(line, ax=ax)
    
    plt.show()
    print("Trajectory Done")
    
    # if q[-1,3] >= track.track_length:
    #     idx_s = np.where(q[:,3] >= track.track_length)[0][0]
    # else:
    #     idx_s = -1

    # if q[-1,3] >= track.track_length*2:
    #     idx_s2 = np.where(q[:,3] >= track.track_length*2)[0][0]
    # else:
    #     idx_s2 = -1

    # points = states[:idx_s,:2].reshape(-1, 1, 2)
  
    # print("points",points)
    # speed = q[:idx_s,2]

    # ## Save the optimized trajectory
    # traj = np.zeros((len(speed),6))
    # traj[:,:2] = states[:idx_s,:2]
    # traj[:,2] = states[:idx_s,2]
    # traj[:,3] = q[:idx_s,2]
    # traj[:,4:] = u[:idx_s]
    
    # # np.savetxt('./data/optimized_traj'+str(i)+'.txt',traj, delimiter=",")
    # np.savetxt('./data/optimized_traj.txt',traj, delimiter=",")

    # fig = plt.figure()
    # ax = plt.gca()
    # ax.axis('equal')
    # plt.plot(track.x, track.y,'--k')
    # plt.plot(inner[:,0],inner[:,1], 'k')
    # plt.plot(outer[:,0],outer[:,1], 'k')
    # segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # norm = plt.Normalize(min(speed), max(speed))
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # lc.set_array(speed)
    # lc.set_linewidth(5)
    # line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)
    # """
    # calculate the  optimal time to complete the track
    
    # """
    # # print(q,states)
    # v_value=q[:,2].flatten()
    # v_value = np.where(v_value==0,np.inf, v_value)
        
    # start_s = q[3,-1]
    # print("start",start_s)
    # end_s= track.track_length
    # print("end", end_s)
    # step_s = end_s / model.N
    # print("Step,", step_s)
    # s_values = np.linspace(start_s,end_s,len(v_value))    # if s_values[-1] != end_s:
    # #     s_values = np.append(s_values,end_s)

    # print("v_value shape:", v_value.shape)
    # print("s_values shape:", s_values.shape)
    # opt_t = np.trapz(1/v_value, s_values)
    # print("opt",opt_t)
    # ############################
    # points2 = states[idx_s:idx_s2,:2].reshape(-1, 1, 2)
    # speed2 = q[idx_s:idx_s2,2]

    # ## Save the optimized trajectory
    # traj = np.zeros((len(speed2),4))
    # traj[:,:2] = states[idx_s:idx_s2,:2]
    # traj[:,2] = states[idx_s:idx_s2,2]
    # traj[:,3] = q[idx_s:idx_s2,3]

    # np.savetxt('./data/optimized_traj_.txt',traj, delimiter=",")

    # fig = plt.figure()
    # ax = plt.gca()
    # ax.axis('equal')
    # plt.plot(track.x, track.y,'--k')
    # plt.plot(inner[:,0],inner[:,1], 'k')
    # plt.plot(outer[:,0],outer[:,1], 'k')
    # segments = np.concatenate([points2[:-1], points2[1:]], axis=1)
    # norm = plt.Normalize(min(speed2), max(speed2))
    # lc = LineCollection(segments, cmap='viridis', norm=norm)
    # lc.set_array(speed2)
    # lc.set_linewidth(5)
    # line = ax.add_collection(lc)
    # fig.colorbar(line, ax=ax)

    # plt.show()
    # print("Trajectory Done")
