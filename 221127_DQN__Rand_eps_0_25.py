# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:52:13 2022

@author: Enrico
"""
# -------------------------------------
# Import libraries
# -------------------------------------
import copy
import math
import matplotlib.font_manager as font_manager
import numpy as np
import os
import pandas as pd
import random
import subprocess
import time
time_0 = time.perf_counter()
import torch

from collections import deque
from matplotlib import pylab as plt
from scipy.optimize import minimize

# ---------------------------------------------------------------
# DAUB
# ---------------------------------------------------------------
# Definition of the variables
eta_E = 1.00
eta_W = 1.00
eta_F = 1.00
gamma_phi = 1.00
gamma_G = 1.00
theta_deg = 60
gamma_w = 10
K_1 = 1
gamma_B_2 = 26.50
d_10 = 0.80
n_2 = 0.35
gamma_s = 12.00
tau_f = 30
Delta = 10
sigma_s = 0

def Req_Supp_Force(theta_deg):
    global S_ci
    # Calculate additional variables
    theta_rad = math.radians(theta_deg)
    b = D/math.tan(theta_rad)
    U = 2*(D+b)
    A = D*b
    # h_w_crown = h_w_axis - D/2
    # d_w = t_crown - h_w_crown
    gamma_1_av = gamma_1
    # gamma_1_av_min = gamma_1
    phi_1_rad = math.radians(phi_1_deg)
    phi_2_rad = math.radians(phi_2_deg)
    k_0 = 1- math.sin(phi_2_rad)
    k_a = math.tan(math.radians(45)-phi_2_rad/2)**2
    K_2 = (k_0 + k_a)/2

    # Case 1
    W_ci = 0
    # Check exponential function
    if -U/A*K_1*t_crown*math.tan(phi_1_rad) > 709:
        exp_ = math.exp(709)
    else:
        exp_ = math.exp(-U/A*K_1*t_crown*math.tan(phi_1_rad))
    if t_crown <= 2*D:
        sigma_v = gamma_1_av*t_crown+sigma_s
    else:
        sigma_v = (A/U*gamma_1_av-c_1)/(K_1*math.tan(phi_1_rad)*(1-exp_)+sigma_s*exp_)    
    P_v = A*sigma_v
    G = 1/2*D**3/math.tan(theta_rad)*gamma_2_av
    T_c = c_2*D**2/(2*math.tan(theta_rad))
    T_R_2 = math.tan(phi_2_rad)*K_2*((D**2*sigma_v)/(2*math.tan(theta_rad))+(D**3*gamma_2_av)/(6*math.tan(theta_rad)))
    T = T_c + T_R_2
    E = ((G+P_v)*(math.sin(theta_rad)-math.cos(theta_rad)*math.tan(phi_2_rad))-2*T-c_2*D**2/math.sin(theta_rad))/(math.sin(theta_rad)*math.tan(phi_2_rad)+math.cos(theta_rad))
    E_ci = E*math.pi*D**2/4/D**2
    S_E_ci = eta_E*E_ci
    S_W_ci = eta_W*W_ci
    S_ci = S_E_ci+S_W_ci
    
    return -E

# -------------------------------------
# Set seed
# -------------------------------------
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# -------------------------------------
# Set font
# -------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.bf"] = "serif"
plt.rcParams["mathtext.it"] = "serif"
plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.sf"] = "serif"
plt.rcParams["mathtext.tt"] = "serif"
# -------------------------------------
# Create empty files
# -------------------------------------
f = open('Settlement.dat','w')
for kk in range(1,52):
    if kk != 52:
        f.write("0,")
    else:
        f.write("0\n")
f.close()
# -------------------------------------
# Geometry
# -------------------------------------
# Bounds
D = 10 # tunnel diameter
C_D = [min(max(random.random()*3,0.5),3)] # cover to diameter ratio
C = [C_D[0]*D] # soil cover
dx = 2 # advance rate
y_min = 1.5*D # min height ground surface
y_max = 5*D # max height ground surface
tun_len = 2000 # m
# Initial values
x_surf = [0]
y_surf = [D/2+C[0]]
# -------------------------------------
# Soil properties
# -------------------------------------
# Bounds
c_min = 0
c_max = 20
phi_min = 20
phi_max = 40
corr = -0.5 # correlation between c and phi
gamma_min = 11
gamma_max = 24
E_min = 10e3 # kPa
E_max = 100e3 # kPa
sett_max = 100 # Max. settlement for gameover (mm)
fd_y_max = 50 # Max. face displacement for gameover (mm)
# initial values
coh = [random.randrange(c_min,c_max+1,1)]
phi = [random.randrange(phi_min,phi_max+1,1)]
gamma = [random.randrange(gamma_min,gamma_max+1,1)]
E = [random.randrange(E_min,E_max+100,100)]
# Coeff. variation
COV_coh = 0.8
COV_phi = 0.5
COV_gamma = 0.1
COV_E = 0.1
# Characteristic length
l_coh = 10
l_phi = 10
l_gamma = 10
l_E = 10
prob_fault = 0.01 # Probability of fault

# -------------------------------------
# DNN Definition
# -------------------------------------
l1 = 5 # Number of state variables
# l2 = 50 # Neurons in 1st hidden layer
# l3 = 50 # Neurons in 2nd hidden layer
l4 = 5 # Number of actions
# Build DNN
# model = torch.nn.Sequential(
#     torch.nn.Linear(l1,l2),
    # torch.nn.ReLU(),
    # torch.nn.Linear(l2,l3),
    # torch.nn.ReLU(),
    # torch.nn.Linear(l3,l4)
    # )
# Load model
model = torch.load('ANN.pt')
model.eval()
# Creates a second model by making an identical copy 
# of the original copy of the original Q-network model
model2 = model2 = copy.deepcopy(model) 
model2.load_state_dict(model.state_dict()) # Copies the parameters of the original model

# Model parameters
loss_fn = torch.nn.MSELoss() # Define loss function
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
dis_fac = 0.15 # Discount factor
epsilon = 0.25 # Initial probability of random actions
eps_min = 0.0 # Minimum epsilon
episodes = 100 # Number of episodes

# Initialise variables/vectors
losses = [] # Loss vector
# Synchronises the frequency parameters
# Every 5 steps we will copy the parameters
# of the model into model2
sync_freq = 5 
mem_size = 10 # Set the total size of the experience replay memory
batch_size = 2 # Set the mini-batch size
replay = deque(maxlen=mem_size) # Creates the memory replay as a deque list
j = 0 # Counter to update the TARGET network

# Episode tracker
rew_his = [0]       # Reward history
rew_ep_his = []     # Reward history of the whole episode
tot_rew_his = []    # Total reward history vector
pf_TBM_his = []     # History of face support pressure
pf_req_his = []     # History of the req. face support
sett_surf_his = []  # History of surface settlement 
x_surf_his = []

# -------------------------------------
# Main training loop
# -------------------------------------
for i in range(1,episodes+1):
    
    # -------------------------------------
    # Generate ground surface
    # -------------------------------------
    while x_surf[-1] < tun_len:
        # change coordinates
        dy_surf = random.random()*4-2
        x_surf.append(x_surf[-1]+dx)
        y_surf.append(y_surf[-1]+dy_surf)
        y_surf[-1] = min(max(y_min,y_surf[-1]),y_max)
        # change soil properties
        r_var = (random.random()-0.5)
        dcoh = r_var*coh[-1]*COV_coh/dx
        dphi = r_var*corr*COV_phi/dx
        dgamma = r_var*gamma[-1]*COV_gamma/dx
        dE = r_var*E[-1]*COV_E/dx
        coh.append(min(max(coh[-1] + dcoh,c_min),c_max))
        phi.append(min(max(phi[-1] + dphi,phi_min),phi_max))
        gamma.append(min(max(gamma[-1] + dgamma,gamma_min),gamma_max))
        E.append(min(max(E[-1] + dE,E_min),E_max))
        # Fault
        if random.random() < prob_fault:
            coh[-1] = random.randrange(c_min,c_max+1,1)
            phi[-1] = random.randrange(phi_min,phi_max+1,1)
            gamma[-1] = random.randrange(gamma_min,gamma_max+1,1)
            E[-1] = random.randrange(E_min,E_max+100,100)
    
    # Initialise episode variables
    status = 1 # Game is on
    done = False
    # Initialise state variables
    x_TBM = [0] # Location TBM
    pf_TBM = [0] # Face support pressure provided by TBM
    sett_surf = [0] # Ground surface
    # Initialise reward variables
    tot_reward = 0
    pf_req = [0]    # Required support pressure
    pf_max = [0]    # Vertical soil stress
    lam = [0]       # Stress reduction factor
    
    # Define initial state
    state_ = np.zeros((1,l1), dtype=float, order='C')         
    # state_[0,0] = x_TBM[0]
    state_[0,0] = coh[0]/c_max 
    state_[0,1] = phi[0]/phi_max 
    state_[0,2] = E[0]/E_max
    state_[0,3] = y_surf[0]/y_max
    state_[0,4] = gamma[0]/gamma_max
    # state_[0,5] = pf_TBM[0]/200
    state1 = torch.from_numpy(state_).float()
        
    while (status == 1):
        k = len(x_TBM) # current number of rounds
        j += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        
        # Select face pressure increase with epsilon-greedy policy                
        if (random.random() < epsilon): # Select action based on epsilon-greedy strategy
            action = np.random.randint(0,l4)
        else:
            action = np.argmax(qval_) # Select action based on expected highest reward           

        # Face pressure increasing/decreasing based on action
        if action == 0:
            pf_TBM_new = 50 # kPa
        if action == 1:
            pf_TBM_new = 100 # kPa
        if action == 2:
            pf_TBM_new = 150 # kPa     
        if action == 3:
            pf_TBM_new = 200 # kPa     
        if action == 4:
            pf_TBM_new = 250 # kPa     
        # print(pf_TBM_new)
    
        # Calculate required support pressure
        k = len(x_TBM) # current number of rounds
        g = gamma[k]
        pf_max.append(g*y_surf[k])
        tan_phi = math.tan(math.radians(phi[k]))
        # pf_req.append(min(max(g*D*(((2+3*(dx/D)**(6*tan_phi))/(18*tan_phi))-0.05)-coh[k]/tan_phi,0),pf_max[k])) # kPa

        gamma_1 = g    # Soil unit weight (overburden area)
        gamma_1_min = gamma_1       # Min. soil unit weight (overburden area)
        gamma_2_av = gamma_1        # Av. soil unit weight (tunnel face area)
        c_1 = coh[k]                # Cohesion (overburden area)
        c_2 = c_1                   # Cohesion (tunnel face area)
        phi_1_deg = phi[k]          # Friction angle (overburden area)
        phi_2_deg = phi_1_deg       # Friction angle (tunnel face area)
        t_crown = y_surf[k] - D/2   # Overburden
        h_w_axis = -D/2             # Water table above axis
        # Find p_f
        res = minimize(Req_Supp_Force,theta_deg,method="Nelder-Mead")
        Area = D**2*math.pi/4
        pf_req.append(max(round(S_ci/Area,3),0))

        if pf_TBM_new < pf_req[k]:
            lam.append(1-pf_TBM_new/pf_req[k])
            sett_surf_new = (lam[k]*g*(D/2)**2/E[k])*1e3 # mm
        else:
            lam.append(0)
            sett_surf_new = 0

        # Update state
        x_TBM.append(x_TBM[-1] + dx)
        pf_TBM.append(pf_TBM_new)
        sett_surf.append(sett_surf_new)       
        
        # Reward
        reward = 0
        reward =+ 1 # At every step
        # Higher reward for lower pf
        reward = reward -pf_TBM_new/200
        # Negative reward for settlement (-1/mm)
        reward = reward - sett_surf[-1]

        # Penalty for switch
        # if pf_TBM[-1] != pf_TBM[-2]:
        #     reward = reward - 1

        if sett_surf[-1] > sett_max: # mm
            reward = reward - 100 # Game Over
            done = True

        # Check if excavation completed
        if x_TBM[-1] >= tun_len:
            # reward = reward + 100 # Completion bonus
            done = True

        # # Put a floor on cumulative penalties
        # if tot_reward + reward < -500:
        #     done = True
        
        state2_ = np.zeros((1,l1), dtype=float, order='C')
        # state2_[0,0] = x_TBM[-1]
        state2_[0,0] = coh[k]/c_max 
        state2_[0,1] = phi[k]/phi_max 
        state2_[0,2] = E[k]/E_max 
        state2_[0,3] = y_surf[k]/y_max 
        state2_[0,4] = gamma[k]/gamma_max
        # state2_[0,5] = pf_TBM[-1]/200
        state2 = torch.from_numpy(state2_).float()

        exp = (state1,action,reward,state2,done) # Creates an experience of state, reward, action and the next state
        replay.append(exp) # Adds the experience to the experience replay list
        state1 = state2
        # Randomly samples a subset of the replay list
        # if the replay list is at least as long as the mini-batch size
        # begins the mini-batch training
        if len(replay) > batch_size: 
            minibatch = random.sample(replay,batch_size) # Randomly samples a subset of the replay list
            # Separates out the components of each experience into separate mini-bacth tensors
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]) 
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
        
            # Recomputes Q values for the mini-batch of states to get gradients
            Q1 = model(state1_batch)
            with torch.no_grad():
            # Uses the TARGET network to get the max Q value for the next state
                Q2 = model2(state2_batch) 
        
            # Computes Q values for the mini-batch of the next states, 
            # but does not compute gradients
            Y = reward + (dis_fac*(1 - done_batch)*torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X,Y.detach())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            # Copies the main model parameters to the target network
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())

        # If the game is over reset status and mov number
        if done == True:
            status = 0
            # tot_reward = tot_reward + reward # Adds also last reward
            # rew_his.append(tot_reward) 
            # rew_ep_his.append(rew_his)
            # rew_his = []
            # x_surf_his.append(x_TBM)
            # # Reset
            C_D = [min(max(random.random()*3,0.5),3)] # cover to diameter ratio
            C = [C_D[0]*D] # soil cover
            x_surf = [0]
            y_surf = [D/2+C[0]]
            coh = [random.randrange(c_min,c_max+1,1)]
            phi = [random.randrange(phi_min,phi_max+1,1)]
            gamma = [random.randrange(gamma_min,gamma_max+1,1)]
            E = [random.randrange(E_min,E_max+100,100)]
            
        # Calculate total reward
        tot_reward = tot_reward + reward
        # print('x: ' + str(x_TBM[-1]) + ' Reward: ' + str(round(reward,3)) + ' Tot.Rew.: ' + str(round(tot_reward,3)))
        # Append to total reward history
        rew_his.append(tot_reward)

    if epsilon > eps_min:
        epsilon -= (1/episodes)    
    tot_rew_his.append(tot_reward)
    tot_reward = 0
    rew_ep_his.append(rew_his)
    rew_his = [0]
    x_surf_his.append(x_TBM)
    pf_TBM_his.append(pf_TBM)
    pf_req_his.append(pf_req)
    sett_surf_his.append(sett_surf)
    # Convert settlement from m to mm
    sett_mm = []
    # for s_c in sett_surf[-1]:
    #     sett_mm.append(s_c*-1000)            
    # sett_surf_his.append(sett_mm)
    # sett_surf = []
    # f = open('Settlement.dat','w')
    # for kk in range(1,202):
    #     if kk != 202:
    #         f.write("0,")
    #     else:
    #         f.write("0\n")
    # f.close()    
    print(str(i)+'/'+str(episodes)+  
          ' Tot.reward: '+ str(round(tot_rew_his[-1],3)))
losses = np.array(losses) # Converts to numpy array for plots

# -------------------------------------
# PLOTS
# -------------------------------------
# Plot reward
plt.figure()
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(1,episodes)
plt.ylim(-600,600)
plt.xlabel('Episodes',family='Times new roman',size=10)
plt.ylabel('Total reward',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
x_ = np.array(range(1, episodes+1))
plt.plot(x_,tot_rew_his,linewidth=1)
# plt.plot(tot_rew_his,linewidth=1)
# Rolling average
# rew_his_roll = []
# rew_his_roll_std = []
# for mm in range(0,len(tot_rew_his)):
#     rew_his_roll.append(pd.DataFrame(tot_rew_his).iloc[:mm].mean()[0])
#     rew_his_roll_std.append(pd.DataFrame(tot_rew_his).iloc[:mm].std()[0])
rew_his_roll = pd.DataFrame(tot_rew_his) 
rew_his_roll_mean = rew_his_roll.iloc[:].rolling(window=round(episodes/10)).mean()
rew_his_roll_std = rew_his_roll.iloc[:].rolling(window=round(episodes/10)).std()
# x_err = np.linspace(0, episodes, 1)
# plt.errorbar(x_,rew_his_roll, yerr=rew_his_roll_std,alpha=0.5,color='tab:cyan')
plt.plot(rew_his_roll_mean, linewidth=1,color='tab:orange')
# plt.fill_between(range(episodes),np.array(rew_his_roll)-np.array(rew_his_roll_std),np.array(rew_his_roll)+np.array(rew_his_roll_std),alpha=.25,color='tab:orange')
plt.fill_between(range(episodes),rew_his_roll_mean[0]-rew_his_roll_std[0],rew_his_roll_mean[0]+rew_his_roll_std[0],alpha=.25,color='tab:orange')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['Total reward','Moving average and st.dev.'],fancybox=False,edgecolor='0',prop=font,loc='lower right',facecolor='white', framealpha=1)
plt.savefig('DQN_Rew_His_FLAC_100.pdf',bbox_inches='tight')
plt.show()
# Write results in file:
fs = open("Rew_His.txt", "w")
for i in tot_rew_his:
    fs.write(str(i) + "\n")    
fs.close()

# -------------------------------------
# INSPECT EPISODES
# -------------------------------------
insp_ep_1 = 1 # Episode number to be inspected
insp_ep_1 = insp_ep_1 - 1
insp_ep_2 = int(episodes*0.25) # Episode number to be inspected
insp_ep_2 = insp_ep_2 - 1
insp_ep_3 = int(episodes*0.50) # Episode number to be inspected
insp_ep_3 = insp_ep_3 - 1
insp_ep_4 = int(episodes*0.75) # Episode number to be inspected
insp_ep_4 = insp_ep_4 - 1
insp_ep_5 = episodes # Episode number to be inspected
insp_ep_5 = insp_ep_5 - 1
# Plot reward
plt.figure()
plt.grid(visible=True,which='both',alpha=0.5)
ylim_min = -600
ylim_max = 600
for i in rew_ep_his:
    ylim_min = min(max(i),ylim_min)
    ylim_max = max(max(i),ylim_max)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(0,tun_len)
plt.ylim(ylim_min,ylim_max)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Total reward (-)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_1][:len(pf_TBM_his[insp_ep_1])],rew_ep_his[insp_ep_1],linewidth=1)
plt.plot(x_surf_his[insp_ep_2][:len(pf_TBM_his[insp_ep_2])],rew_ep_his[insp_ep_2],linewidth=1)
plt.plot(x_surf_his[insp_ep_3][:len(pf_TBM_his[insp_ep_3])],rew_ep_his[insp_ep_3],linewidth=1)
plt.plot(x_surf_his[insp_ep_4][:len(pf_TBM_his[insp_ep_4])],rew_ep_his[insp_ep_4],linewidth=1)
plt.plot(x_surf_his[insp_ep_5][:len(pf_TBM_his[insp_ep_5])],rew_ep_his[insp_ep_5],linewidth=1)
for i in range(0,episodes):
    if rew_ep_his[i][-1] - rew_ep_his[i][-2] <= -80:
        plt.plot(x_surf_his[i][:len(pf_TBM_his[i])],rew_ep_his[i],linewidth=1,color='black',alpha=0.3)
        plt.text(x_surf_his[i][len(pf_TBM_his[i])-1],rew_ep_his[i][-1]-50,'†')
    else:
        plt.plot(x_surf_his[i][:len(pf_TBM_his[i])],rew_ep_his[i],linewidth=0.5,color='gray',alpha=0.3)        
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['Episode ' + str(insp_ep_1+1),
            'Episode ' + str(insp_ep_2+1),
            'Episode ' + str(insp_ep_3+1),
            'Episode ' + str(insp_ep_4+1),
            'Episode ' + str(insp_ep_5+1)],fancybox=False,edgecolor='0',prop=font,loc='lower left',facecolor='white', framealpha=1,ncol=2)
filename = 'DQN_Rew_His_Ep_FLAC_100.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()
# Write results in file:
fs = open("Rew_His_Ep.txt", "w")
for i in rew_ep_his:
    fs.write(str(i) + "\n")    
fs.close()

# pf
# 1
ylim_min = 0
ylim_max = 400
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(2,tun_len)
plt.ylim(ylim_min,ylim_max)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Face support pressure (kPa)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_1][:len(pf_TBM_his[insp_ep_1])],pf_TBM_his[insp_ep_1],linewidth=1,alpha=1,color='tab:blue')
plt.plot(x_surf_his[insp_ep_1][:len(pf_req_his[insp_ep_1])],pf_req_his[insp_ep_1],linewidth=1,alpha=0.5,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['$p_\mathrm{f}$','$p_\mathrm{f,req}$'],
            fancybox=False,edgecolor='0',prop=font, 
            loc="upper left",facecolor='white', framealpha=1,ncol=2)
filename = 'DQN_pf_TBM_His_Ep_' + str(insp_ep_1) + '_FLAC_100.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 2
ylim_min = 0
ylim_max = 300
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(2,tun_len)
plt.ylim(ylim_min,ylim_max)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Face support pressure (kPa)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_2][:len(pf_TBM_his[insp_ep_2])],pf_TBM_his[insp_ep_2],linewidth=1,alpha=1,color='tab:orange')
plt.plot(x_surf_his[insp_ep_2][:len(pf_req_his[insp_ep_2])],pf_req_his[insp_ep_2],linewidth=1,alpha=0.5,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['$p_\mathrm{f}$','$p_\mathrm{f,req}$'],
            fancybox=False,edgecolor='0',prop=font, 
            loc="upper left",facecolor='white', framealpha=1,ncol=2)
filename = 'DQN_pf_TBM_His_Ep_' + str(insp_ep_2) + '_FLAC_100.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 3
ylim_min = 0
ylim_max = 300
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(2,tun_len)
plt.ylim(ylim_min,ylim_max)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Face support pressure (kPa)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_3][:len(pf_TBM_his[insp_ep_3])],pf_TBM_his[insp_ep_3],linewidth=1,alpha=1,color='tab:green')
plt.plot(x_surf_his[insp_ep_3][:len(pf_req_his[insp_ep_3])],pf_req_his[insp_ep_3],linewidth=1,alpha=0.5,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['$p_\mathrm{f}$','$p_\mathrm{f,req}$'],
            fancybox=False,edgecolor='0',prop=font, 
            loc="upper left",facecolor='white', framealpha=1,ncol=2)
filename = 'DQN_pf_TBM_His_Ep_' + str(insp_ep_3) + '_FLAC_100.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 4
ylim_min = 0
ylim_max = 400
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(2,tun_len)
plt.ylim(ylim_min,ylim_max)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Face support pressure (kPa)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_4][:len(pf_TBM_his[insp_ep_4])],pf_TBM_his[insp_ep_4],linewidth=1,alpha=1,color='tab:red')
plt.plot(x_surf_his[insp_ep_4][:len(pf_req_his[insp_ep_4])],pf_req_his[insp_ep_4],linewidth=1,alpha=0.5,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['$p_\mathrm{f}$','$p_\mathrm{f,req}$'],
            fancybox=False,edgecolor='0',prop=font, 
            loc="upper left",facecolor='white', framealpha=1,ncol=2)
filename = 'DQN_pf_TBM_His_Ep_' + str(insp_ep_4) + '_FLAC_100.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 5
ylim_min = 0
ylim_max = 400
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(1,tun_len)
plt.ylim(ylim_min,ylim_max)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Face support pressure (kPa)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_5][:len(pf_TBM_his[insp_ep_5])],pf_TBM_his[insp_ep_5],linewidth=1,alpha=1,color='tab:purple')
plt.plot(x_surf_his[insp_ep_5][:len(pf_req_his[insp_ep_5])],pf_req_his[insp_ep_5],linewidth=1,alpha=0.5,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
plt.legend(['$p_\mathrm{f}$','$p_\mathrm{f,req}$'],
            fancybox=False,edgecolor='0',prop=font, 
            loc="upper right",facecolor='white', framealpha=1,ncol=2)
filename = 'DQN_pf_TBM_His_Ep_' + str(insp_ep_5) + '_FLAC_100.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# Write results in file:
fs = open("pf_TBM_His_Ep.txt", "w")
for i in pf_TBM_his:
    fs.write(str(i) + "\n")    
fs.close()

# Plot settlement
ylim_min = 0
# ylim_max1 = max(sett_surf_his[insp_ep_1])
# ylim_max2 = max(sett_surf_his[insp_ep_2])
# ylim_max3 = max(sett_surf_his[insp_ep_3])
# ylim_max4 = max(sett_surf_his[insp_ep_4])
# ylim_max5 = max(sett_surf_his[insp_ep_5])
# ylim_max = round(max(ylim_max1, ylim_max2, ylim_max3, ylim_max4, ylim_max5),0)+1
ylim_max = 10
# 1
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.gca().invert_yaxis()
plt.xlim(0,tun_len)
plt.ylim(-ylim_max,0)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Surface settlement (mm)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_1][:len(sett_surf_his[insp_ep_1])],-1*np.array(sett_surf_his[insp_ep_1]),linewidth=1,alpha=0.5,color='tab:blue')
# sett_roll_1 = pd.DataFrame(sett_surf_his[insp_ep_1])
# sett_roll_1 = sett_roll_1.iloc[:].rolling(window=round(len(sett_roll_1)/10)).mean()
# plt.plot(x_surf_his[insp_ep_1][:len(sett_roll_1)],-1*sett_roll_1[0],linewidth=1,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
filename = 'DQN_Sett_Surf_His_Ep_' + str(insp_ep_1) + '.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 2
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.gca().invert_yaxis()
plt.xlim(0,tun_len)
plt.ylim(-ylim_max,0)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Surface settlement (mm)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_2][:len(sett_surf_his[insp_ep_2])],-1*np.array(sett_surf_his[insp_ep_2]),linewidth=1,alpha=0.5,color='tab:orange')
# sett_roll_1 = pd.DataFrame(sett_surf_his[insp_ep_1])
# sett_roll_1 = sett_roll_1.iloc[:].rolling(window=round(len(sett_roll_1)/10)).mean()
# plt.plot(x_surf_his[insp_ep_1][:len(sett_roll_1)],-1*sett_roll_1[0],linewidth=1,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
filename = 'DQN_Sett_Surf_His_Ep_' + str(insp_ep_2) + '.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 3
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.gca().invert_yaxis()
plt.xlim(0,tun_len)
plt.ylim(-ylim_max,0)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Surface settlement (mm)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_3][:len(sett_surf_his[insp_ep_3])],-1*np.array(sett_surf_his[insp_ep_3]),linewidth=1,alpha=0.5,color='tab:green')
# sett_roll_1 = pd.DataFrame(sett_surf_his[insp_ep_1])
# sett_roll_1 = sett_roll_1.iloc[:].rolling(window=round(len(sett_roll_1)/10)).mean()
# plt.plot(x_surf_his[insp_ep_1][:len(sett_roll_1)],-1*sett_roll_1[0],linewidth=1,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
filename = 'DQN_Sett_Surf_His_Ep_' + str(insp_ep_3) + '.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 4
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.gca().invert_yaxis()
plt.xlim(0,tun_len)
plt.ylim(-ylim_max,0)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Surface settlement (mm)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_4][:len(sett_surf_his[insp_ep_4])],-1*np.array(sett_surf_his[insp_ep_4]),linewidth=1,alpha=0.5,color='tab:red')
# sett_roll_1 = pd.DataFrame(sett_surf_his[insp_ep_1])
# sett_roll_1 = sett_roll_1.iloc[:].rolling(window=round(len(sett_roll_1)/10)).mean()
# plt.plot(x_surf_his[insp_ep_1][:len(sett_roll_1)],-1*sett_roll_1[0],linewidth=1,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
filename = 'DQN_Sett_Surf_His_Ep_' + str(insp_ep_4) + '.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# 5
plt.figure(figsize=(6, 2))
plt.grid(visible=True,which='both',alpha=0.5)
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.gca().invert_yaxis()
plt.xlim(0,tun_len)
plt.ylim(-ylim_max,0)
plt.xlabel('$x$ (m)',family='Times new roman',size=10)
plt.ylabel('Surface settlement (mm)',family='Times new roman',size=10)
plt.xticks(family='Times new roman',size=10)
plt.yticks(family='Times new roman',size=10)
plt.plot(x_surf_his[insp_ep_5][:len(sett_surf_his[insp_ep_5])],-1*np.array(sett_surf_his[insp_ep_5]),linewidth=1,alpha=0.5,color='tab:purple')
# sett_roll_1 = pd.DataFrame(sett_surf_his[insp_ep_1])
# sett_roll_1 = sett_roll_1.iloc[:].rolling(window=round(len(sett_roll_1)/10)).mean()
# plt.plot(x_surf_his[insp_ep_1][:len(sett_roll_1)],-1*sett_roll_1[0],linewidth=1,color='black')
font = font_manager.FontProperties(family='Times new roman',
                                    size=10)
filename = 'DQN_Sett_Surf_His_Ep_' + str(insp_ep_5) + '.pdf'
plt.savefig(filename,bbox_inches='tight')
plt.show()

# Write results in file:
fs = open("Sett_Surf_His_Ep.txt", "w")
for i in sett_surf_his:
    fs.write(str(i) + "\n")    
fs.close()

# # Compare rolling averages
# plt.figure()
# plt.grid(visible=True,which='both',alpha=0.5)
# for pos in ['right', 'top', 'bottom', 'left']:
#     plt.gca().spines[pos].set_linewidth(1)
# plt.gca().invert_yaxis()
# plt.xlim(0,tun_len)
# plt.ylim(-50,0)
# plt.xlabel('$x$ (m)',family='Times new roman',size=10)
# plt.ylabel('Surface settlement (mm)',family='Times new roman',size=10)
# plt.xticks(family='Times new roman',size=10)
# plt.yticks(family='Times new roman',size=10)
# plt.plot(x_surf_his[insp_ep_1][:len(sett_roll_1)],-1*sett_roll_1[0],linewidth=1)
# plt.plot(x_surf_his[insp_ep_2][:len(sett_roll_2)],-1*sett_roll_2[0],linewidth=1)
# plt.plot(x_surf_his[insp_ep_3][:len(sett_roll_3)],-1*sett_roll_3[0],linewidth=1)
# plt.plot(x_surf_his[insp_ep_4][:len(sett_roll_4)],-1*sett_roll_4[0],linewidth=1)
# plt.plot(x_surf_his[insp_ep_5][:len(sett_roll_5)],-1*sett_roll_5[0],linewidth=1)
# for i in range(0,100):
#     sett_roll_ = pd.DataFrame(sett_surf_his[i])
#     sett_roll_ = sett_roll_.iloc[:].rolling(window=round(len(sett_roll_)/10)).mean()
#     plt.plot(x_surf_his[i][:len(sett_roll_)],-1*sett_roll_[0],linewidth=0.5,color='gray',alpha=0.2)    
# font = font_manager.FontProperties(family='Times new roman',
#                                     size=10)
# plt.legend(['Episode 1','Episode 25', 'Episode 50', 'Episode 75', 'Episode 100'],fancybox=False,edgecolor='0',prop=font,loc='lower right',facecolor='white', framealpha=1)
# filename = 'DQN_Sett_Surf_Roll.pdf'
# plt.savefig(filename)
# plt.show()

# Stats
print('Min: ' + str(round(min(tot_rew_his),3)))
print('Q1: ' + str(round(np.percentile(tot_rew_his,25),3)))
print('Median: ' + str(round(np.percentile(tot_rew_his,50),3)))
print('Q3: ' + str(round(np.percentile(tot_rew_his,75),3)))
print('Max: ' + str(round(max(tot_rew_his),3)))
print()
print('Mean: ' + str(round(np.average(tot_rew_his),3)))
print('St.Dev: ' + str(round(np.std(tot_rew_his),3)))
# Save model
torch.save(model, 'ANN_Rand.pt')

# Write results in file:
fs = open("0.txt", "w")
for i in tot_rew_his:
    fs.write(str(i) + "\n")    
fs.close()

time_1 = time.perf_counter()
print()
print('Time elapsed: ' + str(round(time_1-time_0,2))+' sec')


