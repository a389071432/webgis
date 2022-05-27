# classes
import random
import math
import numpy as np
import time
from numba import cuda,guvectorize
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32,create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import GPUtil
import psutil
import threading
from flask import request
import json
from flask import Flask
from flask_cors import CORS
from flask import render_template,jsonify

class Agents:
    def __init__(self):  #以前端data为输入，目前没想好
        self.data=None
        #基本参数
        self.n_agents=None
        self.agent_activity_type =None
        self.agent_activity_duration =None
        self.agent_activity_distance = None
        self.agent_activity_total =None
        self.agent_commuting_distance =None
        self.agent_age=None

    def set_data(self,data):  #接收前端数据
        self.data=data

    def data_to_numpy(self):
        self.n_agents=15000

        # activity type统一编号如下：
        # 0:H 1:W 2:L 3:S
        self.agent_activity_type = np.zeros([self.n_agents, 8]).astype(int)
        self.agent_activity_duration = np.zeros([self.n_agents, 8])
        self.agent_activity_distance = np.zeros([self.n_agents, 8])
        self.agent_activity_position=np.zeros([self.n_agents,8]).astype(int)
        self.agent_activity_total = np.zeros(self.n_agents).astype(int)
        self.agent_commuting_distance=np.zeros(self.n_agents)
        self.agent_age=np.zeros(self.n_agents).astype(int)
        self.agent_family=np.zeros(self.n_agents).astype(int)
        #随机生成agent数据
        #活动模式编号
        #0: H-W-H-W-H
        #1: H-W-L-W-H
        #2: H-L-H
        for i in range(0,self.n_agents):
            self.agent_age[i]=random.randint(0,2)
        for i in range(0,self.n_agents):
            type=random.randint(0,2)
            if type==0:
                self.agent_activity_total[i] = 5

                self.agent_activity_type[i][0]=0
                self.agent_activity_type[i][1]=1
                self.agent_activity_type[i][2] = 0
                self.agent_activity_type[i][3] = 1
                self.agent_activity_type[i][4] = 0

                self.agent_activity_duration[i][0]=random.normalvariate(7.5, 0.6)
                self.agent_activity_duration[i][1] = random.normalvariate(3, 0.2)
                self.agent_activity_duration[i][2] = random.normalvariate(1.5, 0.2)
                self.agent_activity_duration[i][3] = random.normalvariate(5, 1)
                self.agent_activity_duration[i][4] =24-self.agent_activity_duration[i][0]-self.agent_activity_duration[i][1]-self.agent_activity_duration[i][2]-self.agent_activity_duration[i][3]
            elif type==1:
                self.agent_activity_total[i] = 5

                self.agent_activity_type[i][0]=0
                self.agent_activity_type[i][1]=1
                self.agent_activity_type[i][2] = 2
                self.agent_activity_type[i][3] = 1
                self.agent_activity_type[i][4] = 0

                self.agent_activity_duration[i][0]=random.normalvariate(7.5, 0.6)
                self.agent_activity_duration[i][1] = random.normalvariate(3, 0.2)
                self.agent_activity_duration[i][2] = random.normalvariate(1.5, 0.2)
                self.agent_activity_duration[i][3] = random.normalvariate(5, 1)
                self.agent_activity_duration[i][4] =24-self.agent_activity_duration[i][0]-self.agent_activity_duration[i][1]-self.agent_activity_duration[i][2]-self.agent_activity_duration[i][3]

                self.agent_activity_distance[i][1]=random.randint(500,1500)
                self.agent_activity_distance[i][2]=self.agent_activity_distance[i][1]
            elif type==2:
                self.agent_activity_total[i] = 3

                self.agent_activity_type[i][0]=0
                self.agent_activity_type[i][1]=2
                self.agent_activity_type[i][2] = 0

                self.agent_activity_duration[i][0]=random.normalvariate(7.5, 0.6)
                self.agent_activity_duration[i][1] = random.normalvariate(3, 0.2)
                self.agent_activity_duration[i][4] =12-self.agent_activity_duration[i][0]-self.agent_activity_duration[i][1]-self.agent_activity_duration[i][2]

                self.agent_activity_distance[i][0] = random.randint(1000,3000)
                self.agent_activity_distance[i][1] = self.agent_activity_distance[i][0]

        for i in range(0, math.ceil(0.306 * self.n_agents)):
            if self.agent_activity_total[i] == 5:
                self.agent_activity_distance[i][0] = random.randint(1500, 5000)
                self.agent_activity_distance[i][3] = self.agent_activity_distance[i][0]
                if self.agent_activity_type[i][2] == 0:
                    self.agent_activity_distance[i][0] = random.randint(1500, 5000)
                    self.agent_activity_distance[i][3] = self.agent_activity_distance[i][0]
                self.agent_commuting_distance[i] = self.agent_activity_distance[i][0]
            else:
                self.agent_commuting_distance[i] = -1

        for i in range(math.ceil(0.306 * self.n_agents), math.ceil(0.66 * self.n_agents)):
            if self.agent_activity_total[i] == 5:
                self.agent_activity_distance[i][0] = random.randint(5000, 10000)
                self.agent_activity_distance[i][3] = self.agent_activity_distance[i][0]
                if self.agent_activity_type[i][2] == 0:
                    self.agent_activity_distance[i][0] = random.randint(5000, 10000)
                    self.agent_activity_distance[i][3] = self.agent_activity_distance[i][0]
                self.agent_commuting_distance[i] = self.agent_activity_distance[i][0]
            else:
                self.agent_commuting_distance[i] = -1

        for i in range(math.ceil(0.66 * self.n_agents), math.ceil(self.n_agents)):
            if self.agent_activity_total[i] == 5:
                self.agent_activity_distance[i][0] = random.randint(10000, 12000)
                self.agent_activity_distance[i][3] = self.agent_activity_distance[i][0]
                if self.agent_activity_type[i][2] == 0:
                    self.agent_activity_distance[i][0] = random.randint(10000, 12000)
                    self.agent_activity_distance[i][3] = self.agent_activity_distance[i][0]
                self.agent_commuting_distance[i] = self.agent_activity_distance[i][0]
            else:
                self.agent_commuting_distance[i] = -1

@cuda.jit
def init_building_free_list_gpu(building_agents_free_list,building_agents_free_list_front,building_agents_free_list_rear,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    for i in range(0,building_agents_free_list.shape[1]-1):
        building_agents_free_list[idx][i]=i
    building_agents_free_list_front[idx]=0
    building_agents_free_list_rear[idx]=building_agents_free_list.shape[1]-1

@cuda.jit
def init_infected_trasition_events_gpu(agent_infected_status,agent_next_infected_staus,new_event_mutex,new_event_pointer,new_event_type,new_event_agent,new_event_delay,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            if agent_infected_status[now_id]>0:
                agent_next_infected_staus[now_id]=1
                while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                    continue
                new_event_type[new_event_pointer[0]] = 2
                new_event_delay[new_event_pointer[0]] =2
                new_event_agent[new_event_pointer[0]] = now_id
                new_event_pointer[0] += 1
                cuda.atomic.exch(new_event_mutex, 0, 0)

@cuda.jit
def init_activity_trasition_events_gpu(agent_activity_duration,new_event_mutex,new_event_pointer, new_event_type,new_event_delay, new_event_agent,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            delay=agent_activity_duration[now_id][0]
            while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                continue
            new_event_type[new_event_pointer[0]] =1
            new_event_delay[new_event_pointer[0]] = delay
            new_event_agent[new_event_pointer[0]] = now_id
            new_event_pointer[0]+=1
            cuda.atomic.exch(new_event_mutex,0,0)


@cuda.jit
def event_step_gpu(event_type,event_agent,new_event_mutex,new_event_pointer,new_event_type,new_event_agent,new_event_delay,
                   agent_age,agent_activity_position,agent_activity_distance,agent_position,agent_activity_total,agent_activity_index,agent_activity_type,agent_family,agent_work,agent_activity_duration,agent_infected_status,agent_next_infected_status,
                   trasition_graph_degree,trasition_graph_prob,trasition_graph_node,trasition_graph_delay,
                   building_agents_cnt,building_mutex,building_size,building_location,building_agents,building_free_list,building_free_list_front,building_free_list_rear,grid_buildings,grid_building_total,
                   grid_size,rng_states,N,
                   current_status_trasition_cnt,status_mutex):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return

    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            new_type = -1
            new_delay = -1
            new_agent = -1
            agent=event_agent[now_id]
            type=event_type[now_id]
            # if agent_infected_status==3:   #跳过死亡的agent
            #     continue
            #处理单个事件
            if type == 0:                 # 0代表transport_finish(activity_start)
                # #跳过隔离中的agent
                # if agent_infected_status[agent]==1:
                #     continue
                agent_activity_index[agent] += 1
                agent_activity_index[agent] %= agent_activity_total[agent]
                # 先查询新的活动地点（building_id）
                building_id = agent_activity_position[agent][agent_activity_index[agent]]
                # 将agent置入新的building
                while cuda.atomic.compare_and_swap(building_mutex[building_id], 0, 1) == 1:
                    continue
                npos = building_free_list[building_id][building_free_list_front[building_id]]
                building_free_list_front[building_id] += 1
                building_free_list_front[building_id] %= building_agents.shape[1]
                building_agents[building_id][npos] = agent
                building_agents_cnt[building_id] += 1
                cuda.atomic.exch(building_mutex[building_id], 0, 0)
                agent_position[agent] = building_id

                #注册一个活动结束事件
                next_index = (agent_activity_index[agent] + 1) % agent_activity_duration.shape[1]
                #next_activity_type = agent_activity_type[agent][next_index]
                new_type=1
                new_agent = agent
                new_delay = agent_activity_duration[agent][agent_activity_index[agent]]
            elif type == 1:               # 1代表activity结束
                now_building_id=agent_position[agent]
                # # 跳过隔离中的agent
                # if agent_infected_status[agent] == 1:
                #     continue
                next_activity_index=(agent_activity_index[agent]+1) %agent_activity_total[agent]
                next_building_id = -1
                # 选择下一个目的地
                # 分为H/W/other三种情况
                activity_type = agent_activity_type[agent][next_activity_index]
                if activity_type == 0:  # H
                    next_building_id = agent_family[agent]
                elif activity_type == 1:  # W
                    next_building_id = agent_work[agent]
                else:  # other
                    # 选择下一个目的地
                    # 查询两次activity间隔距离
                    dis = agent_activity_distance[agent][agent_activity_index[agent]]
                    # 确定具体的活动场所（building_id）
                    now_x = building_location[now_building_id][0]
                    now_y = building_location[now_building_id][1]
                    nx=-1
                    ny=-1
                    while not (nx >= 0 and nx < grid_buildings.shape[0] and ny >= 0 and ny < grid_buildings.shape[1]):
                        theta = xoroshiro128p_uniform_float32(rng_states, now_id) * 360
                        theta = theta * math.pi / 180
                        dx = int(dis*math.cos(theta)/grid_size)
                        dy = int(dis*math.sin(theta)/grid_size)
                        nx = now_x + dx
                        ny = now_y + dy
                    while next_building_id == -1:
                        id = int(xoroshiro128p_uniform_float32(rng_states, now_id) * (grid_building_total[nx][ny] - 1))
                        next_building_id=grid_buildings[nx][ny][id]
                        if building_agents_cnt[id] < building_size[id]:
                            next_building_id = id
                agent_activity_position[agent][next_activity_index]=next_building_id
                #估算旅行时长

                #将agent从当前的活动地点移除
                # 查询agent目前所在地点（building_id）
                now_building_id = agent_position[agent]
                pos = int(-1)
                k = 0
                while k < building_agents.shape[1]:
                    if building_agents[now_building_id][k] == agent:
                        pos = k
                        break
                    k += 1
                if pos != -1:
                    building_agents[now_building_id][pos] = -1
                    building_agents_cnt[now_building_id]-=1
                    while cuda.atomic.compare_and_swap(building_mutex[now_building_id], 0, 1) == 1:
                        continue
                    building_free_list[now_building_id][building_free_list_rear[now_building_id]] = pos
                    building_free_list_rear[now_building_id] += 1
                    building_free_list_rear[now_building_id] %= building_agents.shape[1]
                    cuda.atomic.exch(building_mutex[now_building_id], 0, 0)
                agent_position[agent] = -1
                #注册一个到达事件（待修改）
                new_type = 0
                new_agent = agent
                new_delay = 2  #（待修改!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!）
            elif type==2:                # 2代表infected_status_trasition
                state0=agent_infected_status[agent]
                agent_infected_status[agent]=agent_next_infected_status[agent]
                now_status = agent_infected_status[agent]
                state1=now_status
                while cuda.atomic.compare_and_swap(status_mutex[state0][state1], 0, 1) == 1:
                    continue
                current_status_trasition_cnt[state0][state1]+=1
                cuda.atomic.exch(status_mutex[state0][state1], 0, 0)
                # #如果感染状态为1.则入院
                # if now_status==1:
                #     # 先选择目标医院/隔离点(hospital_id)
                #     target_hospital_id = -1
                #     for k in range(0, building_near_hospital.shape[0]):
                #         id = building_near_hospital[k]
                #         if hospital_agents_cnt[id] < hospital_capacity[id]:
                #             target_hospital_id = id
                #             break
                #     #如果附近医院没有容量，移送集中隔离点
                #
                #     #将agent置入医院/隔离点
                #     while cuda.atomic.compare_and_swap(hospital_mutex[target_hospital_id], 0, 1) == 1:
                #         continue
                #     npos = hospital_free_list[target_hospital_id][hospital_free_list_front[target_hospital_id]]
                #     hospital_free_list_front[target_hospital_id] += 1
                #     hospital_free_list_front[target_hospital_id] %= hospital_agents.shape[2]
                #     hospital_agents[target_hospital_id][npos] = agent
                #     hospital_agents_cnt[target_hospital_id] += 1
                #     cuda.atomic.exch(hospital_mutex[target_hospital_id], 0, 0)
                #     #再将其从所在building移除
                #     activity_type=agent_activity_type[agent][agent_activity_index[agent]]
                #     now_building_id = agent_position[agent]
                #     pos = int(-1)
                #     k = 0
                #     while k < building_agents.shape[1]:
                #         if building_agents[now_building_id][k] == agent:
                #             pos = k
                #             break
                #         k += 1
                #     if pos != -1:
                #         building_agents[now_building_id][pos] = -1
                #         while cuda.atomic.compare_and_swap(building_mutex[now_building_id], 0, 1) == 1:
                #             continue
                #         building_free_list[now_building_id][building_free_list_rear[now_building_id]] = pos
                #         building_free_list_rear[now_building_id] += 1
                #         building_free_list_rear[now_building_id] %= building_agents.shape[1]
                #         cuda.atomic.exch(building_mutex[now_building_id], 0, 0)
                #如果感染状态为2，则出院
                # elif now_status==2:
                #     #后期再考虑恢复该agent的日常活动
                #
                #     #将agent从hospital移除
                #     agent_hospital
                #根据转移概率确定下一个感染状态
                p=xoroshiro128p_uniform_float32(rng_states, now_id)
                new_status=-1
                p_sum=0
                age_type=agent_age[agent]
                for k in range(0,trasition_graph_degree[now_status]):
                    if p>= p_sum and p<=p_sum+trasition_graph_prob[now_status][k][age_type]:
                        new_status=trasition_graph_node[now_status][k]
                        new_delay=trasition_graph_delay[now_status][k][age_type]
                        break
                    p_sum+=trasition_graph_prob[now_status][k][age_type]
                agent_next_infected_status[agent]=new_status
                new_agent=agent
                new_type=2
                if trasition_graph_degree[now_status]==0:
                    new_type=-1
            # 若插入新事件，互斥访问new_event
            if new_type == -1:
                continue
            while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                continue
            new_event_type[new_event_pointer[0]] = new_type
            new_event_agent[new_event_pointer[0]] = new_agent
            new_event_delay[new_event_pointer[0]] = new_delay
            new_event_pointer[0] += 1
            cuda.atomic.exch(new_event_mutex, 0, 0)


@cuda.jit
def update_infected_status_gpu(agent_infected_status,
                               building_agents,building_type,building_free_list_front,building_free_list_rear,
                               new_event_mutex,new_event_pointer,new_event_type,new_event_agent,new_event_delay,
                               node_infectivity,node_quarantine,rng_states,next_infected_status,land_contacts,infected_p,temp,N,building_cnt,building_contact):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            #该建筑体内当前有多少agent
            front=building_free_list_front[now_id]
            rear=building_free_list_rear[now_id]
            cnt=0
            for k in range(0,building_agents.shape[1]):
                if building_agents[now_id][k]!=-1:
                    temp[now_id][cnt]=building_agents[now_id][k]
                    cnt+=1
            #该建筑体属于那种用地类型，根据类型确定接触次数
            land_type=building_type[now_id]
            ave_contact=land_contacts[land_type]
            #随机从该建筑体内选择若干对agent进行接触
            now=0

            building_cnt[now_id]=cnt
            building_contact[now_id]=math.ceil(ave_contact*(1.0*cnt)/2)

            while now<math.ceil(ave_contact*(1.0*cnt)/2):
                a=xoroshiro128p_uniform_float32(rng_states, now_id)*(cnt-1)
                b=xoroshiro128p_uniform_float32(rng_states, now_id)*(cnt-1)
                a=int(a)
                b=int(b)
                agent_a=temp[now_id][a]
                agent_b=temp[now_id][b]
                now += 1
                infectivity_a=node_infectivity[agent_infected_status[agent_a]]
                infectivity_b = node_infectivity[agent_infected_status[agent_b]]
                quarantine_a=node_quarantine[agent_infected_status[agent_a]]
                quarantine_b = node_quarantine[agent_infected_status[agent_b]]
                if (infectivity_a==1 and quarantine_a==0) or (infectivity_b==1 and quarantine_b==0):  #接触检测
                    p = xoroshiro128p_uniform_float32(rng_states, now_id)
                    if infected_p>p:
                        infected_agent=-1
                        if agent_infected_status[agent_a]==0:
                            infected_agent=agent_a
                            next_infected_status[agent_a] = 1  # 1代表刚被感染
                        elif agent_infected_status[agent_b]==0:
                            infected_agent=agent_b
                            next_infected_status[agent_b] = 1  # 1代表刚被感染
                        while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                            continue
                        new_event_type[new_event_pointer[0]] = 2
                        new_event_delay[new_event_pointer[0]] = 2
                        new_event_agent[new_event_pointer[0]] = infected_agent
                        new_event_pointer[0] += 1
                        cuda.atomic.exch(new_event_mutex, 0, 0)



@cuda.jit
def statis_building_infected_cnt_gpu(agent_position,agent_infected_status,building_infected_status_cnt,temp_mutex,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            # if agent_infected_status[now_id]!=1:
            #     continue
            building_id = agent_position[now_id]
            status=agent_infected_status[now_id]
            while cuda.atomic.compare_and_swap(temp_mutex[building_id][status], 0, 1) == 1:
                continue
            building_infected_status_cnt[building_id][status] += 1
            cuda.atomic.exch(temp_mutex[building_id][status], 0, 0)

@cuda.jit
def init_agent_position_gpu(agent_position,agent_family,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    agent_position[idx]=agent_family[idx]

@cuda.jit
def assign_agent_to_family_gpu(agent_family,sum,SUM,grid_buildings,grid_buildings_total,rng_states,N,ans_pos):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    p=xoroshiro128p_uniform_float32(rng_states, idx)*SUM[0]
    pos=0
    while pos<sum.shape[0] and sum[pos]<p:
        pos+=1
    ans_pos[idx]=SUM[0]
    x=int(pos/grid_buildings.shape[1])
    y=pos%grid_buildings.shape[1]
    id = int(xoroshiro128p_uniform_float32(rng_states, idx) * (grid_buildings_total[x][y] - 1))
    agent_family[idx]=grid_buildings[x][y][id]

@cuda.jit
def assign_agent_to_work_gpu(agent_family,agent_commuting_distance,agent_work,building_location,grid_buildings,grid_buildings_total,grid_size,rng_states,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    if agent_commuting_distance[idx]==-1:
        agent_work[idx]=-1
        return
    dis = agent_commuting_distance[idx]
    now_building_id=agent_family[idx]
    # 确定具体的活动场所（building_id）
    now_x = building_location[now_building_id][0]
    now_y = building_location[now_building_id][1]
    nx = -1
    ny = -1
    while not (nx >= 0 and nx < grid_buildings.shape[0] and ny >= 0 and ny < grid_buildings.shape[1]):
        theta = xoroshiro128p_uniform_float32(rng_states, idx) * 360
        theta = theta * math.pi / 180
        dx = int(dis * math.cos(theta) / grid_size)
        dy = int(dis * math.sin(theta) / grid_size)
        nx = now_x + dx
        ny = now_y + dy
    id = int(xoroshiro128p_uniform_float32(rng_states, idx) * (grid_buildings_total[nx][ny] - 1))
    agent_work[idx]=grid_buildings[nx][ny][id]

class EpidemicSceneGPU:
    def __init__(self, event_handler, agents, geoinfo,infected_model, interval=1,duration=168):
        #仿真参数
        self.current_step = 0
        self.event_handler=event_handler
        self.interval=interval
        self.total_step=None
        self.on_simulation=False

        #来自Agents
        #activity type统一编号如下：
        #0:H 1:W 2:L 3:S
        self.n_agents=agents.n_agents
        self.agent_activity_type=agents.agent_activity_type
        self.agent_activity_duration=agents.agent_activity_duration
        self.agent_activity_distance=agents.agent_activity_distance
        self.agent_activity_total=agents.agent_activity_total
        self.agent_activity_position=agents.agent_activity_position
        self.agent_family=agents.agent_family
        self.agent_commuting_distance=agents.agent_commuting_distance
        self.agent_age = agents.agent_age
        self.agent_work=None
        self.agent_activity_index=np.zeros(self.n_agents).astype(int)
        self.agent_current_activity_index=np.zeros(self.n_agents).astype(int)
        self.agent_infected_status=np.zeros(self.n_agents).astype(int)
        self.agent_next_infected_status=np.zeros(self.n_agents).astype(int)
        self.agent_position=np.zeros(self.n_agents).astype(int)


        #感染状态转移图相关,来自infected_model
        #规定感染状态的编号，0-未感染(Susceptible) 1-住院 2-恢复(Recovered/出院) 3-死亡
        self.infected_p=infected_model.p
        self.node_name=infected_model.node_name
        self.n_nodes=infected_model.n_nodes
        self.edges = infected_model.edges
        self.edge2num=infected_model.edge2num
        self.trasition_graph_node=infected_model.trasition_graph_node   #结点
        self.trasition_graph_prob=infected_model.trasition_graph_prob   #转移概率
        self.trasition_graph_delay=infected_model.trasition_graph_delay   #转移时延
        self.trasition_graph_degree=infected_model.trasition_graph_degree
        self.node_infectivity=infected_model.node_infectivity
        self.node_quarantine=infected_model.node_quarantine
        self.type_contacts=infected_model.type_contacts
        self.current_status_trasition_cnt = np.zeros([self.n_nodes, self.n_nodes]).astype(int)

        #城市空间信息相关，来自geoinfo
        self.dlon=geoinfo.dlon
        self.dlat=geoinfo.dlat
        self.min_lon=geoinfo.min_lon
        self.min_lat=geoinfo.min_lat
        self.n_buildings=geoinfo.n_buildings
        self.building_location=geoinfo.building_location
        self.building_location_lonlat=geoinfo.building_location_lonlat
        self.grid_size=geoinfo.grid_size
        self.grid_buildings=geoinfo.grid_buildings
        self.grid_buildings_total=geoinfo.grid_buildings_total
        self.building_size=geoinfo.building_size
        self.building_type=geoinfo.building_type

        self.building_agents=np.zeros([self.n_buildings,300]).astype(int)
        self.building_agents_cnt=np.zeros(self.n_buildings).astype(int)
        self.building_agents_free_list=np.zeros([self.n_buildings,300]).astype(int)     #每个building的agent列表的空闲位置索引，以循环队列的形式实现
        self.building_agents_free_list_front=np.zeros(self.n_buildings).astype(int)          #front和rear指针
        self.building_agents_free_list_rear = np.zeros(self.n_buildings).astype(int)
        self.building_agents_cnt=np.zeros(self.n_buildings).astype(int)   #当前每个building内部的agent数量
        self.building_infected_status_cnt=np.zeros([self.n_buildings,self.n_nodes]).astype(int)
        self.building_near_hospital=None #存储每个建筑就近的医院

        #event相关
        self.new_event_mutex = np.zeros(1).astype(int)
        self.new_event_pointer = np.zeros(1).astype(int)
        self.new_event_type = np.zeros(10000000).astype(int)
        self.new_event_delay = np.zeros(10000000)
        self.new_event_agent = np.zeros(10000000).astype(int)


        #医疗资源相关
        self.hospital_total=None     #总数
        self.hospital_capacity=None  #容量
        self.hospital_agents=None  #当前所收容的患者(编号)
        self.hospital_agents_cnt=None #当前医院收容的患者数
        self.hospital_agents_free_list =None
        # front和rear指针
        self.hospital_agents_free_list_front = None
        self.hospital_agents_free_list_rear = None
        self.grid_hospitals =None

        #感染状态宏观统计
        self.step_infected_status_cnt=None
        self.step_status_trasition_cnt=None

        #初始化各数组
        self.init_arrays_for_gpu()
        self.assign_agent_to_work(self)
        self.init_agent_position(self)
        #self.init_agent_infected_status()
        self.init_infected_trasition_events(self)
        self.init_activity_trasition_events(self)
        self.event_handler.insert_new_events(self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay)
        self.new_event_pointer[0]= 0
        #self.transmit_arrays_to_gpu()
        print('scene init done')

        #统计时间消耗
        self.time_step=[]
        self.gpu_step=[]
        self.mem_step=[]

        print(self.node_infectivity)
    # 初始化各类数组
    def init_arrays_for_gpu(self):
        #subarea_agents初始化为-1
        temp=np.ones(self.building_agents.shape)
        self.building_agents=(self.building_agents-temp).astype(int)
        self.init_building_free_list(self)
        self.agent_work=np.zeros(self.n_agents).astype(int)

    #为每个agent注册work地点(某个building内)
    @staticmethod
    def assign_agent_to_work(self):
        #根据人口密度为agent注册family
        total=0
        sum=np.zeros(self.grid_buildings_total.shape[0]*self.grid_buildings_total.shape[1])
        sum[0]=self.grid_buildings_total[0][0]
        for i in range(1,sum.shape[0]):
            x=int(i/self.grid_buildings_total.shape[1])
            y=i%self.grid_buildings_total.shape[1]
            sum[i]=sum[i-1]+self.grid_buildings_total[x][y]
        SUM = np.zeros(1)
        SUM[0]=self.n_buildings

        ans_pos=np.zeros(self.n_agents)

        rng_states = create_xoroshiro128p_states(1024 * math.ceil(self.n_agents / 1024),seed=time.time())
        assign_agent_to_family_gpu[math.ceil(self.n_agents/1024),1024](self.agent_family,sum,SUM,self.grid_buildings,self.grid_buildings_total,rng_states,self.n_agents,ans_pos)
        assign_agent_to_work_gpu[math.ceil(self.n_agents/1024),1024](self.agent_family,self.agent_commuting_distance,self.agent_work,self.building_location,self.grid_buildings,self.grid_buildings_total,self.grid_size,rng_states,self.n_agents)

        # for i in range(0,self.n_agents):
        #     id=self.agent_family[i]
        #     print(id)
    #将频繁读写的array预先装入gpu
    def transmit_arrays_to_gpu(self):
        self.agent_activity_duration=cuda.to_device(self.agent_activity_duration)
        self.new_event_agent=cuda.to_device(self.new_event_agent)
        self.new_event_type = cuda.to_device(self.new_event_type)
        self.new_event_delay = cuda.to_device(self.new_event_delay)
        self.new_event_mutex = cuda.to_device(self.new_event_mutex)
        self.new_event_pointer = cuda.to_device(self.new_event_pointer)
        self.subarea_agents=cuda.to_device(self.subarea_agents)
        self.agent_infected_status=cuda.to_device(self.agent_infected_status)

    @staticmethod
    def init_building_free_list(self):
        N = self.n_buildings
        init_building_free_list_gpu[math.ceil(N/1024),1024](self.building_agents_free_list,self.building_agents_free_list_front,self.building_agents_free_list_rear,N)

    @staticmethod
    def init_agent_position(self):
        N=self.n_agents
        init_agent_position_gpu[math.ceil(N/1024),1024](self.agent_position,self.agent_family,N)

    def init_agent_infected_status(self,status_cnt):
        agent=0
        for i in range(0,len(status_cnt)):
            cnt=0
            while cnt<int(status_cnt[i]):
                self.agent_infected_status[agent]=i
                agent+=1
                cnt+=1

    #为初始感染者注册状态转移事件
    @staticmethod
    def init_infected_trasition_events(self):
        init_infected_trasition_events_gpu[math.ceil(self.n_agents/1024),1024](self.agent_infected_status,self.agent_next_infected_status,self.new_event_mutex,self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay,self.n_agents)

    #注册初始事件
    @staticmethod
    def init_activity_trasition_events(self):
        init_activity_trasition_events_gpu[math.ceil(self.n_agents/1024),1024](self.agent_activity_duration,self.new_event_mutex,self.new_event_pointer, self.new_event_type,
                                                             self.new_event_delay, self.new_event_agent,self.n_agents)
        cuda.synchronize()

    #一步迭代
    def update(self):
        self.event_handler.to_numpy()
        t0=time.time()
        # 触发一般事件，生成新的一般事件
        self.event_step(self)
        self.update_infected_status(self)
        t1=time.time()
        self.time_step.append(t1-t0)
        #self.mem_step.append(float(psutil.virtual_memory().used))
        self.statis_infected_status()
        #将新生成的一般事件统一注册到event_handler
        self.event_handler.insert_new_events(self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay)
        self.new_event_pointer[0] = 0
        self.current_time=self.current_step*self.interval
        if self.current_time>0: #and self.current_time%2==0:    #每24小时统计感染的空间分布情况
          self.statis_building_infected_cnt(self)
        print('now infected cnt:',self.agent_infected_status[self.agent_infected_status==1].shape[0])

    @staticmethod
    def event_step(self):
        if self.event_handler.now_event_type.shape[0]==0 or self.event_handler.now_event_type[0]==-1:
            return
        self.current_status_trasition_cnt = np.zeros([self.n_nodes, self.n_nodes]).astype(int)
        building_mutex=np.zeros([self.n_buildings,1]).astype(int)
        rng_states = create_xoroshiro128p_states(1024 * math.ceil(self.event_handler.now_event_type.shape[0] / 1024), seed=time.time())
        status_mutex=np.zeros([self.n_nodes,self.n_nodes,1]).astype(int)
        event_step_gpu[math.ceil(self.event_handler.now_event_type.shape[0]/ 1024), 1024](self.event_handler.now_event_type,self.event_handler.now_event_agent,self.new_event_mutex,self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay,
                                                                                          self.agent_age,self.agent_activity_position,self.agent_activity_distance,self.agent_position,self.agent_activity_total,self.agent_activity_index,self.agent_activity_type,self.agent_family,self.agent_work,self.agent_activity_duration,self.agent_infected_status,self.agent_next_infected_status,
                                                                                          self.trasition_graph_degree,self.trasition_graph_prob,self.trasition_graph_node,self.trasition_graph_delay,
                                                                                          self.building_agents_cnt,building_mutex,self.building_size,self.building_location,self.building_agents,self.building_agents_free_list,self.building_agents_free_list_front,self.building_agents_free_list_rear,self.grid_buildings,self.grid_buildings_total,
                                                                                          self.grid_size,rng_states,self.event_handler.now_event_type.shape[0],
                                                                                          self.current_status_trasition_cnt,status_mutex)
        #self.gpu_step.append(GPUtil.getGPUs()[0].memoryUsed)
        cuda.synchronize()



    @staticmethod
    def update_infected_status(self):

        #临时
        building_cnt=np.zeros(self.n_buildings).astype(int)
        building_contact=np.zeros(self.n_buildings).astype(int)

        N=self.n_buildings
        rng_states = create_xoroshiro128p_states(1024 * math.ceil(N / 1024), seed=time.time())
        temp=np.zeros(self.building_agents.shape).astype(int)
        update_infected_status_gpu[math.ceil(N/1024),1024](self.agent_infected_status,
                                                           self.building_agents,self.building_type,self.building_agents_free_list_front,self.building_agents_free_list_rear,
                                                           self.new_event_mutex,self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay,
                                                           self.node_infectivity,self.node_quarantine,rng_states,self.agent_next_infected_status,self.type_contacts,self.infected_p,temp,N,building_cnt,building_contact)
        cuda.synchronize()
    #各感染状态宏观统计（每个step进行一次）
    def statis_infected_status(self):
        now_step=self.current_step
        for i in range(0,self.step_infected_status_cnt.shape[1]):
            self.step_infected_status_cnt[now_step][i]=(self.agent_infected_status[self.agent_infected_status==i]).shape[0]
        self.step_status_trasition_cnt[now_step]=self.current_status_trasition_cnt

    #前端请求宏观统计数据时调用
    def get_statis(self):
        ans0={}
        ans1={}
        for i in range(0,self.current_step):
            now_data={}
            for k in range(0,len(self.node_name)):
                name=self.node_name[k]
                now_data[name]=int(self.step_infected_status_cnt[i][k])
            ans0[str(i)]=now_data
        for i in range(0,self.current_step):
            now_data = {}
            for k in range(0, len(self.edges)):
                node0=self.edges[k][0]
                node1 = self.edges[k][1]
                name0=self.node_name[node0]
                name1 = self.node_name[node1]
                now_data[name0+'->'+name1] = int(self.step_status_trasition_cnt[i][node0][node1])
            ans1[str(i)] = now_data
        ans = {}
        ans['status']=ans0
        ans['trasition'] = ans1
        return ans


    #感染状态的空间分布统计(每24h进行一次)
    @staticmethod
    def statis_building_infected_cnt(self):
        N=self.n_agents
        self.building_infected_status_cnt=np.zeros([self.n_buildings,self.n_nodes]).astype(int)
        temp_mutex=np.zeros([self.n_buildings,self.n_nodes,1]).astype(int)
        statis_building_infected_cnt_gpu[math.ceil(N/1024),1024](self.agent_position,self.agent_infected_status,self.building_infected_status_cnt,temp_mutex,N)
        cuda.synchronize()
        self.step_building_infected_status_cnt[self.current_step]=self.building_infected_status_cnt
    #前端请求heat信息时调用
    @staticmethod
    def get_heats(self):
        heats={}
        for i in range(0,self.n_nodes):
            now_heat={}
            for k in range(0,self.building_infected_status_cnt.shape[0]):
                if self.building_infected_status_cnt[k][i]==0:
                    continue
                now_heat[str(k)]={'lon':self.building_location_lonlat[k][1],
                                  'lat':self.building_location_lonlat[k][0],
                                  'cnt':int(self.building_infected_status_cnt[k][i])}
            heats[self.node_name[i]]=now_heat
        return {'heats':heats,'now_step':self.current_step}

    #滑条选中某step的heatmap
    def get_heats_step(self,step):
        heats = {}
        for i in range(0, self.n_nodes):
            now_heat = {}
            for k in range(0, self.step_building_infected_status_cnt.shape[1]):
                if self.step_building_infected_status_cnt[step][k][i] == 0:
                    continue
                now_heat[str(k)] = {'lon': self.building_location_lonlat[k][1],
                                    'lat': self.building_location_lonlat[k][0],
                                    'cnt': int(self.step_building_infected_status_cnt[step][k][i])}
            heats[self.node_name[i]] = now_heat
        return heats

    def get_trips(self):
        ans={}   #数据格式：{'00':[tripdata],'01':[tripdata]....} '00'代表网格index，latlon
        for x in range(0,self.grid_buildings.shape[0]*self.grid_buildings.shape[1]):
            ans[str(x)]=[]
        for i in range(0,self.n_agents):
            home_id=self.agent_family[i]
            work_id=self.agent_work[i]
            grid_x=self.building_location[home_id][0]
            grid_y = self.building_location[home_id][1]
            home_lon=self.building_location_lonlat[home_id][1]
            home_lat = self.building_location_lonlat[home_id][0]
            work_lon = self.building_location_lonlat[work_id][1]
            work_lat = self.building_location_lonlat[work_id][0]
            index=grid_x*self.grid_buildings.shape[1]+grid_y
            ans[str(index)].append({'path':[[self.min_lon+grid_y*self.dlon,self.min_lat+grid_x*self.dlat],[work_lon,work_lat]],'timestamps':[0,200]})
            #ans[str(index)].append({'path': [[home_lon, home_lat], [0, 0]], 'timestamps': [0, 200]})
        return ans

    def step(self):
        self.current_step+=1
        self.event_handler.step()

    def monitor(self):
        while self.on_simulation==True:
            #time.sleep(1)
            self.gpu_step.append(GPUtil.getGPUs()[0].memoryUsed)
            self.mem_step.append(float(psutil.virtual_memory().used))

    def simulation_thread(self):
        while self.current_step < self.total_step:
            if self.on_simulation == False:
                continue
            self.update()
            self.step()
            print('step', self.current_step, 'done')
        self.on_simulation=False
        self.time_step=np.array(self.time_step)
        self.mem_step = np.array(self.mem_step)
        self.gpu_step = np.array(self.gpu_step)
        self.I_step=self.step_infected_status_cnt[0:self.total_step,1]
        np.save('E:/ABMEpidemic/results/withPull/time_'+str(self.total_step)+'_'+str(self.n_agents)+'.npy',self.time_step)
        np.save('E:/ABMEpidemic/results/withPull/mem_'+str(self.total_step)+'_'+str(self.n_agents)+'.npy',self.mem_step)
        np.save('E:/ABMEpidemic/results/withPull/gpu_'+str(self.total_step)+'_'+str(self.n_agents)+'.npy',self.gpu_step)
        np.save('E:/ABMEpidemic/results/withPull/I_' + str(self.total_step) + '_' + str(self.n_agents) + '.npy',self.I_step)

    def simulation(self,total_step):
        self.on_simulation = True
        self.total_step=total_step
        self.step_infected_status_cnt=np.zeros([self.total_step,self.n_nodes]).astype(int)
        self.step_status_trasition_cnt=np.zeros([self.total_step,self.n_nodes,self.n_nodes]).astype(int)
        self.step_building_infected_status_cnt = np.zeros([self.total_step, self.n_buildings, self.n_nodes]).astype(int)
        #开启线程监测内存和显存情况
        thread0=threading.Thread(target=self.simulation_thread)
        thread1 = threading.Thread(target=self.monitor)
        thread1.start()
        thread0.start()
        # while self.current_step < self.total_step:
        #     if self.on_simulation == False:
        #         continue
        #     self.update()
        #     self.step()
        #     print('step', self.current_step, 'done')
        # self.on_simulation=False
        # self.time_step=np.array(self.time_step)
        # self.mem_step = np.array(self.mem_step)
        # self.gpu_step = np.array(self.gpu_step)
        # np.save('E:/ABMEpidemic/results/original/time_'+str(self.total_step)+'_'+str(self.n_agents)+'.npy',self.time_step)
        # np.save('E:/ABMEpidemic/results/original/mem_'+str(self.total_step)+'_'+str(self.n_agents)+'.npy',self.mem_step)
        # np.save('E:/ABMEpidemic/results/original/gpu_'+str(self.total_step)+'_'+str(self.n_agents)+'.npy',self.gpu_step)


    def pause(self):
        self.on_simulation=False

    def go_on(self):
        self.on_simulation=True

    def get_nodes_info(self):
        return {'n_nodes':self.n_nodes,'node_name':self.node_name,'edges':self.edges}

    def reset_simulation(self):
        self.on_simulation=False
        self.current_step=0

    def reset_infected_model(self,infected_model):
        self.trasition_graph_prob = infected_model.trasition_graph_prob  # 转移概率
        self.trasition_graph_delay = infected_model.trasition_graph_delay  # 转移时延

@cuda.jit
def count_new_events(cnt_delay_mutex,cnt_delay,new_event_type,new_event_agent,new_event_delay,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            while cuda.atomic.compare_and_swap(cnt_delay_mutex[new_event_delay[now_id]], 0, 1) == 1:
                continue
            cnt_delay[new_event_delay[now_id]]+=1
            cuda.atomic.exch(cnt_delay_mutex, new_event_delay[now_id], 0)


class EventHandler:
    def __init__(self):
        self.event_dict_type={}   #每个时刻存放的都是numpy
        self.event_dict_agent={}
        self.event_handlers={}
        self.now_event_type=np.zeros(1).astype(int)
        self.now_event_type[0]=-1
        self.now_event_agent=np.zeros(1).astype(int)
        self.current_step=0

    def step(self):
        self.current_step+=1

    def to_numpy(self):
        t=self.current_step
        self.now_event_type=np.zeros(1).astype(int)
        if self.event_dict_type.__contains__(t)==False:
            self.now_event_type[0]=-1
            return
        self.now_event_type = self.event_dict_type[t].astype(int)
        self.now_event_agent = self.event_dict_agent[t].astype(int)

    def register_event(self,event_type,handler):
        self.event_handlers[event_type]=handler

    #插入新生成的事件们
    def insert_new_events(self,new_event_pointer,new_event_type,new_event_agent,new_event_delay):
        if new_event_pointer[0]==0:
            return
        # new_event_delay=new_event_delay_gpu.copy_to_host()
        # new_event_type=new_event_type_gpu.copy_to_host()
        # new_event_agent=new_event_agent_gpu.copy_to_host()

        new_event_delay_temp=new_event_delay[0:new_event_pointer[0]]
        new_event_type_temp=new_event_type[0:new_event_pointer[0]]
        new_event_agent_temp=new_event_agent[0:new_event_pointer[0]]

        new_event_delay_temp = new_event_delay_temp.astype(int)
        order_index=np.argsort(new_event_delay_temp)
        ordered_new_event_delay=new_event_delay_temp[order_index]
        ordered_new_event_delay=np.concatenate((ordered_new_event_delay,np.array([-1]))) #技巧，无意义
        ordered_new_event_type=new_event_type_temp[order_index]
        ordered_new_event_agent=new_event_agent_temp[order_index]
        current_type=[]
        current_agent=[]
        current_type.append(ordered_new_event_type[0])
        current_agent.append(ordered_new_event_agent[0])
        for i in range(1,new_event_pointer[0]+1):
            if ordered_new_event_delay[i]==ordered_new_event_delay[i-1]:
                current_type.append(ordered_new_event_type[i])
                current_agent.append(ordered_new_event_agent[i])
            else:
                last_delay=ordered_new_event_delay[i-1]
                current_type=np.asarray(current_type)
                current_agent=np.asarray(current_agent)
                target_t=self.current_step+last_delay+1
                if self.event_dict_type.__contains__(target_t) == False:
                    self.event_dict_type[target_t] =current_type
                    self.event_dict_agent[target_t]=current_agent
                else:
                    self.event_dict_type[target_t]=np.concatenate((self.event_dict_type[target_t],current_type))
                    self.event_dict_agent[target_t] = np.concatenate((self.event_dict_agent[target_t], current_agent))
                current_type=[]
                current_agent=[]
                if i<new_event_pointer[0]:
                  current_type.append(ordered_new_event_type[i])
                  current_agent.append(ordered_new_event_agent[i])

class GeoInfo:
    def __init__(self):
        #来自前端的url
        self.url_buildings=None
        self.url_landuse=None
        self.url_population=None
        #GIS数据(Geopandas.DataFrame格式)
        self.data_buildings=None
        self.data_landuse=None
        self.data_population=None
        #bounding_box（经纬度范围）
        self.min_lat=None
        self.max_lat=None
        self.min_lon=None
        self.max_lon=None
        self.dlat=None
        self.dlon=None
        #基本参数
        self.grid_size=None
        self.n_buildings=None
        # 以数组存储信息
        self.building_location = None  # 每个建筑的位置(以grid为单位)
        self.building_location_lonlat=None #每个建筑的位置（以经纬度为单位）
        self.grid_buildings = None # 记录每个网格内的建筑列表（建筑编号）
        self.grid_buildings_total = None  # 每个网格内的建筑总数
        self.building_size = None  # 每个建筑的规模
        self.building_type=None    #每个建筑的类型（根据用地类型确定）

    def set_url_buildings(self,url_buildings):
        self.url_buildings = url_buildings

    def set_url_landuse(self,url_landuse):
        self.url_landuse = url_landuse

    def set_url_population(self,urL_population):
        self.url_population = self.url_population

    def fetch_data(self):  #调用WFS服务从GeoServer依次获取shp文件
        wfs = WebFeatureService(url=self.url_buildings, version='1.1.0')
        layer = list(wfs.contents)[0]
        params = dict(service='WFS', version="1.0.0", request='GetFeature',
                      typeName=layer, outputFormat='json')
        q = Request('GET', self.url_buildings, params=params).prepare().url
        self.data_buildings = gpd.read_file(q)

        wfs = WebFeatureService(url=self.url_landuse, version='1.1.0')
        layer = list(wfs.contents)[0]
        params = dict(service='WFS', version="1.0.0", request='GetFeature',
                      typeName=layer, outputFormat='json')
        # Parse the URL with parameters
        q = Request('GET', self.url_buildings, params=params).prepare().url
        self.data_landuse = gpd.read_file(q)

        wfs = WebFeatureService(url=self.url_population, version='1.1.0')
        layer = list(wfs.contents)[0]
        params = dict(service='WFS', version="1.0.0", request='GetFeature',
                      typeName=layer, outputFormat='json')
        # Parse the URL with parameters
        q = Request('GET', self.url_buildings, params=params).prepare().url
        self.data_population = gpd.read_file(q)

    def data_to_numpy(self):   #根据GIS数据生成必要信息，并将数据组织为numpy（目前无真实数据输入，随意生成）
        self.grid_size=4500
        self.n_buildings=4000

        #不需要改动
        dlat = self.grid_size / (1000 * 111)
        dlon=dlat
        self.dlat=dlat
        self.dlon=dlon
        nlat = math.ceil((self.max_lat - self.min_lat) / dlat)
        nlon = math.ceil((self.max_lon - self.min_lon) / dlon)
        self.building_location=np.zeros([self.n_buildings,2]).astype(int)
        self.building_location_lonlat = np.zeros([self.n_buildings, 2])
        self.grid_buildings_total=np.zeros([nlat,nlon]).astype(int)
        self.building_size=np.zeros([self.n_buildings]).astype(int)
        self.building_type=np.zeros([self.n_buildings]).astype(int)

        Dlat=self.max_lat-self.min_lat
        Dlon=self.max_lon-self.min_lon
        temp_cnt=np.zeros(self.grid_buildings_total.shape).astype(int)




        start=int(0)
        self.min_lat=31.1012
        self.max_lat=31.3243
        self.min_lon=121.3399
        self.max_lon=121.5823
        for i in range(0,math.floor(0.5*self.n_buildings)):
            lat=self.min_lat+Dlat*random.random()
            lon=self.min_lon+Dlon*random.random()
            index_lat=math.floor((lat-self.min_lat)/dlat)
            index_lon=math.floor((lon-self.min_lon)/dlon)
            self.building_location[i+start][0] = index_lat
            self.building_location[i+start][1] = index_lon
            self.building_location_lonlat[i+start][0]=lat
            self.building_location_lonlat[i+start][1]=lon
            # print('nlat nlon',nlat,nlon)
            temp_cnt[index_lat][index_lat]+=1
            start+=1
        
        self.min_lat=31.0224
        self.max_lat=31.3243
        self.min_lon=121.1243
        self.max_lon=121.3399
        for i in range(0,math.floor(0.15*self.n_buildings)):
            lat=self.min_lat+Dlat*random.random()
            lon=self.min_lon+Dlon*random.random()
            index_lat=math.floor((lat-self.min_lat)/dlat)
            index_lon=math.floor((lon-self.min_lon)/dlon)
            self.building_location[i+start][0] = index_lat
            self.building_location[i+start][1] = index_lon
            self.building_location_lonlat[i+start][0]=lat
            self.building_location_lonlat[i+start][1]=lon
            # print('nlat nlon',nlat,nlon)
            temp_cnt[index_lat][index_lat]+=1
            start+=1
        
        self.min_lat=31.3243
        self.max_lat=31.4474
        self.min_lon=121.1243
        self.max_lon=121.4264
        for i in range(0,math.floor(0.15*self.n_buildings)):
            lat=self.min_lat+Dlat*random.random()
            lon=self.min_lon+Dlon*random.random()
            index_lat=math.floor((lat-self.min_lat)/dlat)
            index_lon=math.floor((lon-self.min_lon)/dlon)
            self.building_location[i+start][0] = index_lat
            self.building_location[i+start][1] = index_lon
            self.building_location_lonlat[i+start][0]=lat
            self.building_location_lonlat[i+start][1]=lon
            # print('nlat nlon',nlat,nlon)
            temp_cnt[index_lat][index_lat]+=1
            start+=1
        
        self.min_lat=30.8929
        self.max_lat=31.1012
        self.min_lon=121.1243
        self.max_lon=121.8768
        for i in range(0,math.floor(0.20*self.n_buildings)):
            lat=self.min_lat+Dlat*random.random()
            lon=self.min_lon+Dlon*random.random()
            index_lat=math.floor((lat-self.min_lat)/dlat)
            index_lon=math.floor((lon-self.min_lon)/dlon)
            self.building_location[i+start][0] = index_lat
            self.building_location[i+start][1] = index_lon
            self.building_location_lonlat[i+start][0]=lat
            self.building_location_lonlat[i+start][1]=lon
            # print('nlat nlon',nlat,nlon)
            temp_cnt[index_lat][index_lat]+=1
            start+=1




        self.grid_buildings = np.zeros([nlat, nlon, np.max(temp_cnt)]).astype(int)
        for i in range(0,self.n_buildings):
            lat = self.building_location_lonlat[i][0]
            lon = self.building_location_lonlat[i][1]
            index_lat = math.floor((lat - self.min_lat) / dlat)
            index_lon = math.floor((lon - self.min_lon) / dlon)
            # print('nlat nlon',nlat,nlon)
            # print('index_lat index_lon', index_lat, index_lon)
            # print('grid_buildings',self.grid_buildings.shape)
            # print('grid_buildings_total', self.grid_buildings_total.shape)
            self.grid_buildings[index_lat][index_lon][self.grid_buildings_total[index_lat][index_lon]]=i
            self.grid_buildings_total[index_lat][index_lon]+=1
            type = random.randint(0, 3)  # 四种type
            self.building_type[i]=type

class InfectedModel:
    def __init__(self):
        self.trasition_form=None
        self.p=None
        self.land_contacts=np.zeros(4).astype(int)
        #基本参数
        self.n_nodes=None
        self.max_degree=None
        #以数组存储信息
        # 规定感染状态的编号，0-未感染(Susceptible) 1-住院 2-恢复(Recovered/出院) 3-死亡
        self.node_name=None   #每个结点（状态）的名称
        self.edges=None       #记录边，数据格式：[[source,target],[]...,[]]
        self.edge2num=None    #边到编号的mapping
        self.node_degree=None
        self.name2num=None    #结点名称到编号的映射
        self.infectivity=None
        self.quarantine=None
        self.node_infectivity=None   #某状态是否具有传染力
        self.node_quarantine=None      #某状态是否被隔离（限制活动）
        self.trasition_graph_node = None  # 结点
        self.trasition_graph_prob = None  # 转移概率
        self.trasition_graph_delay = None  # 转移时延
        self.trasition_graph_degree = None #每个结点的度
        self.type_contacts=None   #每种类型建筑内的接触次数，由land_contacts确定

    def set_model(self,trasition_form):
        # 接收前端数据
        self.trasition_form = trasition_form

    def set_params(self,land_contacts,p,node_infectivity,node_quarantine):
        self.land_contacts=land_contacts
        self.p=float(p)
        self.infectivity =node_infectivity  #字典形式
        self.quarantine = node_quarantine

    def data_to_numpy(self):
        #根据trasition_form构建
        self.name2num={}
        temp_name2num={}
        self.n_nodes=0
        for edge in self.trasition_form:
            node0=edge['state0']
            node1=edge['state1']
            if not temp_name2num.__contains__(node0):
                temp_name2num[node0]=self.n_nodes
                self.n_nodes += 1
            if not temp_name2num.__contains__(node1):
                temp_name2num[node1] = self.n_nodes
                self.n_nodes += 1

        self.node_name=[0]*int(self.n_nodes)
        self.n_nodes=0
        for edge in self.trasition_form:
            node0=edge['state0']
            node1=edge['state1']
            num0=-1
            num1=-1
            if self.name2num.__contains__(node0):
                num0=self.name2num[node0]
            else:
                num0=self.n_nodes
                self.node_name[num0]=node0
                self.name2num[node0]=num0
                self.n_nodes += 1
            if self.name2num.__contains__(node1):
                num1=self.name2num[node1]
            else:
                num1 = self.n_nodes
                self.node_name[num1] = node1
                self.name2num[node1] = num1
                self.n_nodes += 1

        self.trasition_graph_degree = np.zeros(self.n_nodes).astype(int)
        self.node_degree=np.zeros(self.n_nodes).astype(int)
        for edge in self.trasition_form:
            node0=edge['state0']
            num0=self.name2num[node0]
            self.trasition_graph_degree[num0]+=1
        self.max_degree=np.max(self.trasition_graph_degree)

        self.edges=[]
        self.edge2num={}
        cnt=0
        for edge in self.trasition_form:
            node0=edge['state0']
            node1=edge['state1']
            num0 = self.name2num[node0]
            num1 = self.name2num[node1]
            self.edges.append([num0,num1])
            self.edge2num['state'+str(num0)+'state'+str(num1)]=cnt
            cnt+=1

        self.trasition_graph_node = np.zeros([self.n_nodes, self.max_degree]).astype(int)
        self.trasition_graph_prob = np.zeros([self.n_nodes, self.max_degree,3])
        self.trasition_graph_delay = np.zeros([self.n_nodes, self.max_degree,3])
        for edge in self.trasition_form:
            node0=edge['state0']
            node1=edge['state1']
            p0=float(edge['p0'])
            p1=float(edge['p1'])
            p2=float(edge['p2'])
            t0=float( edge['t0'])
            t1=float(edge['t1'])
            t2=float( edge['t2'])

            num0=self.name2num[node0]
            num1=self.name2num[node1]
            degree0=self.node_degree[num0]
            self.trasition_graph_node[num0][degree0]=num1
            self.trasition_graph_prob[num0][degree0][0]=p0
            self.trasition_graph_prob[num0][degree0][1] =p1
            self.trasition_graph_prob[num0][degree0][2] =p2

            self.trasition_graph_delay[num0][degree0][0] = t0
            self.trasition_graph_delay[num0][degree0][1] = t1
            self.trasition_graph_delay[num0][degree0][2] = t2
            self.node_degree[num0]+=1


        #以下待修改前端，应从前端获取
        self.node_infectivity=np.zeros(self.n_nodes).astype(int)
        self.node_quarantine=np.zeros(self.n_nodes).astype(int)
        self.type_contacts=np.zeros([4])
        for i in range(0,self.type_contacts.shape[0]):
            self.type_contacts[i]=float(self.land_contacts[i])
        for i in range(0,self.n_nodes):
            self.node_infectivity[i]=int(self.infectivity[i])
        for i in range(0,self.n_nodes):
            self.node_quarantine[i]=int(self.quarantine[i])


    def modify_infected_model(self):
        cnt=np.zeros(self.n_nodes).astype(int)
        for edge in self.trasition_form:
            node0=edge['state0']
            node1=edge['state1']
            p0=float(edge['p0'])
            p1=float(edge['p1'])
            p2=float(edge['p2'])
            t0=float( edge['t0'])
            t1=float(edge['t1'])
            t2=float( edge['t2'])
            num0=self.name2num[node0]
            num1=self.name2num[node1]
            cnt0=cnt[num0]
            self.trasition_graph_node[num0][cnt0]=num1
            self.trasition_graph_prob[num0][cnt0][0]=p0
            self.trasition_graph_prob[num0][cnt0][1] =p1
            self.trasition_graph_prob[num0][cnt0][2] =p2

            self.trasition_graph_delay[num0][cnt0][0] = t0
            self.trasition_graph_delay[num0][cnt0][1] = t1
            self.trasition_graph_delay[num0][cnt0][2] = t2
            cnt[num0]+=1

#app.routes
app = Flask(__name__)
cors = CORS(app, resources={r"/submit_agent_file": {"origins": "*"},r"/upload_url_buildings": {"origins": "*"},r"/upload_url_landuse": {"origins": "*"},r"/upload_population_file": {"origins": "*"},
                            r"/upload_infected_model": {"origins": "*"},r"/construct_scene": {"origins": "*"},
                            r"/start_simulation": {"origins": "*"},r"/pause_simulation": {"origins": "*"},
                            r"/get_heats": {"origins": "*"},r"/get_statis": {"origins": "*"},r"/get_nodes_info": {"origins": "*"},
                            r"/continue_simulation": {"origins": "*"},
                            r"/upload_infected_params": {"origins": "*"}, r"/modify_infected_model": {"origins": "*"},
                            r"/init_globvars": {"origins": "*"},
                            r"/get_heats_3d": {"origins": "*"},
                            r"/get_trip_data": {"origins": "*"},
                            r"/construct_scene_temp": {"origins": "*"},
                            r"/reset_simulation": {"origins": "*"},
                            r"/get_heats_step": {"origins": "*"}
                           })

agents=None
geoinfo=None
infected_model=None
event_handler=None
scene=None

@app.route('/init_globvars',methods=['POST'])
def init_globvars():
    global agents
    global geoinfo
    global infected_model
    global event_handler
    global scene
    agents=Agents()
    infected_model=InfectedModel()
    geoinfo=GeoInfo()
    event_handler=EventHandler()
    print('globvars init done')
    return 'globvars init done'

@app.route('/submit_agent_file',methods=['POST'])  #接收agent文件并保存，待补充
def upload_agent_file():
    global infected_model
    print(type(infected_model))
    #globvars.agents.set_data()
    return 'ok'

@app.route('/upload_url_buildings',methods=['POST'])  #接收url并保存，待补充
def upload_url_building():
    #globvars.geoinfo.set_url_buildings(url_buildings)
    return 'ok'

@app.route('/upload_url_landuse',methods=['POST'])  #接收url并保存，待补充
def upload_url_landuse():
    #globvars.geoinfo.set_url_landuse(url_landuse)
    return 'ok'

@app.route('/upload_population_file',methods=['POST'])  #接收url并保存，待补充
def upload_url_population():
    #globvars.geoinfo.set_url_population(url_population)
    return 'ok'

@app.route('/upload_infected_model',methods=['POST'])  #接收infected_model，待补充
def upload_infected_model():
    global infected_model
    data = request.get_json()
    trasition_form=data['form']
    infected_model.set_model(trasition_form)
    return 'ok'

@app.route('/upload_infected_params',methods=['POST'])  #接收感染参数
def upload_infected_params():
    global infected_model
    data=request.get_json()
    p=data['p']
    land_contacts=data['land_contacts']
    node_infectivity=data['node_infectivity']
    node_quarantine=data['node_quarantine']
    infected_model.set_params(land_contacts, p,node_infectivity,node_quarantine)
    return 'ok'

@app.route('/construct_scene',methods=['GET'])  #根据已上传数据构建仿真场景
def construct_scene():
    global agents
    global geoinfo
    global infected_model
    global scene
    #globvars.geoinfo.fetch_data()  暂时不管
    #将各类数据组织为numpy
    agents.data_to_numpy()
    geoinfo.data_to_numpy()
    infected_model.data_to_numpy()
    mem = psutil.virtual_memory()
    gpus = GPUtil.getGPUs()
    # 系统已经使用内存(单位字节)
    print('mem.used before start',float(mem.used))  #记录仿真启动前系统内存和显存情况
    print('gpu.used before start',gpus[0].memoryUsed)
    print('construct scene done')
    #构建scene
    scene=EpidemicSceneGPU(event_handler, agents, geoinfo,infected_model, interval=1,duration=168)
    #globvars.scene = EpidemicSceneStream(globvars.event_handler, globvars.agents, globvars.geoinfo,globvars.infected_model, interval=1, duration=168)
    #globvars.scene = EpidemicScenePull(globvars.event_handler, globvars.agents, globvars.geoinfo,globvars.infected_model, interval=1, duration=168)
    return 'ok'

@app.route('/construct_scene_temp',methods=['GET'])  #根据已上传数据构建仿真场景
def construct_scene_temp():
    global agents
    global infected_model
    global geoinfo
    global event_handler
    global scene
    agents = Agents()
    infected_model = InfectedModel()
    geoinfo = GeoInfo()
    event_handler = EventHandler()

    agents.data_to_numpy()
    geoinfo.data_to_numpy()
    infected_model.set_model(
        [{'state0': 'S', 'state1': 'I', 'p0': '1', 'p1': '1', 'p2': '1', 't0': '1', 't1': '1', 't2': '1'},
         {'state0': 'I', 'state1': 'R', 'p0': '1', 'p1': '1', 'p2': '1', 't0': '120', 't1': '120', 't2': '120'}])
    infected_model.set_params(['1', '1', '1', '1'], '1', ['0', '1', '0'], ['0', '0', '0'])
    infected_model.data_to_numpy()
    #构建scene
    scene=EpidemicSceneGPU(event_handler, agents, geoinfo,infected_model, interval=1,duration=168)
    return 'ok'

@app.route("/modify_infected_model",methods=['POST'])
def modify_infected_model():
    global infected_model
    data = request.get_json()
    trasition_form = data['form']
    infected_model.set_model(trasition_form)
    infected_model.modify_infected_model()
    scene.reset_infected_model(infected_model)
    return 'ok'

on_simulation=False

@app.route("/start_simulation",methods=['POST'])
def start_simulation():
    global on_simulation
    global scene
    if on_simulation==True:
        return 'running...'
    on_simulation=True
    data=request.get_json()
    total_step=data['total_step']
    total_step=int(total_step)
    status_cnt=data['status_cnt']
    scene.init_agent_infected_status(status_cnt)
    scene.simulation(total_step)
    return 'ok'

@app.route("/pause_simulation",methods=['POST'])
def pause_simulation():
    global on_simulation
    global scene
    on_simulation=False
    scene.pause()
    return 'ok'

@app.route("/continue_simulation",methods=['POST'])
def continue_simulation():
    global on_simulation
    global scene
    on_simulation=True
    scene.go_on()
    return 'ok'

@app.route("/get_heats",methods=['GET'])
def get_heats():
    global scene
    heats=scene.get_heats(scene)
    #print('heats:',heats['I'],heats['R'])
    return json.dumps(heats)

@app.route("/get_heats_step",methods=['GET'])
def get_heats_step():
    global scene
    step=int(request.values['step'])
    heats=scene.get_heats_step(step)
    #print('heats:',heats['I'],heats['R'])
    return json.dumps(heats)

@app.route("/get_nodes_info",methods=['GET'])
def get_ndoes_info():
    global scene
    return json.dumps(scene.get_nodes_info())

@app.route("/reset_simulation",methods=['POST'])
def reset_simulation():
    global on_simulation
    global scene
    on_simulation=False
    scene.reset_simulation()
    return 'ok'

@app.route("/get_statis",methods=['GET'])
def get_statis():
    global scene
    statis= scene.get_statis()
    return json.dumps(statis)

@app.route("/get_heats_3d",methods=['GET'])
def get_heats_3d():
    global scene
    heats=scene.get_heats(scene)['heats']
    ans0={}
    ans1={}
    for key0 in heats.keys():
        heat=heats[key0]
        now_heat = []
        now_grid= []
        for key1 in heat.keys():
            lon=heat[key1]['lon']
            lat=heat[key1]['lat']
            cnt=heat[key1]['cnt']
            now_grid.append({'CNT': cnt, 'COORDINATES': [lon, lat]})
            for i in range(0,heat[key1]['cnt']):
                now_dict={}
                now_dict['lng']=lon
                now_dict['lat']=lat
                now_heat.append(now_dict)
        ans0[key0]=now_heat
        ans1[key0]=now_grid
        #补齐为零的数据
        for name in scene.node_name:
            if name not in heats.keys():
                ans0[name] = []
                ans1[name] = []
    return json.dumps({'heat_data':ans0,'grid_data':ans1})

@app.route("/get_trip_data",methods=['GET'])
def get_trip_data():
    global scene
    trips=scene.get_trips()
    return json.dumps(trips)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
