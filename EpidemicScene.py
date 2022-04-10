import time
import matplotlib.pyplot as plt
import numpy as np
from Scene import Scene
from numba import cuda,guvectorize
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32,create_xoroshiro128p_states, xoroshiro128p_uniform_float64
import math


@cuda.jit
def init_subarea_free_list_gpu(subarea_agents_free_list,subarea_agents_free_list_front,subarea_agents_free_list_rear,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    len = math.sqrt(N)
    len = int(len)
    x = idx/ len
    y = idx% len
    x = int(x)
    y = int(y)
    for i in range(0,subarea_agents_free_list.shape[2]-1):
        subarea_agents_free_list[x][y][i]=i
    subarea_agents_free_list_front[x][y]=0
    subarea_agents_free_list_rear[x][y]=subarea_agents_free_list.shape[2]-1

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
                new_event_delay[new_event_pointer[0]] =0
                new_event_agent[new_event_pointer[0]] = now_id
                new_event_pointer[0] += 1
                cuda.atomic.exch(new_event_mutex, 0, 0)

@cuda.jit
def init_events_gpu(agent_activity_duration,new_event_mutex,new_event_pointer, new_event_type,new_event_delay, new_event_agent,N):
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
            if now_id==0:
                print('agent0 will trasit to activity1 after',delay)
            while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                continue
            new_event_type[new_event_pointer[0]] =1
            new_event_delay[new_event_pointer[0]] = delay
            new_event_agent[new_event_pointer[0]] = now_id
            new_event_pointer[0]+=1
            cuda.atomic.exch(new_event_mutex,0,0)


@cuda.jit
def event_step_gpu(event_type,event_agent,new_event_mutex,new_event_pointer,new_event_type,new_event_agent,new_event_delay,agent_activity_index,agent_activity_type,agent_activity_center,agent_activity_duration,agent_infected_status,
                                                            rng_states,agent_next_infected_status,trasition_graph_degree,trasition_graph_prob,trasition_graph_node,trasition_graph_delay,building_mutex,building_agents,building_free_list,building_free_list_front,building_free_list_rear,N):
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
            if agent_infected_status==3:   #跳过死亡的agent
                continue
            #处理单个事件
            if type == 0:                 # 0代表transport_finish(activity_start)
                #跳过隔离中的agent
                if agent_infected_status[agent]==1:
                    continue
                agent_activity_index[agent] += 1
                agent_activity_index[agent] %= agent_activity_total[agent]
                # 分三种情况，当前活动是H/W/other
                activity_type=agent_activity_type[agent][agent_activity_index[agent]]
                if activity_type==0: #若当前是H
                    family_id = agent_activity_position[agent][agent_activity_index[agent]]
                    while cuda.atomic.compare_and_swap(family_mutex[family_id], 0, 1) == 1:
                        continue
                    npos = family_free_list[family_id][family_free_list_front[family_id]]
                    family_free_list_front[family_id] += 1
                    family_free_list_front[family_id] %= family_agents.shape[2]
                    family_agents[family_id][npos] = agent
                    cuda.atomic.exch(family_mutex[family_id], 0, 0)
                    agent_position[agent]=family_id
                elif activity_type==1:  #若当前是W
                    work_id = agent_activity_position[agent][agent_activity_index[agent]]
                    while cuda.atomic.compare_and_swap(work_mutex[work_id], 0, 1) == 1:
                        continue
                    npos = work_free_list[work_id][work_free_list_front[work_id]]
                    work_free_list_front[work_id] += 1
                    work_free_list_front[work_id] %= work_agents.shape[2]
                    work_agents[work_id][npos] = agent
                    cuda.atomic.exch(work_mutex[work_id], 0, 0)
                    agent_position[agent] = work_id
                else: #若当前是other
                    # 先查询新的活动地点（building_id）
                    building_id = agent_activity_position[agent][agent_activity_index[agent]]
                    # 将agent置入新的building
                    while cuda.atomic.compare_and_swap(building_mutex[building_id], 0, 1) == 1:
                        continue
                    npos = building_free_list[building_id][building_free_list_front[building_id]]
                    building_free_list_front[building_id] += 1
                    building_free_list_front[building_id] %= building_agents.shape[2]
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
                # 跳过隔离中的agent
                if agent_infected_status[agent] == 1:
                    continue
                next_activity_index=(agent_activity_index[agent]+1) %agent_activity_total[agent]
                next_building_id = -1
                #选择下一个目的地
                #分为H/W/other三种情况
                activity_type = agent_activity_type[agent][next_activity_index]
                if activity_type==0:   #H
                    next_building_id=agent_family[agent]
                elif activity_type==1: #W
                    next_building_id=agent_work[agent]
                else:         #other
                    # 查询两次activity间隔距离
                    dis = agent_activity_distance[agent][agent_activity_index[agent]]
                    # 确定具体的活动场所（building_id）
                    now_x = building_location[now_building_id][0]
                    now_y = building_location[now_building_id][1]
                    while not (nx >= 0 and nx < grid_buildings.shape[0] and ny >= 0 and ny < grid_buildings.shape[1]):
                        theta = xoroshiro128p_uniform_float32(rng_states, now_id) * 360
                        theta = theta * math.pi / 180
                        dx = math.cos(theta)
                        dy = math.sin(theta)
                        nx = int(building_location[now_building_id][0] + dx)
                        ny = int(building_location[now_building_id][1] + dy)
                    while next_building_id == -1:
                        id = int(xoroshiro128p_uniform_float32(rng_states, now_id) * (grid_building_total[building_id] - 1))
                        if build_agents_cnt[id] < building_size[id]:
                            next_building_id = id
                agent_activity_position[agent][next_activity_index]=next_building_id
                #估算旅行时长

                #将agent从当前的活动地点移除
                # 查询agent目前所在地点（building_id）
                now_building_id = agent_position[agent]
                #仍然分三种情况 H/W/other
                if activity_type==0:  #H
                    pos = int(-1)
                    k = 0
                    while k < family_agents.shape[1]:
                        if family_agents[now_building_id][k] == agent:
                            pos = k
                            break
                        k += 1
                    if pos != -1:
                        family_agents[now_building_id][pos] = -1
                        while cuda.atomic.compare_and_swap(family_mutex[now_building_id], 0, 1) == 1:
                            continue
                        family_free_list[now_building_id][family_free_list_rear[now_building_id]] = pos
                        family_free_list_rear[now_building_id] += 1
                        family_free_list_rear[now_building_id] %= family_agents.shape[1]
                        cuda.atomic.exch(family_mutex[now_building_id], 0, 0)
                elif activity_type==1: #W
                    pos = int(-1)
                    k = 0
                    while k < work_agents.shape[1]:
                        if work_agents[now_building_id][k] == agent:
                            pos = k
                            break
                        k += 1
                    if pos != -1:
                        work_agents[now_building_id][pos] = -1
                        while cuda.atomic.compare_and_swap(work_mutex[now_building_id], 0, 1) == 1:
                            continue
                        work_free_list[now_building_id][work_free_list_rear[now_building_id]] = pos
                        work_free_list_rear[now_building_id] += 1
                        work_free_list_rear[now_building_id] %= work_agents.shape[1]
                        cuda.atomic.exch(work_mutex[now_building_id], 0, 0)
                else:                  #other
                    pos = int(-1)
                    k = 0
                    while k < building_agents.shape[1]:
                        if building_agents[now_building_id][k] == agent:
                            pos = k
                            break
                        k += 1
                    if pos != -1:
                        building_agents[now_building_id][pos] = -1
                        while cuda.atomic.compare_and_swap(building_mutex[now_building_id], 0, 1) == 1:
                            continue
                        building_free_list[now_building_id][building_free_list_rear[now_building_id]] = pos
                        building_free_list_rear[now_building_id] += 1
                        building_free_list_rear[now_building_id] %= building_agents.shape[1]
                        cuda.atomic.exch(building_mutex[now_building_id], 0, 0)
                agent_position[agent] = -1
                #注册一个到达事件
                new_type = 0
                new_agent = agent
                new_delay =
            elif type==2:                # 2代表infected_status_trasition
                agent_infected_status[agent]=agent_next_infected_status[agent]
                now_status = agent_infected_status[agent]
                #如果感染状态为1.则入院
                if now_status==1:
                    # 先选择目标医院/隔离点(hospital_id)
                    target_hospital_id = -1
                    for k in range(0, building_near_hospital.shape[0]):
                        id = building_near_hospital[k]
                        if hospital_agents_cnt[id] < hospital_capacity[id]:
                            target_hospital_id = id
                            break
                    #如果附近医院没有容量，移送集中隔离点

                    #将agent置入医院/隔离点
                    while cuda.atomic.compare_and_swap(hospital_mutex[target_hospital_id], 0, 1) == 1:
                        continue
                    npos = hospital_free_list[target_hospital_id][hospital_free_list_front[target_hospital_id]]
                    hospital_free_list_front[target_hospital_id] += 1
                    hospital_free_list_front[target_hospital_id] %= hospital_agents.shape[2]
                    hospital_agents[target_hospital_id][npos] = agent
                    hospital_agents_cnt[target_hospital_id] += 1
                    cuda.atomic.exch(hospital_mutex[target_hospital_id], 0, 0)
                    #再将其从所在building移除
                    #仍分三种情况 H/W/other
                    activity_type=agent_activity_type[agent][agent_activity_index[agent]]
                    now_building_id = agent_position[agent]
                    if activity_type==0:   #H
                        pos = int(-1)
                        k = 0
                        while k < family_agents.shape[1]:
                            if family_agents[now_building_id][k] == agent:
                                pos = k
                                break
                            k += 1
                        if pos != -1:
                            family_agents[now_building_id][pos] = -1
                            while cuda.atomic.compare_and_swap(family_mutex[now_building_id], 0, 1) == 1:
                                continue
                            family_free_list[now_building_id][family_free_list_rear[now_building_id]] = pos
                            family_free_list_rear[now_building_id] += 1
                            family_free_list_rear[now_building_id] %= family_agents.shape[1]
                            cuda.atomic.exch(family_mutex[now_building_id], 0, 0)
                    elif activity_type==1: #W
                        pos = int(-1)
                        k = 0
                        while k < work_agents.shape[1]:
                            if work_agents[now_building_id][k] == agent:
                                pos = k
                                break
                            k += 1
                        if pos != -1:
                            work_agents[now_building_id][pos] = -1
                            while cuda.atomic.compare_and_swap(work_mutex[now_building_id], 0, 1) == 1:
                                continue
                            work_free_list[now_building_id][work_free_list_rear[now_building_id]] = pos
                            work_free_list_rear[now_building_id] += 1
                            work_free_list_rear[now_building_id] %= work_agents.shape[1]
                            cuda.atomic.exch(work_mutex[now_building_id], 0, 0)
                    else:                  #other
                        pos = int(-1)
                        k = 0
                        while k < building_agents.shape[1]:
                            if building_agents[now_building_id][k] == agent:
                                pos = k
                                break
                            k += 1
                        if pos != -1:
                            building_agents[now_building_id][pos] = -1
                            while cuda.atomic.compare_and_swap(building_mutex[now_building_id], 0, 1) == 1:
                                continue
                            building_free_list[now_building_id][building_free_list_rear[now_building_id]] = pos
                            building_free_list_rear[now_building_id] += 1
                            building_free_list_rear[now_building_id] %= building_agents.shape[1]
                            cuda.atomic.exch(building_mutex[now_building_id], 0, 0)

                #如果感染状态为2，则出院
                elif now_status==2:
                    #后期再考虑恢复该agent的日常活动

                    #将agent从hospital移除
                    agent_hospital
                #根据转移概率确定下一个感染状态
                p=xoroshiro128p_uniform_float32(rng_states, now_id)
                new_status=-1
                p_sum=0
                for k in range(0,trasition_graph_degree[now_status]):
                    if p>= p_sum and p<=p_sum+trasition_graph_prob[now_status][k]:
                        new_status=trasition_graph_node[now_status][k]
                        new_delay=trasition_graph_delay[now_status][k]
                        break
                    p_sum+=trasition_graph_prob[now_status][k]
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
def update_infected_points_gpu(agent_initial_points,agent_infected_status,subarea_agents,new_event_mutex,new_event_pointer,new_event_type,new_event_agent,new_event_delay,rng_states,subarea_free_list_front,subarea_free_list_rear,next_infected_status,temp,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            len=math.sqrt(N)
            len=int(len)
            x=now_id/len
            y=now_id%len
            x=int(x)
            y=int(y)
            #该区域当前有多少agent
            front=subarea_free_list_front[x][y]
            rear=subarea_free_list_rear[x][y]
            cnt=0
            for k in range(0,subarea_agents.shape[2]):
                if subarea_agents[x][y][k]!=-1:
                    temp[x][y][cnt]=subarea_agents[x][y][k]
                    cnt+=1
            #随机从该区域选择若干对agent进行接触，平均每个agent接触2人
            ave_contact=1
            now=0
            while now<ave_contact*cnt/2:
                a=xoroshiro128p_uniform_float32(rng_states, now_id)*(cnt-1)
                b=xoroshiro128p_uniform_float32(rng_states, now_id)*(cnt-1)
                a=int(a)
                b=int(b)
                agent_a=temp[x][y][a]
                agent_b=temp[x][y][b]
                now += 1
                if agent_infected_status[agent_a] > 0 or agent_infected_status[agent_b] > 0:
                    agent_initial_points[agent_a] -= 1
                    agent_initial_points[agent_b] -= 1
                    if agent_initial_points[agent_a] == 0:  # 若减为0，插入新事件
                        next_infected_status[agent_a] = 1  # 1代表刚被感染
                        while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                            continue
                        new_event_type[new_event_pointer[0]] = 2
                        new_event_delay[new_event_pointer[0]] = 0
                        new_event_agent[new_event_pointer[0]] = agent_a
                        new_event_pointer[0] += 1
                        cuda.atomic.exch(new_event_mutex, 0, 0)
                    if agent_initial_points[agent_b] == 0:  # 若减为0，插入新事件
                        next_infected_status[agent_b] = 1
                        while cuda.atomic.compare_and_swap(new_event_mutex, 0, 1) == 1:
                            continue
                        new_event_type[new_event_pointer[0]] = 2
                        new_event_delay[new_event_pointer[0]] = 0
                        new_event_agent[new_event_pointer[0]] = agent_b
                        new_event_pointer[0] += 1
                        cuda.atomic.exch(new_event_mutex, 0, 0)


@cuda.jit
def statis_region_infected_cnt_gpu(agent_position,agent_infected_status,temp,temp_mutex,N):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if idx >= N:
        return
    id_in_warp = idx % cuda.warpsize
    if id_in_warp != 0:
        return
    for i in range(0, cuda.warpsize):
        now_id = idx + i
        if now_id < N:
            if agent_infected_status[now_id]!=1:
                continue
            x = agent_position[now_id][0]
            y = agent_position[now_id][1]
            x = int(x / 10)
            y = int(y / 10)
            while cuda.atomic.compare_and_swap(temp_mutex[x][y], 0, 1) == 1:
                continue
            temp[x][y] += 1
            cuda.atomic.exch(temp_mutex[x][y], 0, 0)


class EpidemicSceneGPU(Scene):
    def __init__(self, event_handler, agents, families, works, geoinfo, traffic, trasition_graph, interval=1,duration=168):
        super().__init__(event_handler, agents, geoinfo,interval, duration)
        self.trasition_graph=trasition_graph
        self.families = families  # 家庭
        self.works = works  # 企业
        self.trasition_graph = trasition_graph
        self.current_step = 0
        self.total_steps=duration

        #numpy arrays FOR GPU
        #activity type统一编号如下：
        #0:H 1:W 2:L 3:S
        self.agent_family=np.zeros(len(self.agents)).astype(int)
        self.agent_work = np.zeros(len(self.agents)).astype(int)
        self.agent_activity_type=np.zeros([len(self.agents),8]).astype(int)
        self.agent_activity_duration=np.zeros([len(self.agents),8])
        self.agent_activity_distance=np.zeros([len(self.agents),8])
        self.agent_activity_total=np.zeros([len(self.agents),8])
        self.agent_current_activity_index=np.zeros(len(self.agents)).astype(int)
        self.agent_infected_status=np.zeros(len(self.agents)).astype(int)
        self.agent_next_infected_status=np.zeros(len(self.agents)).astype(int)
        self.agent_daily_contacts=np.zeros([len(self.agents),48])
        self.agent_daily_contacts_pointer=np.zeros(len(self.agents))
        self.agent_position=np.zeros([len(self.agents),2])
        self.agent_infected_status=np.zeros([len(self.agents)]).astype(int)
        self.agent_initial_points=np.zeros([len(self.agents)]).astype(int)

        self.family_size=np.zeros(len(self.families)).astype(int)
        self.family_members=np.zeros([len(self.families),5]).astype(int)
        self.family_location=np.zeros([len(self.families),2])
        self.family_agents=np.zeros([len(self.families),5])    #当前family内的agent(区别于self.family_member)
        self.work_size=np.zeros(len(self.works)).astype(int).astype(int)
        self.work_members=np.zeros([len(self.works),50]).astype(int)
        self.work_location=np.zeros([len(self.works),2])
        self.work_agents=np.zeros([len(self.works),5])


        self.building_location=None     #每个建筑的位置(以grid为单位)
        self.grid_buildings=np.zeros([self.geoinfo.n_grids,self.geoinfo.n_grids,20]) #记录每个网格内的建筑列表（建筑编号）
        self.grid_buildings_total=None    #每个网格内的建筑总数
        self.building_size=None   #每个建筑的规模
        self.building_agents=np.zeros([len(self.buildings),300]) #记录当前每个建筑内部的agents
        #每个building的agent列表的空闲位置索引，以循环队列的形式实现
        self.buildng_agents_free_list=np.zeros([len(self.buildings),300]).astype(int)
        #front和rear指针
        self.building_agents_free_list_front=np.zeros([len(self.buildings),300]).astype(int)
        self.building_agents_free_list_rear = np.zeros([len(self.buildings),300]).astype(int)
        self.building_agents_cnt=None   #当前每个building内部的agent数量
        self.building_near_hospital=None #存储每个建筑就近的医院
        self.new_event_mutex = np.zeros(1).astype(int)
        self.new_event_pointer = np.zeros(1).astype(int)
        self.new_event_type = np.zeros(10000000).astype(int)
        self.new_event_delay = np.zeros(10000000)
        self.new_event_agent = np.zeros(10000000).astype(int)

        #感染状态转移图相关
        #规定感染状态的编号，0-未感染(Susceptible) 1-住院 2-恢复(Recovered/出院) 3-死亡
        self.trasition_graph_node=None   #结点
        self.trasition_graph_prob=None   #转移概率
        self.trasiton_graph_delay=None   #转移时延
        self.trasition_graph_degree=None

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


        #感染状态统计
        self.step_infected_status_cnt=None

        self.init_arrays_for_gpu()
        self.transmit_arrays_to_gpu()

        self.fill_agent_activity_center(self)
        self.init_agent_position()
        self.current_step = 0
        #self.init_infected_trasition_events(self)
        self.init_events(self)
        self.event_handler.insert_new_events(self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay)
        self.new_event_pointer[0]= 0
        self.plt=plt
        self.figure=self.plt.figure()
        self.plt.rcParams['agg.path.chunksize'] = 10000000
        #self.plt.ion()
        print('scene init done')
        print('total agents:',len(self.agents))
        print('agent0 init position:',self.agent_position[0][0], self.agent_position[0][1])
        print('agent0 work position:',self.work_location[self.agent_work[0]][0],self.work_location[self.agent_work[0]][1])
        print('agent0 activity0 duration:',self.agent_activity_duration[0][0])
        #绘制land use图层
        #self.geoinfo.plot_landuse(self.plt)


    # 初始化用于gpu计算的数组
    def init_arrays_for_gpu(self):
        for i in range(0,self.agent_commuting_distance.shape[0]):
            self.agent_commuting_distance[i]=self.agents[i].commuting_distance

        for i in range(0,self.agent_activity_type.shape[0]):
            for j in range(0,self.agent_activity_type.shape[1]):
                self.agent_activity_type[i][j]=self.agents[i].activity_type[j]

        for i in range(0,self.agent_activity_duration.shape[0]):
            for j in range(0,self.agent_activity_duration.shape[1]):
                self.agent_activity_duration[i][j]=self.agents[i].activity_duration[j]

        for i in range(0,self.agent_activity_range.shape[0]):
            for j in range(0,self.agent_activity_range.shape[1]):
                self.agent_activity_range[i][j]=self.agents[i].activity_range[j]

        for i in range(0,self.family_size.shape[0]):
            self.family_size[i]=self.families[i].size

        for i in range(0,self.work_size.shape[0]):
            self.work_size[i]=self.works[i].size

        for i in range(0,self.agent_infected_status.shape[0]):
            self.agent_infected_status[i]=self.agents[i].infected_status

        for i in range(0,self.agent_initial_points.shape[0]):
            self.agent_initial_points[i]=self.agents[i].initial_points

        #subarea_agents初始化为-1
        temp=np.ones(self.subarea_agents.shape)
        self.subarea_agents=(self.subarea_agents-temp).astype(int)
        self.init_subarea_free_list(self)

        al_prob=self.trasition_graph['prob']
        al_delay=self.trasition_graph['delay']
        max_degree=-1   #最大度
        for i in range(0,len(al_prob.al)):
            max_degree=max(max_degree,len(al_prob.al[i]))
        self.trasition_graph_node=np.zeros([al_prob.n_nodes,max_degree]).astype(int)
        self.trasition_graph_prob = np.zeros([al_prob.n_nodes, max_degree])
        self.trasition_graph_delay = np.zeros([al_prob.n_nodes, max_degree])
        self.trasition_graph_degree=np.zeros(al_prob.n_nodes).astype(int)
        self.step_infected_status_cnt=np.zeros([self.total_steps,al_prob.n_nodes]).astype(int)

        for i in range(0,len(al_prob.al)):
            self.trasition_graph_degree[i]=len(al_prob.al[i])

        for i in range(0,len(al_prob.al)):
            for k in range(0,len(al_prob.al[i])):
                self.trasition_graph_node[i][k]=al_prob.al[i][k].y

        for i in range(0,len(al_prob.al)):
            for k in range(0,len(al_prob.al[i])):
                self.trasition_graph_prob[i][k]=al_prob.al[i][k].w

        for i in range(0,len(al_delay.al)):
            for k in range(0,len(al_delay.al[i])):
                self.trasition_graph_delay[i][k]=al_delay.al[i][k].w

    #将频繁读写的array预先装入gpu
    def transmit_arrays_to_gpu(self):
        self.agent_family=cuda.to_device(self.agent_family)
        self.family_members=cuda.to_device(self.family_members)
        self.family_real_size=cuda.to_device(self.family_real_size)
        self.geoinfo.land_use=cuda.to_device(self.geoinfo.land_use)
        self.family_location=cuda.to_device(self.family_location)
        self.work_location=cuda.to_device(self.work_location)
        self.agent_work=cuda.to_device(self.agent_work)
        self.work_members=cuda.to_device(self.work_members)
        self.work_real_size=cuda.to_device(self.work_real_size)
        self.agent_activity_duration=cuda.to_device(self.agent_activity_duration)
        self.new_event_agent=cuda.to_device(self.new_event_agent)
        self.new_event_type = cuda.to_device(self.new_event_type)
        self.new_event_delay = cuda.to_device(self.new_event_delay)
        self.new_event_mutex = cuda.to_device(self.new_event_mutex)
        self.new_event_pointer = cuda.to_device(self.new_event_pointer)
        self.subarea_agents=cuda.to_device(self.subarea_agents)
        #self.agent_infected_status=cuda.to_device(self.agent_infected_status)
        self.agent_initial_points=cuda.to_device(self.agent_initial_points)


    @staticmethod
    def init_subarea_free_list(self):
        N = self.geoinfo.n_grids * self.geoinfo.n_sub * self.geoinfo.n_grids * self.geoinfo.n_sub
        init_subarea_free_list_gpu[math.ceil(N/1024),1024](self.subarea_agents_free_list,self.subarea_agents_free_list_front,self.subarea_agents_free_list_rear,N)

    def init_agent_position(self):
        for i in range(0,len(self.agents)):
            self.agent_position[i][0]=self.agent_activity_center[i][0][0]
            self.agent_position[i][1] = self.agent_activity_center[i][0][1]

    #为初始感染者注册状态转移事件
    @staticmethod
    def init_infected_trasition_events(self):
        init_infected_trasition_events_gpu[math.ceil(len(self.agents)/1024),1024](self.agent_infected_status,self.agent_next_infected_status,self.new_event_mutex,self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay,len(self.agents))

    #注册初始事件
    @staticmethod
    def init_events(self):
        init_events_gpu[math.ceil(len(self.agents)/1024),1024](self.agent_activity_duration,self.new_event_mutex,self.new_event_pointer, self.new_event_type,
                                                             self.new_event_delay, self.new_event_agent,len(self.agents))
        cuda.synchronize()

    #一步迭代
    def update(self):
        self.event_handler.to_numpy()
        # 触发一般事件，生成新的一般事件
        self.event_step(self)
        self.update_infected_points(self)
        self.statis_infected_status()
        #将新生成的一般事件统一注册到event_handler
        self.event_handler.insert_new_events(self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay)
        self.new_event_pointer[0] = 0
        self.current_time=self.current_step*self.interval
        if self.current_time>0 and self.current_time%24==0:    #每24小时更新agent的感染状态，根据每日接触列表
            self.update_infected_status()
        print('step',self.current_step,'done')
        print('now infected cnt:',self.agent_infected_status[self.agent_infected_status==1].shape[0])

    @staticmethod
    def event_step(self):
        if self.event_handler.now_event_type.shape[0]==0 or self.event_handler.now_event_type[0]==-1:
            return

        rng_states = create_xoroshiro128p_states(1024 * math.ceil(self.event_handler.now_event_type.shape[0] / 1024), seed=time.time())
        event_step_gpu[math.ceil(self.event_handler.now_event_type.shape[0]/ 1024), 1024](self.event_handler.now_event_type, self.event_handler.now_event_agent, self.new_event_mutex,self.new_event_pointer, self.new_event_type
                                                            , self.new_event_agent, self.new_event_delay,self.agent_current_activity_index,
                                                            self.agent_activity_type, self.agent_activity_center,self.agent_activity_duration,
                                                            self.agent_infected_status,
                                                                                          rng_states,
                                                                                          self.agent_next_infected_status,
                                                                                          self.trasition_graph_degree,
                                                                                          self.trasition_graph_prob,
                                                                                          self.trasition_graph_node,
                                                                                          self.trasition_graph_delay,
                                                            self.event_handler.now_event_type.shape[0])
        cuda.synchronize()



    @staticmethod
    def update_infected_points(self):
        N=self.geoinfo.n_grids*self.geoinfo.n_sub*self.geoinfo.n_grids*self.geoinfo.n_sub
        rng_states = create_xoroshiro128p_states(1024 * math.ceil(N / 1024), seed=time.time())
        temp=np.zeros(self.subarea_agents.shape).astype(int)
        update_infected_points_gpu[math.ceil(N/1024),1024](self.agent_initial_points,self.agent_infected_status,self.subarea_agents,self.new_event_mutex,self.new_event_pointer,self.new_event_type,self.new_event_agent,self.new_event_delay,rng_states,self.subarea_agents_free_list_front,self.subarea_agents_free_list_rear,self.agent_next_infected_status,temp,N)
        cuda.synchronize()

    def statis_infected_status(self):
        now_step=self.current_step
        for i in range(0,self.step_infected_status_cnt.shape[1]):
            self.step_infected_status_cnt[now_step][i]=(self.agent_infected_status[self.agent_infected_status==i]).shape[0]


    def update_infected_status(self):
        pass

    def visualize(self):
        # self.plt.clf()
        xs, ys = zip(*self.agent_position)
        # self.plt.scatter(xs, ys,s=0.1,alpha=0.5)
        #self.plt.show()
        #self.plt.pause(0.5)
        # print('Agent0 current activity index:',self.agent_current_activity_index[0])
        # print('agent0:',self.agent_position[0][0],self.agent_position[0][1])

    def step(self):
        self.current_step+=1
        self.traffic.step()
        self.event_handler.step()

