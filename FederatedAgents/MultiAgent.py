import numpy as np
from keras import models
from tensorflow.keras.optimizers import Adam
from utils.libHelpers import get_configured_logger

class Multi_Agent:
    def __init__(self, agent_class, n_agents, env, config, config_name, log_level, slices_id):
        self.config = config
        self.config_name = config_name
        self.env = env
        self.slices_id = slices_id
        self.prb_set_internal = env.PRBs_set
        self.prb_set_internal[0] = 0 # use the first entry as zero for agents. Amarisoft does not allow
        self.name = env.name
        self.len_episode = int(self.config.get('DRL', 'Max_Episode'))
        self.federation_type = str(self.config.get('DRL', 'Intra_FDRL_Strategy'))
        self.agents = [agent_class(self.env, i, config) for i in range(n_agents)]
        self.n_agents = n_agents
        self.epsilon_agents = [0 for _ in range(self.n_agents)]
        self.log = get_configured_logger(__name__, log_level)
        self.drl_class = self.config.get('DRL', '_class_ML')
        if (int(self.config.get('DRL', 'Priority_Agent'))) == 1:
            self.log.debug("Fair Agent selection")
            # Add fairness in the decision process
            self.prioritize_agents = self.uniform_priority
        elif (int(self.config.get('DRL', 'Priority_Agent'))) == 0:
            self.log.debug("Sequential Agent selection")
            self.prioritize_agents = self.sequential_priority

    def uniform_priority(self):
        return list(np.argsort(np.random.uniform(0, 1, self.n_agents)))

    def sequential_priority(self):
        return [i for i in range(self.n_agents)]
    '''
    def intra_slice_federation(self):
        if self.drl_class == 'DDQN' and self.n_agents > 1:
            fed_model = [np.zeros(self.agents[0].train_network.get_weights()[w].shape)
                         for w in range(len(self.agents[0].train_network.get_weights()))]
            for i in range(self.n_agents):
                curr_model = self.agents[i].train_network.get_weights()
                for w in range(len(curr_model)):
                    fed_model[w] = fed_model[w] + curr_model[w]
            for w in range(len(fed_model)):
                fed_model[w] = fed_model[w] / self.n_agents
            for i in range(self.n_agents):
                self.agents[i].update_agent_network(fed_model)
    '''

    def run(self):
        score_log = [[] for _ in range(self.n_agents)]
        d_traffic_log = [[] for _ in range(self.n_agents)]
        b_latency_log = [[] for _ in range(self.n_agents)]
        o_channel_log = [[] for _ in range(self.n_agents)]
        m_avg_channel_capacity_log = [[] for _ in range(self.n_agents)]
        average_PRBs_log = [[] for _ in range(self.n_agents)]
        traffic_tracing_log = [[] for _ in range(self.n_agents)]

        if self.config.get('DRL', 'code_model') == 'Inference':
            fdrl_strategy = self.config.get('DRL', 'FDRL_Strategy')
            for z in range(self.n_agents):
                model = models.load_model('data/' + fdrl_strategy + '/Model/' + self.config.get('DRL', 'FDRL_Strategy')
                                          + str(z) + '.h5', compile=False)
                model.compile(loss=self.agents[z]._huber_loss,
                              optimizer=Adam(learning_rate=(float(self.config.get('DRL', 'learning_rate')))))
                self.agents[z].train_network = model
                self.agents[z].target_network = model
                self.agents[z].target_network.set_weights(self.agents[z].train_network.get_weights())
        current_state = self.env.reset()
        tarffic_list_store = []
        PRB_list_store = []
        gap_list_store = []
        
        if self.drl_class == 'DQN' or self.drl_class == 'DDQN' or self.drl_class == 'DDPG':
            for i in range(self.len_episode):
                self.log.info(' ------------------------------------- Internal Episode = {}/{}'.format(i, self.len_episode))
                initial_actions = [0 for z in range(self.n_agents)] # reset initial and opt action at each iteration
                #opt_action = [0 for z in range(self.n_agents)]
    
                #priority_order = self.prioritize_agents()
                if self.drl_class == 'DQN' or self.drl_class == 'DDQN' or self.drl_class == 'DDPG':
                    #self.log.debug("Priority_order: " + format(priority_order))
                    # Collect actions from agents given current state and update Leftover PRBs before next agent selection
                    if self.drl_class == 'DQN' or self.drl_class == 'DDQN':
                        for p_ind, agent_id in enumerate([0,1]):
                            print("------------------------------------------------------------------")
                            print("------------------------------------------------------------------")
                            print("----------------- state inside multi-agent","Agent",agent_id, " :", current_state[agent_id])
                            print("------------------------------------------------------------------")
                            print("------------------------------------------------------------------")
                            initial_actions[agent_id] = self.agents[agent_id].epsgreedyaction(current_state[agent_id])
                    elif self.drl_class == 'DDPG':
                        for p_ind, agent_id in enumerate([0,1]):
                            initial_actions[agent_id] = self.agents[agent_id].choose_action(current_state[agent_id][0], 0.1)
                # run decisions
                #action_order_list[0] = initial_actions
                new_states, rewards, done, info = self.env.step(initial_actions)
    
                # Collect statistics for every decision step
                for z in range(self.n_agents):
                    if self.drl_class == 'DQN' or self.drl_class == 'DDQN':
                        self.log.debug("Initial actions input of replay buffer!!")
                        self.agents[z].replay_memory.append([current_state[z], initial_actions[z], rewards[z], new_states[z], done]) #todo: check new and old state if they are actually different.
                    elif self.drl_class == 'DDPG':
                        transition = [current_state[z][0], [rewards[z]], [initial_actions[z]], new_states[z][0], [done]]
                        self.agents[z].memory.store(transition)
                    if self.drl_class == 'DQN' or self.drl_class == 'DDQN':
                        self.agents[z].train_buffer()
    
                    current_state[z] = new_states[z]
                    #current_state[z][0][-1] = 1
                    if z==0:
                        tarffic_list_store.append(info.get('traffic_sav_info'))
                        PRB_list_store.append(info.get('PRB_ALLOCATED'))
                        gap_list_store.append(info.get('gap_sav_info'))
                        
                    score_log[z].append(rewards[z])
                    d_traffic_log[z].append(info.get('drop_t')[z])
                    b_latency_log[z].append(info.get('b_latency')[z])
                    o_channel_log[z].append(info.get('o_channel')[z])
                    m_avg_channel_capacity_log[z].append(info.get('m_avg_channel_capacity')[z])
                    #average_PRBs_log[z].append(info.get('PRB_ALLOCATED')[z])
                    traffic_tracing_log[z].append(info.get('traf_tracing')[z])
                pass
                if self.drl_class == 'DDPG':
                    if (self.agents[z].memory.size() < self.agents[z].warmup):
                        continue
                    for z in range(self.n_agents):
                        batch = self.agents[z].memory.sample(self.agents[z].batchsize)
                        self.agents[z].critic_learn(batch)
                        self.agents[z].actor_learn(batch)
                        self.agents[z].soft_update()
            # Collect rewards and statistics of X decision steps and update self variables to be collected by main loop
            for z in range(self.n_agents):
                if np.mean(score_log[z]) > self.agents[z].best_score_FDRL:
                    self.agents[z].best_score_FDRL = np.mean(score_log[z])
                    if self.drl_class == 'DQN' or self.drl_class == 'DDQN':
                        self.agents[z].train_network.save( 'data/' + self.config.get('DRL', 'FDRL_Strategy') + '/' + self.config_name + "/" + str(self.agents[z].agent_id) + '.h5')
    
                self.agents[z].best_score = score_log[z]
                self.agents[z].best_d_traffic = d_traffic_log[z]
                self.agents[z].best_b_latency = b_latency_log[z]
                self.agents[z].best_o_channel = o_channel_log[z]
                self.agents[z].best_m_avg_channel_capacity = m_avg_channel_capacity_log[z]
                self.agents[z].best_average_PRBs = average_PRBs_log[z]
                self.agents[z].best_traffic_tracing = traffic_tracing_log[z]
    
            # Exploration Decay
            '''
            if self.drl_class == 'DQN' or self.drl_class == 'DDQN':
                for z in range(self.n_agents):
                    self.agents[z].epsilon -= self.agents[z].epsilon_decay
            '''
    
            # close environment is simulation is over
            if done:
                '''
                with open("tarffic_list_store.txt", mode='a') as f:
                    for s in tarffic_list_store:
                        f.write(str(s) + "," + "\n")
                        
                with open("PRB_list_store.txt", mode='a') as f:
                    for s in PRB_list_store:
                        f.write(str(s) + "," + "\n")
                        
                with open("gap_list_store.txt", mode='a') as f:
                    for s in gap_list_store:
                        f.write(str(s) + "," + "\n")
                        
                '''    
                
                with open("tarffic_list_store.txt", mode='a') as f:
                    for s in tarffic_list_store:
                        f.write(str(s) + "," + "\n")
                        
                with open("PRB_list_store.txt", mode='a') as f:
                    for s in PRB_list_store:
                        f.write(str(s) + "," + "\n")
                        
                with open("gap_list_store.txt", mode='a') as f:
                    for s in gap_list_store:
                        f.write(str(s) + "," + "\n")
                        
                 
                
                #print ("reward slic 1", score_log[0])
                #print ("reward slic 1//////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
                self.env.close()
            
        # TODO Add methods to retrieve single agent metrics, if necessary
        #def get_agent_epsilon(self, agent_id):
            #return self.agents[agent_id].epsilon
        elif self.drl_class == 'Dueling':
            #reward_down = [0,0]
            state = current_state
            #print("state-------------------------------------------------------", state)
            initial_actions = [0 for z in range(self.n_agents)]
            episode_reward = 0
            for time_steps in range(self.len_episode):
                for z in range(self.n_agents):
                    initial_actions[z] = self.agents[z].epsgreedyaction(state[z][0])
               
                next_state, reward, done, _ = self.env.step(initial_actions)
                #episode_reward += reward
                for z in range(self.n_agents):
                    self.agents[z].memoryrep(state[z][0], next_state[z][0], initial_actions[z], reward[z], done)
            
                if done:
                    break
                state = next_state
            
            
            
                #writer.add_scalar('episode reward', episode_reward, global_step=epoch)
