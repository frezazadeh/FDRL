import gym
from gym import spaces
from collections import OrderedDict
import configparser
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.libHelpers import get_configured_logger
import time
import utils.libChannel_Generator as CG

class MultiDict(OrderedDict):
    _unique_SLICE = -1  # class variable
    _unique_BS = -1

    def __setitem__(self, key, val):
        if isinstance(val, dict):
            if key == 'BaseStation':
                self._unique_BS += 1
                key += (str(self._unique_BS))
            if key == 'Slice':
                self._unique_SLICE += 1
                key += (str(self._unique_SLICE) + str(0))
        OrderedDict.__setitem__(self, key, val)


def registration(max_episode, log_level):
    gym.envs.register(id='BS-v0',
                      entry_point='custom_BS:BSEnv',
                      max_episode_steps=max_episode,
                      kwargs={'base_station': None,
                              'monitoring_time_window': None,
                              'configfile': None,
                              'log_level': log_level})


class BSEnv(gym.Env):
    def __init__(self, base_station=None, monitoring_time_window=None, configfile=None, log_level=None):
        # Simulation Section
        self.MONITORING_TIME_WINDOW = monitoring_time_window
        self.name = 'BaseStation' + str(base_station.ID)
        self.bs = base_station
        self.PRBs_set = list(self.bs.allowed_prb_alloc)
        self.action_space = spaces.Discrete(len(self.PRBs_set))
        self.log = get_configured_logger("GYM Env"+format(self.bs.ID), log_level)
        self.config = configparser.RawConfigParser(defaults=None, dict_type=MultiDict, strict=False)
        self.config.read(configfile)
        self.state = None

        # please, keep the lase element obs space element devoted to track agents prb allocation
        OBS_SPACE_DIM = 1
        high = np.ones((OBS_SPACE_DIM,), dtype=np.float32)
        low = np.zeros((OBS_SPACE_DIM,), dtype=np.float32)
        self.observation = []
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        if self.config.get('DRL', '_class_ML') == 'Inference':
            self.q_percentile_th = 79
            self.score_log = []
            self.q_p = 80

        # Observations from BS
        self.dl_bitrate = None
        self.dl_capacity = None
        self.ul_capacity = None
        self.cpu = None
        self.snr = None

    def step(self, decisions):

        actions = decisions[0]
        leftover_prbs = decisions[2]

        self.log.debug(" -----   STEP -----")
        self.prb_allocation = [self.PRBs_set[i] for i in actions]
        

        # set decision
        for ind, id in enumerate(self.bs.Allocated_slices):
            self.bs.decision_interfaces[ind].request(self.prb_allocation[ind], self.prb_allocation[ind], self.bs.Traffic_gen[ind])

        # wait for observation time
        time.sleep(self.MONITORING_TIME_WINDOW)

        # get results
        self.dl_bitrate = [self.bs.monitor_interface.get_measurements(measurement_name='traffic', metric='Download Speed',
                                                                     time_window=self.MONITORING_TIME_WINDOW,
                                                                     operation='mean', filter_out='("traffic_gen_ID" = '+"'"+s.split('_')[-1]+"'"+')')/1e6 for s in self.bs.Traffic_gen]
        self.dl_capacity = [self.bs.monitor_interface.get_measurements(measurement_name='cell', metric='cell_dl_bitrate',
                                                                     time_window=self.MONITORING_TIME_WINDOW,
                                                                     operation='mean', filter_out='("ticker" = '+"'cell_id_"+s.split('_')[-1]+"'"+')')/1e6 for s in self.bs.Traffic_gen]
        self.ul_capacity = [self.bs.monitor_interface.get_measurements(measurement_name='cell', metric='cell_ul_bitrate',
                                                                     time_window=self.MONITORING_TIME_WINDOW,
                                                                     operation='mean', filter_out='("ticker" = '+"'cell_id_"+s.split('_')[-1]+"'"+')')/1e6 for s in self.bs.Traffic_gen]
       
        self.cpu = self.bs.monitor_interface.get_measurements(measurement_name='general', metric='cpu',
                                                                     time_window=self.MONITORING_TIME_WINDOW,
                                                                     operation='mean')

        
        # todo: to solve next 3 todos check if from the ue get the cell id is given
        '''
        self.dl_mcs = [self.bs.monitor_interface.get_measurements(measurement_name='ues', metric='dl_mcs',              #todo: extend to multiple slices
                                                                 time_window=self.MONITORING_TIME_WINDOW,
                                                                 operation='mean')]
        self.ul_mcs = [self.bs.monitor_interface.get_measurements(measurement_name='ues', metric='ul_mcs',              #todo: extend to multiple slices
                                                              time_window=self.MONITORING_TIME_WINDOW,
                                                              operation='mean')]
        '''
        self.snr = [self.bs.monitor_interface.get_measurements(measurement_name='ues', metric='pusch_snr',              #todo: extend to multiple slices
                                                                 time_window=self.MONITORING_TIME_WINDOW,
                                                                 operation='mean') for s in self.bs.Allocated_slices]

        # todo: observation to cost and state
        # Update cost function and give reward to agents
        cost_per_agent = np.zeros((len(self.bs.Allocated_slices)))
        allocation_gap_out = []
        for i in range(len(self.bs.Allocated_slices)):
            #min_channel_capacity = CG.FromSNR_toChannel(self.bs.prb_chunk_size, [self.bs.snr_max])*1000/1e6

            allocation_gap = self.dl_capacity[i] - self.dl_bitrate[i]

            print("Slice: ", i,"--- DL bitrate ", self.dl_bitrate[i], "--- DL capacity: ", self.dl_capacity[i], "--- difference: ", self.dl_capacity[i] - self.dl_bitrate[i])
            # REWARD
            #cost_per_agent[i] = - np.abs(allocation_gap)  # reward function
            cost_per_agent[i] = - np.square(allocation_gap)  # reward function
            '''
            if 0<= self.dl_bitrate[i] <3 and self.prb_allocation [i] == 2:
                cost_per_agent[i] /= 10
            elif 3<= self.dl_bitrate[i] <5 and self.prb_allocation [i] == 4:
                cost_per_agent[i] /= 10
            elif 5<= self.dl_bitrate[i] <7 and self.prb_allocation [i] == 6:
                cost_per_agent[i] /= 10
            elif 7<= self.dl_bitrate[i] <9 and self.prb_allocation [i] == 8:
                cost_per_agent[i] /= 10
            elif 9<= self.dl_bitrate[i] <11 and self.prb_allocation [i] == 10:
                cost_per_agent[i] /= 10                                         
            elif 11<= self.dl_bitrate[i] <13 and self.prb_allocation [i] == 12:
                cost_per_agent[i] /= 10
            elif 13<= self.dl_bitrate[i] <15 and self.prb_allocation [i] == 14:
                cost_per_agent[i] /= 10                
            elif 15<= self.dl_bitrate[i] <17 and self.prb_allocation [i] == 16:
                cost_per_agent[i] /= 10                
            elif 17<= self.dl_bitrate[i] <19 and self.prb_allocation [i] == 18:
                cost_per_agent[i] /= 10                
            elif 19<= self.dl_bitrate[i] <21 and self.prb_allocation [i] == 20:
                cost_per_agent[i] /= 10                
            elif 21<= self.dl_bitrate[i] <23 and self.prb_allocation [i] == 22:
                cost_per_agent[i] /= 10                
            else:
                cost_per_agent[i] = -500
                
            '''    
                
            self.log.info("Reward {}:  {}".format(i, cost_per_agent[i]))
            allocation_gap_out.append(allocation_gap)
            
        print("Reward-----------------------------------------------------", cost_per_agent)

        self.bs.decision_tracking_interface.write_decision(self.prb_allocation, self.prb_allocation, [a* 1e6 for a in allocation_gap_out], [0 for _ in range(len(allocation_gap_out))])
        #self.state = [np.array([np.round(max(0, allocation_gap_out[i]/1200), 3),
        #                        np.round(0, 3),
        #                        np.round(self.snr[i]/150, 3),
        #                        np.round(self.dl_bitrate[i]/1200, 3),
        #                        leftover_prbs[i]]).reshape(1, self.observation_space.shape[0]) for i in range(len(self.bs.Allocated_slices))]
        #self.state = [np.array([np.round(self.dl_capacity[i]/1200, 3),
                                #np.round(self.dl_bitrate[i]/1200, 3),
                                #leftover_prbs[i]]).reshape(1, self.observation_space.shape[0]) for i in range(len(self.bs.Allocated_slices))]
        self.state = [np.array([np.round(self.dl_capacity[i]/1200, 3)]).reshape(1, self.observation_space.shape[0]) for i in range(len(self.bs.Allocated_slices))]
        print("------------------------------------------------------------------")
        print("-----------------self.dl_capacity----------------------", self.dl_capacity)
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        print("----------------- state inside custom_BS"," :", self.state)
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")                         
                                
        self.log.info('State: ' + format( self.state))
        self.log.debug(" ----- END STEP -----")

        self.bs.decision_tracking_interface.write_reward(cost_per_agent)

        return self.state, cost_per_agent, False, dict(drop_t=np.round(self.norm_dropped_traffic, 3),
                                                       b_latency=np.round(self.norm_buffer_latency, 3),
                                                       o_channel=np.round(self.norm_SNR_channel, 3),
                                                       m_avg_channel_capacity=np.round(self.dl_capacity, 3),
                                                       PRB_ALLOCATED=self.prb_allocation,
                                                       traf_tracing=np.round(self.avg_slice_traffic_request, 3)
                                                       )

    def reset(self):

        if self.config.get('DRL', '_class_ML') == 'Inference':
            self.q_percentile_th += 1

        self.index_prb = 0
        self.prb_allocation = []
        self.penalty = 0
        self.state = [ np.ones((1,self.observation_space.shape[0])) for _ in range(len(self.bs.Allocated_slices))]
        self.buffer_size = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.buffer_latency = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.consumed_PRBs_state = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.dropped_traffic = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.traffic_request = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.obs_channel = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.prediction_values = []
        self.avg_slice_traffic_request = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.norm_dropped_traffic = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.norm_buffer_latency = [0.0 for _ in range(len(self.bs.Allocated_slices))]
        self.norm_SNR_channel = [0.0 for _ in range(len(self.bs.Allocated_slices))]

        return self.state

    def Penalty_PRB(self):
        b = np.sum(self.prb_allocation)
        if b > 100:
            self.penalty = b - 100
        else:
            self.penalty = 0

    def scale(self, val, src, dst):
        """
        Scale the given value from the scale of src to the scale of dst.
        """
        return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]

    # todo Farhad Latency-KPI-SLA-Statistical
    def run_Lconstraints(self):
        viol_bound_L = 0
        coef = 0
        penalty_statistical = []
        score_frame = []
        for i in range(len(self.bs.Allocated_slices)):
            prediction_values = []
            if i == 0:
                viol_bound_L = 9.8
                coef = 10
            elif i == 1:
                viol_bound_L = 49.5
                coef = 50
            elif i == 2:
                viol_bound_L = 19
                coef = 20
            for j in range(len(self.prediction_values)):
                prediction_values.append(self.prediction_values[j][i] * coef)

            pred = tf.convert_to_tensor(prediction_values)

            if self.g.get('DRL', '_class_ML') == 'Train':
                percent = tfp.stats.percentile(pred, q=95)
            elif self.config.get('DRL', '_class_ML') == 'Inference':
                # percent = tfp.stats.percentile(pred, q=self.q_percentile_th)
                self.q_p += 1
                percent = tfp.stats.percentile(pred, q=95)
                score_frame.append(float(percent))

            # print(i, float(percent) )
            # print("--------------------------------")

            if float(percent) <= viol_bound_L:
                penalty_statistical.append(0)
            else:
                # print(i, " ", "YES")
                penalty_statistical.append(-0.88)
        if self.config.get('DRL', '_class_ML') == 'Inference':
            self.score_log.append(score_frame)
            np.savez('score_log', np.vstack(self.score_log), allow_pickle=False)
        print(penalty_statistical)
        return (penalty_statistical)
