import numpy as np
import itertools
import utils.global_variables as global_variables



def ChannelGenerator(n_samples, global_seed, SNR_MAX, Channel_variation=2):
    
    # log = helpers.get_configured_logger('CHANNEL', level= "INFO")
    #log.info("Generating Channel..   ")

    np.random.seed(global_seed)
    
    scale_parameter = 0.1 * Channel_variation  # Influences the variance
    multiplier = 16
    
    # Generate samples with second granularity
    n_channel_samples = n_samples//100
    random_realization = np.random.rayleigh(scale_parameter, n_channel_samples ) + 1
    channel_values = multiplier*random_realization

    # Manual fix...
    channel_values = channel_values + 20  # TODO CHANNEL FIX TO BEST VALUE - CHANGE IT IF NECESSARY
    
    channel_values[channel_values > SNR_MAX] = SNR_MAX
    channel_values[channel_values < 10] = 10
    tmp_channel = channel_values.tolist()
    
    # Repeat SNR value for all the TTIs within a second (i.e. x1000)
    channel = list(itertools.chain.from_iterable(itertools.repeat(x, 100) for x in tmp_channel))

    #log.debug("Mean of the Rayleigh distribution: " + format(scale_parameter*math.sqrt(math.pi/2)))
    #log.debug("Variance of the Rayleigh distribution: " + format(((4-2)/math.pi)*scale_parameter*scale_parameter))
    '''
    log.info("Mean of the SNR distribution: " + format(np.mean(channel)))
    log.info("Variance of the SNR distribution: " + format(np.var(channel)))
    '''
    return channel


############################################################################################
def FromSNR_toChannel(PRB_ALLOC, channel_distribution):
    """
    Converts SNR distribution in Mbps based on PRB allocation of the slice
    
    """
    length_ch_distribution = len(channel_distribution)

    #Convert channel SNR (Distribution Generated before) in available Mb/s
    CQI_report_index = np.digitize(channel_distribution, global_variables.CQI_to_SNR[0])
    
    #Build channel behavior
    if PRB_ALLOC != 0 and PRB_ALLOC <= 273:
        MCS_index = global_variables.CQI_to_MCS[CQI_report_index, 4]
        tbs_index = global_variables.MCS_to_TBS[0, MCS_index]
        AvailableBps = int(global_variables.TBS_MATRIX[tbs_index, PRB_ALLOC - 1])*4   # MIMO HERE
        
    # elif PRB_ALLOC > 100:
    #     AvailableBps = np.zeros(length_ch_distribution, dtype=int)
    #     MCS_index = global_variables.CQI_to_MCS[CQI_report_index, 4]
    #     tbs_index = global_variables.MCS_to_TBS[0, MCS_index]
    #     # For PRBs > 100 (5G) simply sum the available tbs
    #     times = PRB_ALLOC // 100
    #     residual = PRB_ALLOC % 100
    #
    #     for i in range(0, length_ch_distribution):
    #         tbs_value = 0
    #         for time in range(times):
    #             tbs_value = tbs_value + global_variables.TBS_MATRIX[tbs_index[i], 100 - 1]
    #         tbs_value = tbs_value + global_variables.TBS_MATRIX[tbs_index[i], residual - 1]
    #         AvailableBps[i] = tbs_value    # MIMO HERE
    else:
        AvailableBps = np.zeros(length_ch_distribution)

    return AvailableBps


