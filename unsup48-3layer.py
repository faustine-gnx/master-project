# VERY HELPFUL: http://neuralensemble.org/docs/PyNN/examples/simple_STDP.html?highlight=population
# file:///C:/Users/faust/Documents/pdm/Papers/02.%20Running%20PyNN%20Simulations%20on%20SpiNNaker%20-%20Lab%20Manual.pdf


import pyNN.spiNNaker as sim
import pylab
import pyNN.utility.plotting as pplt # then use plot.Figure, plot.Panel...
import matplotlib.pyplot as plt
import random as rand
import handleAER as pp #formerly preProcessing.py
import scipy.io as sio
import numpy as np
from pyNN.random import NumpyRNG, RandomDistribution
import scipy.io as sio
import math


__neuronType__ = sim.IF_curr_exp 

# Same params for layer 2 and 3 ?
__neuronParameters2__ = { # tau_m = Rm * cm --> Rm typically 100 Mohm = 10^8 ohm
                         # Rm typically: 10000 ohm/cm2
                         # v_thresh >> v_rest + tau_syn ?
                         # tau_m: independent of geometry (no /cm2) --> 10 ms
    'cm': 70,#2,  # (nF) Capacitance of the LIF neuron: membrane capacity - 1.0 
                         # typically: 1nF/cm2
                         # typical physiological value C = 0.29 nF
    'tau_m': 299.0, #110.0,  # (ms) RC circuit time-constant: membrane time constant - 20.0 # don't exceed 300ms (traj separated by 300ms) otherwise spikes from previous traj will be taken into account
    'tau_refrac': 300.0, # 40.0, # (ms) Duration of refractory period- 0.0 # typically 5 ms : after a spike, membrane potential is clamped to v_reset for a refractory period tau_refrac
    'v_reset': -70.0 , # (mV) Reset potential after a spike: voltage at which neuron is reset (typically lower than v_rest) - -65
    'v_rest': -65.0,  # (mV) Resting mebrane potential: ambient rest voltage of the neuron - -64
    'v_thresh': -55, #-50, # (mV) Spike threshold: threshold voltage at which the neuron spikes - -50
    'tau_syn_E': 2.0, #5.0,  # (ms) Rise time of the excitatory synaptic alpha function: excitatory input current decay time-constant - 5 # 2.728ms
    'tau_syn_I': 25.0, #10.0, # (ms) Rise time of the inhibitory synaptic alpha function: inhibitory input current decay time-constant - 5
    'i_offset': 0.0  # (nA) Offset current: base input current to add at each timestep - 
}

__neuronParameters3__ = { # tau_m = Rm * cm --> Rm typically 100 Mohm = 10^8 ohm
                         # Rm typically: 10000 ohm/cm2
                         # v_thresh >> v_rest + tau_syn ?
                         # tau_m: independent of geometry (no /cm2) --> 10 ms
    'cm': 3.0,#2,  # (nF) Capacitance of the LIF neuron: membrane capacity - 1.0 
                         # typically: 1nF/cm2
                         # typical physiological value C = 0.29 nF
    'tau_m': 5.0, #110.0,  # (ms) RC circuit time-constant: membrane time constant - 20.0 # don't exceed 300ms (traj separated by 300ms) otherwise spikes from previous traj will be taken into account
    'tau_refrac': 5.0, # 40.0, # (ms) Duration of refractory period- 0.0 # typically 5 ms : after a spike, membrane potential is clamped to v_reset for a refractory period tau_refrac
    'v_reset': -66.0 , # (mV) Reset potential after a spike: voltage at which neuron is reset (typically lower than v_rest) - -65
    'v_rest': -65.0,  # (mV) Resting mebrane potential: ambient rest voltage of the neuron - -64
    'v_thresh': -62, #-50, # (mV) Spike threshold: threshold voltage at which the neuron spikes - -50
    'tau_syn_E': 10.0, #5.0,  # (ms) Rise time of the excitatory synaptic alpha function: excitatory input current decay time-constant - 5 # 2.728ms
    'tau_syn_I': 10.0, #10.0, # (ms) Rise time of the inhibitory synaptic alpha function: inhibitory input current decay time-constant - 5
    'i_offset': 0.0  # (nA) Offset current: base input current to add at each timestep - 
}

__neuronParameters3test__ = { # tau_m = Rm * cm --> Rm typically 100 Mohm = 10^8 ohm
                         # Rm typically: 10000 ohm/cm2
                         # v_thresh >> v_rest + tau_syn ?
                         # tau_m: independent of geometry (no /cm2) --> 10 ms
    'cm': 30.0,#2,  # (nF) Capacitance of the LIF neuron: membrane capacity - 1.0 
                         # typically: 1nF/cm2
                         # typical physiological value C = 0.29 nF
    'tau_m': 50.0, #110.0,  # (ms) RC circuit time-constant: membrane time constant - 20.0 # don't exceed 300ms (traj separated by 300ms) otherwise spikes from previous traj will be taken into account
    'tau_refrac': 50.0, # 40.0, # (ms) Duration of refractory period- 0.0 # typically 5 ms : after a spike, membrane potential is clamped to v_reset for a refractory period tau_refrac
    'v_reset': -70.0 , # (mV) Reset potential after a spike: voltage at which neuron is reset (typically lower than v_rest) - -65
    'v_rest': -65.0,  # (mV) Resting mebrane potential: ambient rest voltage of the neuron - -64
    'v_thresh': -62, #-50, # (mV) Spike threshold: threshold voltage at which the neuron spikes - -50
    'tau_syn_E': 100.0, #5.0,  # (ms) Rise time of the excitatory synaptic alpha function: excitatory input current decay time-constant - 5 # 2.728ms
    'tau_syn_I': 100.0, #10.0, # (ms) Rise time of the inhibitory synaptic alpha function: inhibitory input current decay time-constant - 5
    'i_offset': 0.0  # (nA) Offset current: base input current to add at each timestep - 
}

# Higher membrane capacitance prevents neuron from spiking in answer to noise and too much time per pattern per pattern
# The lower Cm is, the fewer input spikes are needed to determine the output spike timing, therefore Cm should not be too high, 
# otherwise there will be no spikes at all. On the other hand, too low Cm would cause superfluous spikes.

# STDP parameters - for 2nd layer
__delay__ = 1.000 # (ms) 
tauPlus = 5.000 #20 # 15 # 16.8 from literature
tauMinus = 30.000 #100.000 #20 # 30 # 33.7 from literature
wMax2 = 10.000#9.999 #1 # G: 0.15
wMaxInit2 = 2.000 #10#0.1#0.100 # essayer normal distrib mu 5 avec additive
wMin2 = 0.001#0.001
aPlus = 0.300  #tum 0.016 #9 #3 #0.5 # 0.03 from literature
aMinus = 0.300 #.200 #0.100 #255 #tum 0.012 #2.55 #2.55 #05 #0.5 # 0.0255 (=0.03*0.85) from literature 
nbIter = 5
#weightFactor2 = 100

# STDP parameters - for 3rd layer
tauPlus3 = 5.000 #20 # 15 # 16.8 from literature
tauMinus3 = 30.000 #100.000 #20 # 30 # 33.7 from literature
wMax3 = 3.000#9.999 #1 # G: 0.15
wMaxInit3 = 3.000 #10#0.1#0.100 # essayer normal distrib mu 5 avec additive
wMin3 = 0.001#0.001
aPlus3 = 0.5  #tum 0.016 #9 #3 #0.5 # 0.03 from literature
aMinus3 = 0.5  #.200 #0.100 #255 #tum 0.012 #2.55 #2.55 #05 #0.5 # 0.0255 (=0.03*0.85) from literature 
nbIter = 5
testWeightFactor = 1.000
#weightFactor2 = 100

inputLayerSize = 1024 # 32*32
middleLayerSize = 48
outputLayerSize = 3
inhibWeight2 = 10.000 #*wMax #20 # same as wMax ? 5 --> 1/3 of threshold ? between 3 (1/5) and 15 (1/1) ? --> paper masquelier: 1/4 of threshold
inhibWeight3 = 10.000 # 50 for training l3 (?)

def plot_spiketrains(segment):
    plt.close('all')
    for spiketrain in segment.spiketrains:
        y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
        plt.plot(spiketrain, y, '.')
   
        #plt.ylabel(segment.name)
        plt.setp(plt.gca().get_xticklabels(), visible=False)

def plot_signal(signal, index, colour='b'):
    plt.close('all')
    label = "Neuron %d" % signal.annotations['source_ids'][index]
    plt.plot(signal.times, signal[:, index], colour, label=label)
    plt.ylabel("%s (%s)" % (signal.name, signal.units._dimensionality.string))
    plt.setp(plt.gca().get_xticklabels(), visible=False)
    plt.legend()

def plot_weight_heatmap2(untrained_weights, trained_weights, block=True):
    plt.close('all')
    if len(untrained_weights)>inputLayerSize:
        u_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                u_weights[i][j] = untrained_weights[k]/wMax
                k += 1
    else:
        u_weights = untrained_weights

    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                t_weights[i][j] = trained_weights[k]/wMax
                k += 1
    else:
        t_weights = trained_weights

    f, axarr = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    for lis in t_weights:
        for weight in lis:
            if weight > 1:
                weight = 1
    # print "weights after maxing them"
    # print weights
    # print "weights after training"
    # print trainedWeights
    

    axarr[0].imshow(u_weights, cmap='hot', interpolation='nearest')
    axarr[1].imshow(t_weights, cmap='hot', interpolation='nearest')

    axarr[0].set_title('Before Training')
    axarr[1].set_title('After Training')

    plt.show(block=block)

def plot_neuron_weight_heatmap2(untrained_weights, trained_weights, block=True):
    plt.close('all')
    if len(untrained_weights)>inputLayerSize:
        u_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                u_weights[i][j] = untrained_weights[k]/wMax
                k += 1
    else:
        u_weights = untrained_weights

    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                t_weights[i][j] = trained_weights[k]/wMax
                k += 1
    else:
        t_weights = trained_weights

    #neuron_u_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(outputLayerSize)]
    #neuron_t_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(outputLayerSize)]

    neuron_u_weights = np.zeros((middleLayerSize, int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))
    neuron_t_weights = np.zeros((middleLayerSize, int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))

    for j in range(middleLayerSize):
        for k in range(int(math.sqrt(inputLayerSize))):
            for l in range(int(math.sqrt(inputLayerSize))):
                neuron_u_weights[j][k][l] = u_weights[k+l*int(math.sqrt(inputLayerSize))][j]
                neuron_t_weights[j][k][l] = t_weights[k+l*int(math.sqrt(inputLayerSize))][j]

    N_to_be_plotted = []

    for N in range(len(neuron_t_weights)):
        if neuron_t_weights[N].any() > 0.5 :
            N_to_be_plotted.append(N)

    f, axarr = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    for N in neuron_t_weights:
        for lis in N:
            for weight in lis:
                if weight > 1:
                    weight = 1

    print len(N_to_be_plotted)

    for N in range(len(N_to_be_plotted)):
            #plt.close('all')
            axarr[0].imshow(neuron_u_weights[N+2], cmap='hot', interpolation='nearest')
            axarr[1].imshow(neuron_t_weights[N+2], cmap='hot', interpolation='nearest')

            axarr[0].set_title('Before Training, neuron ' + str(N))
            axarr[1].set_title('After Training, neuron ' + str(N))
            plt.show(block=block)

def plot_1neuron_weight_heatmap2(untrained_weights, trained_weights, neuron, block=True):
    plt.close('all')
    if len(untrained_weights)>inputLayerSize:
        u_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                u_weights[i][j] = untrained_weights[k]/wMax
                k += 1
    else:
        u_weights = untrained_weights

    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                t_weights[i][j] = trained_weights[k]/wMax
                k += 1
    else:
        t_weights = trained_weights

    #neuron_u_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(middleLayerSize)]
    #neuron_t_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(middleLayerSize)]

    neuron_u_weights = np.zeros((int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))
    neuron_t_weights = np.zeros((int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))
    
    for k in range(int(math.sqrt(inputLayerSize))):
        for l in range(int(math.sqrt(inputLayerSize))):
            neuron_u_weights[k][l] = u_weights[k+l*int(math.sqrt(inputLayerSize))][neuron] #*
            neuron_t_weights[k][l] = t_weights[k+l*int(math.sqrt(inputLayerSize))][neuron]

    f, axarr = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    for lis in neuron_t_weights:
        for weight in lis:
            if weight > 1:
                weight = 1

    #print len(neuron_u_weights), len(neuron_u_weights[0])
    axarr[0].imshow(neuron_u_weights, cmap='hot', interpolation='nearest')
    axarr[1].imshow(neuron_t_weights, cmap='hot', interpolation='nearest')

    axarr[0].set_title('Before Training, neuron ' + str(neuron))
    axarr[1].set_title('After Training, neuron ' + str(neuron))
    plt.show(block=block)

def plot_neuron_weight_heatmap_sequence2(trained_weights, block=True):
    plt.close('all')
    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                t_weights[i][j] = trained_weights[k]/wMax3
                k += 1
    else:
        t_weights = trained_weights
    
    neuron_t_weights = np.zeros((middleLayerSize, int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))

    for j in range(middleLayerSize):
        for k in range(int(math.sqrt(inputLayerSize))):
            for l in range(int(math.sqrt(inputLayerSize))):
                neuron_t_weights[j][k][l] = t_weights[k+l*int(math.sqrt(inputLayerSize))][j]

    N_to_be_plotted = []

    for N in range(len(neuron_t_weights)):
        if neuron_t_weights[N].any() > 0.5 :
            N_to_be_plotted.append(N)

    f, axarr = plt.subplots(nrows=1, ncols=len(N_to_be_plotted), sharex=True, sharey=True)
    for N in neuron_t_weights:
        for lis in N:
            for weight in lis:
                if weight > 1:
                    weight = 1

    print len(N_to_be_plotted)

    for N in range(len(N_to_be_plotted)):
            #plt.close('all')
            axarr[N].imshow(neuron_t_weights[N_to_be_plotted[N]], cmap='hot', interpolation='nearest')
            axarr[N].set_title('Neuron ' + str(N_to_be_plotted[N]))
    plt.show(block=block) 

def extractSpikes(sample):
    organisedData = {}
    spikeTimes = []

    for i in range(len(sample['ts'][0])):
        neuronId = (sample['x'][0][i], sample['y'][0][i])
        if neuronId not in organisedData:
            organisedData[neuronId] = [sample['ts'][0][i]]#/1000]#-sample['ts'][0][0]] # - not necessary in 3indiv_traj files
            # divided by 1000 because AEDAT was in us and SpiNNaker is in ms
            # I don't know how SpyNNkaer cope with 2 spikes at the same ms (different us but how does it know?) 
        else:
            organisedData[neuronId].append(sample['ts'][0][i])#/1000)#-sample['ts'][0][0]) # - not necessary in 3indiv_traj files

    # So that neurons which do not spike still appear:
    for i in range(sample['dim'][0][0]):
        for j in range(sample['dim'][0][1]):
            neuronId = (i,j)
            if neuronId not in organisedData:
                organisedData[neuronId] = []

    print "size organisedData : ", len(organisedData)

    for neuronSpikes in organisedData.values():
        neuronSpikes.sort()
        spikeTimes.append(neuronSpikes)

    print "size spikeTimes : ", len(spikeTimes), len(spikeTimes[0])
    return spikeTimes

def training2ndLayer(sample, untrained_weights=None, endPos_list=None, plot=True, save_w=True): #2nd layer training: STDP

    print "\n \n ---------------------------- TRAINING STARTED ----------------------------"
    print "Training with file ", sample['filename'][0]

    spikeTimes = extractSpikes(sample)

    if untrained_weights == None:
        #untrained_weights = RandomDistribution('uniform', low=wMin, high=wMaxInit).next(inputLayerSize*middleLayerSize)
        untrained_weights = RandomDistribution('normal_clipped', mu=(wMax2-wMin2)/2, sigma=(wMax2-wMin2)/4, low=wMin2, high=wMax2).next(inputLayerSize*middleLayerSize)
        #untrained_weights = np.around(untrained_weights, 3)
        saveWeights(untrained_weights, 'untrained_weights48-2')
    
    if len(untrained_weights)>inputLayerSize:
        training_weights = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                training_weights[i][j] = untrained_weights[k]
                k += 1
    else:
        training_weights = untrained_weights

    for i in range(len(training_weights)):
        for j in range(len(training_weights[i])):
            if training_weights[i][j] == 0:
                training_weights[i][j] == wMin
            if training_weights[i][j] > wMax:
                training_weights[i][j] == wMax

    connections = []
    
    for n_pre in range(inputLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(middleLayerSize): # len(untrained_weight[0]) = middleLayerSize; 0 or any n_pre
            connections.append((n_pre, n_post, training_weights[n_pre][n_post], __delay__)) 
            
    print "size connections", len(connections), len(connections[0])

    runTime = int(max(max(spikeTimes)))+1000
    #####################

    sim.setup(timestep=1)

    timing_rule = sim.SpikePairRule(tau_plus=tauPlus, tau_minus=tauMinus, A_plus=aPlus, A_minus=aMinus)
    #weight_rule = sim.AdditiveWeightDependence(w_min=wMin, w_max=wMax)
    weight_rule = sim.MultiplicativeWeightDependence(w_min=wMin2, w_max=wMax2)

    stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule)#, weight=training_weights, delay=__delay__) # given in connection in fromlistconnector: cannot do weight=trained_weight here

    pre_pop = sim.Population(inputLayerSize, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(middleLayerSize, __neuronType__, __neuronParameters2__, label="post_pop")

    # If random distribution done outside to be able to plot with test, comment next 2 lines:    
    
    stdp_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections), synapse_type=stdp_model)#, weight=training_weights, delay=__delay__) # All to all with diff w
    inhib_proj = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight2, delay=__delay__), receptor_type='inhibitory')
    
    # Lateral inhibition in general: can either clamp the mbrn pot of neighbouring N or inject a negative current
    # here: negative

    post_pop.record(['spikes']) # 'v',
    sim.run(runTime)

    #print("Weights:{}".format(stdp_proj.get('weight', 'list')))
    #print("Lateral inhibition weights:{}".format(inhib_proj.get('weight', 'list')))

    weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)] 
    print "Weights: ", weight_list[0]

    if save_w == True:
        saveWeights(weight_list[1], 'trained_weights48')

    if plot == True:
        plt.close('all')
        neo = post_pop.get_data(["spikes"])#, "v"])
        spikes = neo.segments[0].spiketrains
        #v = neo.segments[0].filter(name='v')[0]

        pplt.Figure(
            #pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0,runTime)),
            pplt.Panel(spikes, xticks=True, xlabel="Time (ms)", yticks=True, markersize=2, xlim=(0,runTime)),
            title="Training with file "+ sample['filename'][0],
            annotations="Simulated with {}".format(sim.name())
        ).save(sample['filename'][0]+'_training48.png')
        #plt.show()

        plot_spiketrains(neo.segments[0])
        plt.title("Training with file "+ sample['filename'][0])
        #plt.show()
        plt.savefig("Training with file "+ sample['filename'][0]+".png")

        #plt.hist(weight_list[1], bins=100)
        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'], range=(0, wMax))
        plt.title('weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()

        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'])
        plt.title('weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()
    sim.end()

    #plt.hist(random_w, bins=100)
    
    return weight_list

def training3rdLayer(sample, trained_weights2=None, untrained_weights3=None, endPos_list=None, plot=True, save_w=True):
    print "\n \n ---------------------------- TRAINING STARTED ----------------------------"
    print "Training with file ", sample['filename'][0]

    spikeTimes = extractSpikes(sample)

    if len(trained_weights2) > inputLayerSize:
        weights2 = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                weights2[i][j] = trained_weights2[k]
                k += 1
    else:
        weights2 = trained_weights2

    connections2 = []

    for n_pre in range(inputLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(middleLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections2.append((n_pre, n_post, weights2[n_pre][n_post], __delay__)) 

    if untrained_weights3 == None:
        untrained_weights3 = RandomDistribution('uniform', low=wMin3, high=wMaxInit3).next(middleLayerSize*outputLayerSize)
        #untrained_weights3 = RandomDistribution('normal_clipped', mu=(wMax-wMin)/2, sigma=(wMax-wMin)/4, low=wMin, high=wMax).next(middleLayerSize*outputLayerSize)
        #untrained_weights = np.around(untrained_weights, 3)
        saveWeights(untrained_weights3, 'untrained_weights3')
    
    if len(untrained_weights3)>middleLayerSize:
        training_weights3 = [[0 for j in range(outputLayerSize)] for i in range(middleLayerSize)] #np array? size 1024x25
        k=0
        for i in range(middleLayerSize):
            for j in range(outputLayerSize):
                training_weights3[i][j] = untrained_weights3[k]
                k += 1
    else:
        training_weights3 = untrained_weights3

    for i in range(len(training_weights3)):
        for j in range(len(training_weights3[i])):
            if training_weights3[i][j] == 0:
                training_weights3[i][j] == wMin3
            if training_weights3[i][j] > wMax3:
                training_weights3[i][j] == wMax3

    connections3 = []
    
    for n_pre in range(middleLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(outputLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections3.append((n_pre, n_post, training_weights3[n_pre][n_post], __delay__)) 
            
    print "size connections", len(connections3)

    runTime = int(max(max(spikeTimes)))+1000
    #####################

    sim.setup(timestep=1)

    timing_rule = sim.SpikePairRule(tau_plus=tauPlus3, tau_minus=tauMinus3, A_plus=aPlus3, A_minus=aMinus3)
    #weight_rule = sim.AdditiveWeightDependence(w_min=wMin, w_max=wMax)
    weight_rule = sim.MultiplicativeWeightDependence(w_min=wMin3, w_max=wMax3)

    stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule)#, weight=training_weights, delay=__delay__) # given in connection in fromlistconnector: cannot do weight=trained_weight here

    pre_pop = sim.Population(inputLayerSize, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(middleLayerSize, __neuronType__, __neuronParameters2__, label="post_pop")
    output_pop = sim.Population(outputLayerSize, __neuronType__, __neuronParameters3__, label="output_pop")

    # If random distribution done outside to be able to plot with test, comment next 2 lines:    
    
    prepost_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections2), synapse_type=sim.StaticSynapse()) # no more learning !!
    stdp_proj = sim.Projection(post_pop, output_pop, sim.FromListConnector(connections3), synapse_type=stdp_model)#, weight=training_weights, delay=__delay__) # All to all with diff w
    inhib_proj3 = sim.Projection(output_pop, output_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight3, delay=__delay__), receptor_type='inhibitory')
    
    # Lateral inhibition in general: can either clamp the mbrn pot of neighbouring N or inject a negative current
    # here: negative

    post_pop.record(['spikes'])
    output_pop.record(['v','spikes']) # 
    sim.run(runTime)

    #print("Weights:{}".format(stdp_proj.get('weight', 'list')))
    #print("Lateral inhibition weights:{}".format(inhib_proj.get('weight', 'list')))

    weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)] 
    print "Weights: ", weight_list[0]

    if save_w == True:
        saveWeights(weight_list[1], 'trained_weights3')

    neo = output_pop.get_data(["spikes", "v"])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]

    post_pop.write_data("middleSpikes_" + sample['filename'][0] + ".mat") 

    if plot == True:
        plt.close('all')
        pplt.Figure(
            pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0,runTime)),
            #pplt.Panel(spikes2, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
            pplt.Panel(spikes, xticks=True, xlabel="Time (ms)", yticks=True, markersize=2, xlim=(0,runTime)),
            title="Training with file "+ sample['filename'][0],
            annotations="Simulated with {}".format(sim.name())
        ).save(sample['filename'][0]+'_training48.png')
        #plt.show()

        plot_spiketrains(neo.segments[0])
        plt.title("Training with file "+ sample['filename'][0])
        #plt.show()
        plt.savefig("Training with file "+ sample['filename'][0]+".png")

        #plt.hist(weight_list[1], bins=100)
        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'], range=(0, wMax3))
        plt.title('weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()

        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'])
        plt.title('weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()
    sim.end()

    #plt.hist(random_w, bins=100)
    
    return weight_list

def training3rdLayerFromMat(middleSpikes, untrained_weights3=None, endPos_list=None, plot=True, save_w=True):
    print "\n \n ---------------------------- TRAINING STARTED ----------------------------"
    #print "Training with file ", sample['filename'][0]

    spikeTimes = []
    #print "middleSpikes['spikes'][0]", middleSpikes['spikes'][0]
    #print "middleSpikes['spikes'][0][1]", middleSpikes['spikes'][0][1]

    for i in range(len(middleSpikes['spikes'][0])):
        if middleSpikes['spikes'][0][i] != []:
            spikeTimes.append(middleSpikes['spikes'][0][i][0].tolist())
        else:
            spikeTimes.append([])

    print len(spikeTimes)
    #print spikeTimes

    if untrained_weights3 == None:
        untrained_weights3 = RandomDistribution('uniform', low=wMin3, high=wMaxInit3).next(middleLayerSize*outputLayerSize)
        #untrained_weights3 = RandomDistribution('normal_clipped', mu=(wMax-wMin)/2, sigma=(wMax-wMin)/4, low=wMin, high=wMax).next(middleLayerSize*outputLayerSize)
        #untrained_weights = np.around(untrained_weights, 3)
        saveWeights(untrained_weights3, 'untrained_weights3endpos-3')
    
    if len(untrained_weights3)>middleLayerSize:
        training_weights3 = [[0 for j in range(outputLayerSize)] for i in range(middleLayerSize)] #np array? size 1024x25
        k=0
        for i in range(middleLayerSize):
            for j in range(outputLayerSize):
                training_weights3[i][j] = untrained_weights3[k]
                k += 1
    else:
        training_weights3 = untrained_weights3

    for i in range(len(training_weights3)):
        for j in range(len(training_weights3[i])):
            if training_weights3[i][j] == 0:
                training_weights3[i][j] == wMin3
            if training_weights3[i][j] > wMax3:
                training_weights3[i][j] == wMax3

    connections3 = []
    
    for n_pre in range(middleLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(outputLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections3.append((n_pre, n_post, training_weights3[n_pre][n_post], __delay__)) 
            
    print "size connections", len(connections3)
    print max(spikeTimes[:])
    print max(max(spikeTimes))
    runTime = int(max(max(spikeTimes)))+301
    #####################

    sim.setup(timestep=1)

    timing_rule = sim.SpikePairRule(tau_plus=tauPlus3, tau_minus=tauMinus3, A_plus=aPlus3, A_minus=aMinus3)
    #weight_rule = sim.AdditiveWeightDependence(w_min=wMin, w_max=wMax)
    weight_rule = sim.MultiplicativeWeightDependence(w_min=wMin3, w_max=wMax3)

    stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule)#, weight=training_weights, delay=__delay__) # given in connection in fromlistconnector: cannot do weight=trained_weight here

    post_pop = sim.Population(middleLayerSize,  sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="post_pop")
    output_pop = sim.Population(outputLayerSize, __neuronType__, __neuronParameters3__, label="output_pop")

    # If random distribution done outside to be able to plot with test, comment next 2 lines:    
    
    stdp_proj = sim.Projection(post_pop, output_pop, sim.FromListConnector(connections3), synapse_type=stdp_model)#, weight=training_weights, delay=__delay__) # All to all with diff w
    inhib_proj3 = sim.Projection(output_pop, output_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight3, delay=__delay__), receptor_type='inhibitory')
    
    # Lateral inhibition in general: can either clamp the mbrn pot of neighbouring N or inject a negative current
    # here: negative

    post_pop.record(['spikes'])
    output_pop.record(['v','spikes']) # 
    sim.run(runTime)

    #print("Weights:{}".format(stdp_proj.get('weight', 'list')))
    #print("Lateral inhibition weights:{}".format(inhib_proj.get('weight', 'list')))

    weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)] 
    print "Weights: ", weight_list[0]

    if save_w == True:
        saveWeights(weight_list[1], 'trained_weights3endpos-3')

    neo = output_pop.get_data(["spikes", "v"])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]

    neo2 = post_pop.get_data(["spikes"])
    spikes2 = neo2.segments[0].spiketrains
    #post_pop.write_data("middleSpikes_" + sample['filename'][0] + ".mat") 

    if plot == True:
        plt.close('all')
        pplt.Figure(
            pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0,runTime)),
            pplt.Panel(spikes2, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
            pplt.Panel(spikes, xticks=True, xlabel="Time (ms)", yticks=True, markersize=2, xlim=(0,runTime)),
            title="Training 3rd layer",
            annotations="Simulated with {}".format(sim.name())
        ).save('3endpos-3rdlayertraining.png')
        plt.show()

        plot_spiketrains(neo.segments[0])
        plt.title("Training 3rd layer")
        #plt.show()
        plt.savefig("3endpos-3rdlayertraining_colour.png")

        #plt.hist(weight_list[1], bins=100)
        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'], range=(0, wMax3))
        plt.title('weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()

        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, label=['neuron 0', 'neuron 1', 'neuron 2'])
        plt.title('weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()
    sim.end()

    #plt.hist(random_w, bins=100)
    
    return weight_list


def test(sample, trained_weights2, trained_weights3, figName=None, plot=True): # 3rd layer training: no more STDP for 2nd layer

    print "\n \n ---------------------------- TEST STARTED ----------------------------"

    # No more learning; keep LI or not?

    spikeTimes = extractSpikes(sample)
    runTime = int(max(max(spikeTimes)))+1000

    if len(trained_weights2) > inputLayerSize:
        weights2 = [[0 for j in range(middleLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(middleLayerSize):
                weights2[i][j] = trained_weights2[k]
                k += 1
    else:
        weights2 = trained_weights2

    connections2 = []

    for n_pre in range(inputLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(middleLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections2.append((n_pre, n_post, weights2[n_pre][n_post], __delay__)) 

    print "trained_weights3 size = ", len(trained_weights3)

    if len(trained_weights3) > middleLayerSize:
        weights3 = [[0 for j in range(outputLayerSize)] for i in range(middleLayerSize)] #np array? size 1024x25
        k=0
        for i in range(middleLayerSize):
            for j in range(outputLayerSize):
                weights3[i][j] = trained_weights3[k]
                k += 1
    else:
        weights3 = trained_weights3

    connections3 = []
    print "weights3 size = ", weights3
    for n_pre in range(middleLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(outputLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections3.append((n_pre, n_post, weights3[n_pre][n_post]*testWeightFactor, __delay__)) 
    ##########################################

    sim.setup(timestep=1)

    pre_pop = sim.Population(inputLayerSize, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(middleLayerSize,  __neuronType__, __neuronParameters2__, label="post_pop")
    output_pop = sim.Population(outputLayerSize,  __neuronType__, __neuronParameters3test__, label="output_pop")
            
    prepost_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections2), synapse_type=sim.StaticSynapse()) # no more learning !!
    postoutput_proj = sim.Projection(post_pop, output_pop, sim.FromListConnector(connections3), synapse_type=sim.StaticSynapse()) # no more learning !!
    # no more lateral inhib
    inhib_proj2 = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight2, delay=__delay__), receptor_type='inhibitory')
    inhib_proj3 = sim.Projection(output_pop, output_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight3, delay=__delay__), receptor_type='inhibitory')
    
    post_pop.record(['spikes'])
    output_pop.record(['v','spikes']) #
    sim.run(runTime)

    neo = output_pop.get_data(["spikes", "v"])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]

    neo2 = post_pop.get_data(["spikes"])
    spikes2 = neo2.segments[0].spiketrains

    print("Weights:{}".format(postoutput_proj.get('weight', 'list')))

    weight_list = [postoutput_proj.get('weight', 'list'), postoutput_proj.get('weight', format='list', with_address=False)] 

    if figName == None:
        figName = sample['filename'][0][0:16]

    if plot == True:
        plt.close('all')
        pplt.Figure(
            # plot voltage 
            pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0, runTime+100)),
            # raster plot
            pplt.Panel(spikes2, xticks=True, yticks=True, markersize=2, xlim=(0, runTime+100)),
            pplt.Panel(spikes, xlabel="Time (ms)", xticks=True, yticks=True, markersize=2, xlim=(0, runTime+100)),
            title='Test with file ' + sample['filename'][0][0:16],
            annotations="Simulated with {}".format(sim.name())
        ).save(figName+'_test48.png')
        plt.show()
        plot_spiketrains(neo.segments[0])
        plt.title('Test with file ' + sample['filename'][0][0:16])
        #plt.show()
        plt.savefig("Test with file "+ sample['filename'][0]+".png")

        plt.hist(weight_list[1], bins=50)
        plt.title(figName + '\n weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()

        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, range=(0, wMax3), label=['neuron 0', 'neuron 1', 'neuron 2'])
        plt.title(figName + '\n weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()
    
    sim.end()


def saveWeights(weights, destFile):
    with open(destFile, 'w') as writeTo:
        for n in weights:
            writeTo.write(str(n) + ' ')  
            #if (n%inputLayerSize == 0):             
                #writeTo.write('\n')

def loadWeights(sourceFile):
    weights = []
    with open(sourceFile) as readFrom:
        lines = readFrom.readlines()
        weights = [] #[] for line in lines])
        for i in range(len(lines)):
            weights = [float(w) for w in lines[i].split()]
    return weights

def unchangedWto0(untrained_weights, trained_weights): # neurons that never spike don't see their synaptic weights modified --> put them to 0
    final_weights = trained_weights
    for i in range(len(untrained_weights)):
        #print roundToPoint5(untrained_weights[i]), trained_weights[i]
        #if roundToPoint5(untrained_weights[i]) == trained_weights[i]:
        if abs(untrained_weights[i]-trained_weights[i]) <= 0.01*wMax3:
            final_weights[i] = 0
    saveWeights(final_weights, "final_weights3endpos-3")
    return final_weights
    
def roundToPoint5(number):
        return round((number * 2) / 2)


# to train the model
training_files = [sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/2-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/4-2_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/2-3_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/4-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/2-2_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/4-3_32x32-3_ON_ms.mat')]

"""training_files = [sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to1-12_4rep_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to3-12_4rep_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to1-12_4rep_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-12_random_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat')]"""

"""training_files3 = [sio.loadmat('endpos3_middleSpikesLI_2-1.mat'),
                    sio.loadmat('endpos3_middleSpikesLI_4-2.mat'),
                    sio.loadmat('endpos3_middleSpikesLI_2-3.mat'),
                    sio.loadmat('endpos3_middleSpikesLI_4-1.mat'),
                    sio.loadmat('endpos3_middleSpikesLI_2-2.mat'),
                    sio.loadmat('endpos3_middleSpikesLI_4-3.mat')]"""
# to evaluate the model
#test_file = sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat') # 2-1 2-3 4-1 4-3
test_file = sio.loadmat('../Data_records/final_mat_max60s/test/from24-3_ON_ms.mat') #

################## TRAINING OUTPUT LAYER ##################
trained_weights2 = loadWeights('final_weights48-3endpos')
untrained_weights2 = loadWeights('untrained_weights48-3endpos')

#middleSpikes = sio.loadmat('middleSpikes_4traj.mat')
middleSpikes = sio.loadmat('endpos3_middleSpikes_LI_from24.mat')
#middleSpikes = sio.loadmat('middleSpikes_2-1_100.mat')

training3rdLayerFromMat(middleSpikes=middleSpikes) 

untrained_weights3 = loadWeights('untrained_weights3endpos-3')
trained_weights3 = loadWeights('trained_weights3endpos-3')
weight_list3 = unchangedWto0(untrained_weights3, trained_weights3)
final_weights3 = loadWeights('final_weights3endpos-3')

weight_list = training3rdLayerFromMat(middleSpikes=middleSpikes, untrained_weights3=weight_list3)
weight_list3 =weight_list[1]
weight_list = training3rdLayerFromMat(middleSpikes=middleSpikes, untrained_weights3=weight_list3)
weight_list3 =weight_list[1]
weight_list = training3rdLayerFromMat(middleSpikes=middleSpikes, untrained_weights3=weight_list3)
weight_list3 =weight_list[1]

untrained_weights3 = loadWeights('untrained_weights3endpos-3')
final_weights3 = loadWeights('final_weights3endpos-3')
test(test_file, trained_weights2, untrained_weights3, plot=True)
test(test_file, trained_weights2, trained_weights3, plot=True)
test(test_file, trained_weights2, final_weights3, plot=True)


################## TESTING ##################


"""
trained_weights3 = loadWeights('trained_weights3')
plt.close('all')
plt.hist(trained_weights3, bins=30, range=(0.05, wMax3))
plt.title('Final weight distribution')
plt.xlabel('Weight value')
plt.ylabel('Weight count')
plt.show()

plt.hist(final_weights3, bins=10)
plt.title('Final weight distribution')
plt.xlabel('Weight value')
plt.ylabel('Weight count')
plt.show()

untrained_weights3 = loadWeights('untrained_weights3')
plt.hist(untrained_weights3, bins=30, range=(0, wMax3))
plt.title('Untrained weight distribution')
plt.xlabel('Weight value')
plt.ylabel('Weight count')
plt.show()"""
