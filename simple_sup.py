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
import matplotlib.colors as colors

__neuronType__ = sim.IF_curr_exp 

__neuronParameters__ = { # tau_m = Rm * cm --> Rm typically 100 Mohm = 10^8 ohm
                         # Rm typically: 10000 ohm/cm2
                         # v_thresh >> v_rest + tau_syn ?
                         # tau_m: independent of geometry (no /cm2) --> 10 ms
    'cm': 70,#2,  # (nF) Capacitance of the LIF neuron: membrane capacity - 1.0 
                         # typically: 1nF/cm2
                         # typical physiological value C = 0.29 nF
    'tau_m': 20.0, #110.0,  # (ms) RC circuit time-constant: membrane time constant - 20.0 # don't exceed 300ms (traj separated by 300ms) otherwise spikes from previous traj will be taken into account
    'tau_refrac': 20.0, # 20 40.0, # (ms) Duration of refractory period- 0.0 # typically 5 ms : after a spike, membrane potential is clamped to v_reset for a refractory period tau_refrac
    'v_reset': -70.0 , # (mV) Reset potential after a spike: voltage at which neuron is reset (typically lower than v_rest) - -65
    'v_rest': -65.0,  # (mV) Resting mebrane potential: ambient rest voltage of the neuron - -64
    'v_thresh': -60, #-50, # (mV) Spike threshold: threshold voltage at which the neuron spikes - -50
    'tau_syn_E': 2.0, #5.0,  # (ms) Rise time of the excitatory synaptic alpha function: excitatory input current decay time-constant - 5 # 2.728ms
    'tau_syn_I': 25.0, #10.0, # (ms) Rise time of the inhibitory synaptic alpha function: inhibitory input current decay time-constant - 5
    'i_offset': 0.0  # (nA) Offset current: base input current to add at each timestep - 
}

# Higher membrane capacitance prevents neuron from spiking in answer to noise and too much time per pattern per pattern
# The lower Cm is, the fewer input spikes are needed to determine the output spike timing, therefore Cm should not be too high, 
# otherwise there will be no spikes at all. On the other hand, too low Cm would cause superfluous spikes.

# STDP parameters
__delay__ = 1.0 # (ms) 
tauPlus = 30 #20 # 15 # 16.8 from literature
tauMinus = 30 #20 # 30 # 33.7 from literature
aPlus = 0.500  #tum 0.016 #9 #3 #0.5 # 0.03 from literature
aMinus = 0.500 #255 #tum 0.012 #2.55 #2.55 #05 #0.5 # 0.0255 (=0.03*0.85) from literature 
wMax = 10.000 #1 # G: 0.15
wMaxInit = 10.000#0.1#0.100
wMin = 0
nbIter = 5
testWeightFactor = 1#0.05177
x = 3 # no supervision for x first traj presentations
y = 0# for inside testing of traj to see if it has been learned /!\ stdp not disabled

inputLayerSize = 1024 # 32*32
outputLayerSize = 2
inhibWeight = 10 #20 # same as wMax ? 5 --> 1/3 of threshold ? between 3 (1/5) and 15 (1/1) ? --> paper masquelier: 1/4 of threshold
stimWeight = 100


def unchangedWto0(untrained_weights, trained_weights): # neurons that never spike don't see their synaptic weights modified --> put them to 0
    final_weights = trained_weights
    for i in range(len(untrained_weights)):
        #print roundToPoint5(untrained_weights[i]), trained_weights[i]
        #if roundToPoint5(untrained_weights[i]) == trained_weights[i]:
        if abs(untrained_weights[i]-trained_weights[i]) <= 0.01*wMax:
            final_weights[i] = 0
    saveWeights(final_weights, "final_weightssupmodel")
    return final_weights

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
        u_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                u_weights[i][j] = untrained_weights[k]/wMax
                k += 1
    else:
        u_weights = untrained_weights

    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
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
        u_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                u_weights[i][j] = untrained_weights[k]/wMax
                k += 1
    else:
        u_weights = untrained_weights

    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                t_weights[i][j] = trained_weights[k]*wMax/max(trained_weights)
                k += 1
    else:
        t_weights = trained_weights

    #neuron_u_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(outputLayerSize)]
    #neuron_t_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(outputLayerSize)]

    neuron_u_weights = np.zeros((outputLayerSize, int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))
    neuron_t_weights = np.zeros((outputLayerSize, int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))

    for j in range(outputLayerSize):
        for k in range(int(math.sqrt(inputLayerSize))):
            for l in range(int(math.sqrt(inputLayerSize))):
                neuron_u_weights[j][k][l] = u_weights[k+l*int(math.sqrt(inputLayerSize))][j]
                neuron_t_weights[j][k][l] = t_weights[k+l*int(math.sqrt(inputLayerSize))][j]

    f, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    """for N in neuron_t_weights:
        for lis in N:
            for weight in lis:
                if weight > 1:
                    weight = 1"""

    N=0
    #plt.close('all')

    a = axarr[0][0].imshow(neuron_u_weights[N], norm=colors.Normalize(vmin=0,vmax=1), cmap='YlOrRd')#, interpolation='nearest')
    plt.colorbar(a)
    b = axarr[0][1].imshow(neuron_t_weights[N], norm=colors.Normalize(vmin=0,vmax=1), cmap='YlOrRd')#, interpolation='nearest')
    plt.colorbar(b)
    c = axarr[1][0].imshow(neuron_u_weights[N+1], norm=colors.Normalize(vmin=0,vmax=1), cmap='YlOrRd')#, interpolation='nearest')
    plt.colorbar(c)
    d = axarr[1][1].imshow(neuron_t_weights[N+1], norm=colors.Normalize(vmin=0,vmax=1), cmap='YlOrRd')#, interpolation='nearest')
    plt.colorbar(d)
    axarr[0][0].set_title('Before Training, neuron ' + str(N))
    axarr[0][1].set_title('After Training, neuron ' + str(N))
    axarr[1][0].set_title('Before Training, neuron ' + str(N+1))
    axarr[1][1].set_title('After Training, neuron ' + str(N+1))
    

    plt.show(block=block)

def plot_1neuron_weight_heatmap2(untrained_weights, trained_weights, neuron, block=True):
    plt.close('all')
    if len(untrained_weights)>inputLayerSize:
        u_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                u_weights[i][j] = untrained_weights[k]/wMax
                k += 1
    else:
        u_weights = untrained_weights

    if len(trained_weights)>inputLayerSize:
        t_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                t_weights[i][j] = trained_weights[k]/wMax
                k += 1
    else:
        t_weights = trained_weights

    #neuron_u_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(outputLayerSize)]
    #neuron_t_weights = [[[0 for l in range(int(math.sqrt(inputLayerSize)))] for k in range(int(math.sqrt(inputLayerSize)))] for j in range(outputLayerSize)]

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
        t_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                t_weights[i][j] = trained_weights[k]/wMax
                k += 1
    else:
        t_weights = trained_weights
    
    neuron_t_weights = np.zeros((outputLayerSize, int(math.sqrt(inputLayerSize)), int(math.sqrt(inputLayerSize))))

    for j in range(outputLayerSize):
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
    
def training(sample, untrained_weights=None, endPos_list=None, plot=True, save_w=True): #2nd layer training: STDP
    organisedStim = {}
    stimSpikes = []

    print " ---------------------------- TRAINING STARTED ----------------------------"
    print "Training with file ", sample['filename'][0]

    spikeTimes = extractSpikes(sample)
   
    # So that neurons which do not spike still appear:
    
    if endPos_list == None: # If endPos is list in sample == several traj 
        for i in range(len(sample['endPos_list'][0])-y-x): # -y: y last presentations to see if it has learned
            neuronNr = 0#sample['endPos_list'][0][i]-1 # neuron 0, 1, 2 whereas pos 1, 2, 3
            if neuronNr == 2: #for 2 endpos: 1 and 3
                neuronNr = 1
            if neuronNr not in organisedStim:
                organisedStim[neuronNr] = [3000*(i+1+x)+300*(i+x)] # 3001, 6301... # 1 maybe not necessary because of delay
            else:
                organisedStim[neuronNr].append(3000*(i+1+x)+300*(i+x))

        for neuronNr in range(outputLayerSize):
            if neuronNr not in organisedStim:
                organisedStim[neuronNr] = []
        
        for neuronSpikes in organisedStim.values():
            neuronSpikes.sort()
            stimSpikes.append(neuronSpikes)
    else: # If endPos is int == 1 traj
        for i in range(outputLayerSize):
            stimSpikes.append([])
        stimSpikes[int(sample['filename'][0][-2])] = [int(max(max(spikeTimes)))+1] # endPos = 1-2-3 but neuron nr = 0-1-2"""

    print "size stimSpikes : ", len(stimSpikes),  len(stimSpikes[0]), " = ", stimSpikes

    if untrained_weights == None:
        untrained_weights = RandomDistribution('uniform', low=wMin, high=wMaxInit).next(inputLayerSize*outputLayerSize)
        #untrained_weights = RandomDistribution('normal_clipped', mu=0.1, sigma=0.05, low=wMin, high=wMaxInit).next(inputLayerSize*outputLayerSize)
        untrained_weights = np.around(untrained_weights, 3)
        saveWeights(untrained_weights, 'untrained_weightssupmodel1traj')
    
    print "length untrained_weights :", len(untrained_weights)

    if len(untrained_weights)>inputLayerSize:
        training_weights = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                training_weights[i][j] = untrained_weights[k]
                k += 1
    else:
        training_weights = untrained_weights

    connections = []
    
    for n_pre in range(inputLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(outputLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections.append((n_pre, n_post, training_weights[n_pre][n_post], __delay__)) 
            
    print "size connections", len(connections), len(connections[0])

    runTime = int(max(max(spikeTimes)))+1000
    #####################

    sim.setup(timestep=1)

    timing_rule = sim.SpikePairRule(tau_plus=tauPlus, tau_minus=tauMinus, A_plus=aPlus, A_minus=aMinus)
    weight_rule = sim.MultiplicativeWeightDependence(w_min=wMin, w_max=wMax)

    stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule)#, weight=training_weights, delay=__delay__) # given in connection in fromlistconnector: cannot do weight=trained_weight here

    pre_pop = sim.Population(inputLayerSize, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(outputLayerSize, __neuronType__, __neuronParameters__, label="post_pop")
    stim_pop= sim.Population(outputLayerSize, sim.SpikeSourceArray, {'spike_times': stimSpikes}, label="stim_pop")

    # If random distribution done outside to be able to plot with test, comment next 2 lines:    
    
    stdp_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections), synapse_type=stdp_model)#, weight=training_weights, delay=__delay__) # All to all with diff w
    inhib_proj = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight, delay=__delay__), receptor_type='inhibitory')
    stim_proj = sim.Projection(stim_pop, post_pop, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=stimWeight, delay=__delay__)) # put weight to 1 ? to wMax?
    
    # Lateral inhibition in general: can either clamp the mbrn pot of neighbouring N or inject a negative current
    # here: negative

    post_pop.record(['v', 'spikes'])
    stim_pop.record(['spikes'])
    sim.run(runTime)

    print("Weights:{}".format(stdp_proj.get('weight', 'list')))

    weight_list = [stdp_proj.get('weight', 'list'), stdp_proj.get('weight', format='list', with_address=False)] 
    
    if save_w == True:
        saveWeights(weight_list[1], 'trained_weightssupmodel1traj')

    if plot == True:
        plt.close('all')
        neo = post_pop.get_data(["spikes", "v"])
        spikes = neo.segments[0].spiketrains
        v = neo.segments[0].filter(name='v')[0]

        neostim = stim_pop.get_data(["spikes"])
        spikestim = neostim.segments[0].spiketrains

        pplt.Figure(
            pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0,runTime)),
            pplt.Panel(spikestim, xticks=True, yticks=True, markersize=2, xlim=(0,runTime)),
            pplt.Panel(spikes, xticks=True, xlabel="Time (ms)", yticks=True, markersize=2, xlim=(0,runTime)),
            title="Training with file "+ sample['filename'][0],
            annotations="Simulated with {}".format(sim.name())
        ).save(sample['filename'][0]+'_training.png')
        plt.show()

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

def test(sample, trained_weights, figName=None, plot=True): # 3rd layer training: no more STDP for 2nd layer

    print " ---------------------------- TEST STARTED ----------------------------"

    # No more learning; keep LI or not?

    spikeTimes = extractSpikes(sample)

    runTime = int(max(max(spikeTimes)))+1000

    ##########################################

    sim.setup(timestep=1)

    pre_pop = sim.Population(inputLayerSize, sim.SpikeSourceArray, {'spike_times': spikeTimes}, label="pre_pop")
    post_pop = sim.Population(outputLayerSize,  __neuronType__, __neuronParameters__, label="post_pop")
   
    if len(trained_weights) > inputLayerSize:
        weigths = [[0 for j in range(outputLayerSize)] for i in range(inputLayerSize)] #np array? size 1024x25
        k=0
        for i in range(inputLayerSize):
            for j in range(outputLayerSize):
                weigths[i][j] = trained_weights[k]
                k += 1
    else:
        weigths = trained_weights

    connections = []
    
    #k = 0
    for n_pre in range(inputLayerSize): # len(untrained_weights) = inputLayerSize
        for n_post in range(outputLayerSize): # len(untrained_weight[0]) = outputLayerSize; 0 or any n_pre
            connections.append((n_pre, n_post, weigths[n_pre][n_post]*(wMax)/max(trained_weights), __delay__)) #
            #k += 1

    prepost_proj = sim.Projection(pre_pop, post_pop, sim.FromListConnector(connections), synapse_type=sim.StaticSynapse(), receptor_type='excitatory') # no more learning !!
    #inhib_proj = sim.Projection(post_pop, post_pop, sim.AllToAllConnector(), synapse_type=sim.StaticSynapse(weight=inhibWeight, delay=__delay__), receptor_type='inhibitory')
    # no more lateral inhib

    post_pop.record(['v', 'spikes'])
    sim.run(runTime)

    neo = post_pop.get_data(['v', 'spikes'])
    spikes = neo.segments[0].spiketrains
    v = neo.segments[0].filter(name='v')[0]

    print("Weights:{}".format(prepost_proj.get('weight', 'list')))

    weight_list = [prepost_proj.get('weight', 'list'), prepost_proj.get('weight', format='list', with_address=False)] 

    if figName == None:
        figName = sample['filename'][0][0:16]

    if plot == True:
        plt.close('all')
        pplt.Figure(
            # plot voltage 
            pplt.Panel(v, ylabel="Membrane potential (mV)", xticks=True, yticks=True, xlim=(0, runTime+100)),
            # raster plot
            pplt.Panel(spikes, xlabel="Time (ms)", xticks=True, yticks=True, markersize=2, xlim=(0, runTime+100)),
            title='Test with file ' + sample['filename'][0][0:16],
            annotations="Simulated with {}".format(sim.name())
        ).save(figName+'_test.png')
        plt.show()

        plt.hist(weight_list[1], bins=50)
        plt.title(figName + '\n weight distribution')
        plt.xlabel('Weight value')
        plt.ylabel('Weight count')
        #plt.show()

        plt.hist([weight_list[1][0:1024], weight_list[1][1024:2048], weight_list[1][2048:]], bins=20, range=(0, wMax), label=['neuron 0', 'neuron 1', 'neuron 2'])
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

    connections = []
    with open(sourceFile) as readFrom:
        lines = readFrom.readlines()
        connections = [[] for line in lines]
        for i in range(len(lines)):
            connections = [w for w in lines[i].split()]
    print connections
    return connections

# to train the model
training_files = [sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/2-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/2-3_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/4-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/4-3_32x32-3_ON_ms.mat')]

training_files9 = [sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/1-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/1-2_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/1-3_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-2_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-3_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-1_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-2_32x32-3_ON_ms.mat'),
                  sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-3_32x32-3_ON_ms.mat')]

# to evaluate the model
test_file = sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat') # 2-1 2-3 4-1 4-3
test_file9 = sio.loadmat('../Data_records/final_mat_max60s/test/from135-3_ON_ms.mat') # 1-1 3-1 5-1 1-2 3-2 5-2 1-3 3-3 5-3

test_otherstart=[sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/1-3_32x32-1_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-1_32x32-2_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-3_32x32-1_ON_ms.mat'), 
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-3_32x32-2_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/1-1_32x32-3_ON_ms.mat'),
                 sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/1-1_32x32-1_ON_ms.mat')]
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-3_32x32-1_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/3-3_32x32-2_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-1_32x32-1_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-1_32x32-2_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-3_32x32-1_ON_ms.mat'),
                 #sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/5-3_32x32-2_ON_ms.mat')]

test_halftraj=[sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_2-1_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_4-1_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_4-1_32x32-2.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_4-1_32x32-1.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_2-3_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_4-3_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_1-1_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_1-3_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_3-1_32x32-1.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_3-1_32x32-2.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_3-1_32x32-3.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_3-3_32x32-2.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_5-1_32x32-2.mat'),
               sio.loadmat('../Data_records/15mat_files/3indiv_traj/half_traj/half_5-3_32x32-3.mat')]

test_filefrom135 = sio.loadmat('../Data_records/final_mat_max60s/test/from135-3_ON_ms.mat')
test_filefrom24 = sio.loadmat('../Data_records/final_mat_max60s/test/from24-3_ON_ms.mat')

test_otherrec = sio.loadmat('../Data_records/final_mat_max60s/training/indiv_files_18rep/2-1_32x32-1_ON_ms.mat')

test_file = sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to13-3_ON_ms.mat') # 2-1 2-3 4-1 4-3
test_half = sio.loadmat('all_halftrajto13.mat')
test_all = sio.loadmat('alltraj_3s_ON_ms.mat')


#test_fileto3 = sio.loadmat('../Data_records/final_mat_max60s/4traj/from24to3-12_4rep_ON_ms.mat')
################## TRAINING OUTPUT LAYER ##################
#weight_list = RandomDistribution('uniform', low=wMin, high=wMaxInit).next(inputLayerSize*outputLayerSize)

#w = training(training_files[0], plot=True)
weight_list = loadWeights('trained_weightssupmodel1traj')
"""for i in range(len(training_files)-1):
    trained_weights = training(training_files[i+1], untrained_weights=weight_list, plot=False)
    weight_list = trained_weights[1]
for i in range(len(training_files)):
    trained_weights = training(training_files[i], untrained_weights=weight_list, plot=False)
    weight_list = trained_weights[1]
for i in range(len(training_files)):
    trained_weights = training(training_files[i], untrained_weights=weight_list, plot=False)
    weight_list = trained_weights[1]
"""

uw = loadWeights('untrained_weightssupmodel1traj')
tw = loadWeights('trained_weightssupmodel1traj')
fw = unchangedWto0(uw, tw)

#plot_neuron_weight_heatmap2(uw, fw, block=True)

#print max(tw)
#test(training_files[0], uw, plot=True)
#test(training_files[0], tw, plot=True)
#test(test_otherrec, tw, plot=True)
#test(test_file, tw, plot=True)




"""

#w = loadWeights('trained_weights2layermodel')
#test(test_indiv2, w, plot=True)
#test(test_indiv, w, plot=True)

#weight_list = trained_weights[1]
#trained_weights = training(training_files[0], untrained_weights=weight_list, plot=True)
#weight_list = trained_weights[1]
trained_weights = training(training_files[1], untrained_weights=weight_list, plot=True)
weight_list = trained_weights[1]
trained_weights = training(training_files[2], untrained_weights=weight_list, plot=True)
weight_list = trained_weights[1]
trained_weights = training(training_files[3], untrained_weights=weight_list, plot=True)
weight_list = trained_weights[1]"""
"""trained_weights = training(training_files[0], plot=False)
weight_list = trained_weights[1]
for i in range(len(training_files)-1):
    trained_weights = training(training_files[i+1], untrained_weights=weight_list, plot=False)
    weight_list = trained_weights[1]

for it in range(nbIter-1):
    for i in range(len(training_files)):
        trained_weights = training(training_files[i], untrained_weights=weight_list, plot=False)
        weight_list = trained_weights[1]
"""


################## TESTING ##################
#weight_list = loadWeights('trained_weights2layermodel')
#test(test_file, weight_list, '2-layer model: After training')