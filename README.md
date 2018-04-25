# master-project

Spiking Neural Network to be run with SpiNNaker to detect the direction of motion and predict the end position of a moving ball

Moving ball simulation: Matlab & Simulink (see folder Matlab_simulation)

Moving ball recordings done with a DVS camera and jAER (.aedat format)


preProcessing.py : set of functions to read and use Address-Event Representation (AER) data in python

spikingNN.py : the neural network architecture that is passed to SpiNNaker

NNcontroller.py : to run the simulation (training and evaluation of NN model)


Requirements:
- Python 2.7 (numpy, matplotlib, scipy, lxml)
- SpiNNaker board
- PyNN for SpiNNaker (sPyNNaker8)
- jAER (requires JAVA)
