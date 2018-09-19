import json
import os

if not os.path.exists('./parameters'):
    os.mkdir('./parameters')

# paramPath = './parameters/p{0}_{1}_{2}_{3}_{4}_{5}_{6}.json'
paramPath = './parameters/p{0}_{1}_{2}_{3}_{4}_{5}.json'

nSteps = [1, 3, 6, 12, 24, 48]
nNeurons = [100, 150, 200]
nLayers = [1, 2, 3, 4, 5]
leaning_rates = [0.01, 0.001]
# nEpochs = [3]
batchSizes = [50, 100, 200]
trainingKeepProbs = [0.5, 0.6, 0.7, 0.8, 0.9]


for nStep in nSteps:
    for nNeuron in nNeurons:
        for nLayer in nLayers:
            for leaning_rate in leaning_rates:
                # for nEpoch in nEpochs:
                for batchSize in batchSizes:
                    for trainingKeepProb in trainingKeepProbs:
                        # writingPath = paramPath.format(nStep, nNeuron, nLayer, leaning_rate, nEpoch, batchSize, trainingKeepProb)
                        # data = {'n_steps': nStep, 'n_neurons': nNeuron, 'n_layers': nLayer, 'lr': leaning_rate, 'n_epochs': nEpoch, 'bs': batchSize, 'training_keep_prob': trainingKeepProb}
                        writingPath = paramPath.format(nStep, nNeuron, nLayer, leaning_rate, batchSize, trainingKeepProb)
                        data = {'n_steps': nStep, 'n_neurons': nNeuron, 'n_layers': nLayer, 'lr': leaning_rate, 'bs': batchSize, 'training_keep_prob': trainingKeepProb}
                        with open(writingPath, 'w') as file:
                            json.dump(data, file)
