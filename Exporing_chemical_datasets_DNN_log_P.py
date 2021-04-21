#Copyright 2017 PandeLab

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import tensorflow as tf
import numpy as np
import deepchem as dc
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import TensorGraph
from deepchem.models.tensorgraph.layers import L2Loss
from deepchem.models.tensorgraph.layers import Feature, GraphConv, BatchNorm, GraphPool, Dense, Dropout, GraphGather, \
  SoftMax, \
  Label, SoftMaxCrossEntropy, Weights, WeightedError, Stack
import csv

#Reads data and converts data to ConvMol objects

def read_data(input_file_path):
    featurizer = dc.feat.ConvMolFeaturizer(use_chirality=True)
    loader = dc.data.CSVLoader(tasks=prediction_tasks, smiles_field="SMILES", featurizer=featurizer)
    dataset = loader.featurize(input_file_path, shard_size=8192)
    # Initialize transformers
    transformer = dc.trans.NormalizationTransformer(transform_w=True, dataset=dataset)
    print("About to transform data")
    dataset = transformer.transform(dataset)

    return dataset

#Generation of batches from the data as input in network

def default_generator(dataset,
                      epochs=1,
                      batch_size=50,
                      predict=False,
                      deterministic=True,
                      pad_batches=True,
                      labels=None,
                      weights=None,
                      atom_features=None,
                      degree_slice=None,
                      membership=None,
                      deg_adjs=None):
  for epoch in range(epochs):
    print('Epoch number:', epoch)  
    for ind, (X_b, y_b, w_b, ids_b) in enumerate(
        dataset.iterbatches(
          batch_size,
          pad_batches=pad_batches,
          deterministic=deterministic)):
      d = {}
      
      for i, label in enumerate(labels):
        d[label] =  np.expand_dims(y_b[:, i],1) 
      d[weights] = w_b
      multiConvMol = ConvMol.agglomerate_mols(X_b)
      d[atom_features] = multiConvMol.get_atom_features()
      d[degree_slice] = multiConvMol.deg_slice
      d[membership] = multiConvMol.membership
      for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
        d[deg_adjs[i - 1]] = multiConvMol.get_deg_adjacency_lists()[i]
      yield d

#Used for loading of trained model

def layer_reference(model, layer):
  if isinstance(layer, list):
    return [model.layers[x.name] for x in layer]
  return model.layers[layer.name]

#Get array for the calculation of rms and r² for dataset
def reshape_y_pred(y_true, y_pred):
 
  n_samples = len(y_true)
  retval = np.vstack(y_pred)
  return retval[:n_samples]

#Define working directory

model_dir = working_directory

#Define prediction task and directory of .csv file
#csv file should include SMILES and the prediction task

prediction_tasks = ['logP']
train_dataset, test_dataset = read_data(data_directory)

#Define model, batch size, learning rate

model = TensorGraph(tensorboard=True,
                    batch_size=50,
                    learning_rate=0.0005,
                    use_queue=False,
                    model_dir=model_dir)
#Placeholders for the chemical structures
#Chirality is used; if chirality should not be used: atom_features = Feature(shape=(None, 75))
atom_features = Feature(shape=(None, 78))
degree_slice = Feature(shape=(None, 2), dtype=tf.int32)
membership = Feature(shape=(None,), dtype=tf.int32)
deg_adjs = []

#Define structure of neural network 

for i in range(0, 10 + 1):
  deg_adj = Feature(shape=(None, i + 1), dtype=tf.int32)
  deg_adjs.append(deg_adj)
#Input 
in_layer = atom_features
#Layer 1
for layer_size in [64, 64]:
  gc1_in = [in_layer, degree_slice, membership] + deg_adjs
  gc1 = GraphConv(layer_size, activation_fn=tf.nn.relu, in_layers=gc1_in)
  batch_norm1 = BatchNorm(in_layers=[gc1])
  gp_in = [batch_norm1, degree_slice, membership] + deg_adjs
  in_layer1 = GraphPool(in_layers=gp_in)
#Layer 2
for layer_size in [128, 128]:
  gc2_in = [in_layer1, degree_slice, membership] + deg_adjs
  gc2 = GraphConv(layer_size, activation_fn=tf.nn.relu, in_layers=gc2_in)
  batch_norm2 = BatchNorm(in_layers=[gc2])
  gp2_in = [batch_norm2, degree_slice, membership] + deg_adjs
  in_layer2 = GraphPool(in_layers=gp2_in)
#Layer 3
for layer_size in [256, 256]:
  gc3_in = [in_layer2, degree_slice, membership] + deg_adjs
  gc3 = GraphConv(layer_size, activation_fn=tf.nn.relu, in_layers=gc3_in)
  batch_norm3 = BatchNorm(in_layers=[gc3])
  gp3_in = [batch_norm3, degree_slice, membership] + deg_adjs
  in_layer3 = GraphPool(in_layers=gp3_in)
dense = Dense(out_channels=256, activation_fn=tf.nn.relu, in_layers=[in_layer3])
batch_norm3 = BatchNorm(in_layers=[dense])
batch_norm3 = Dropout(0.1, in_layers=[batch_norm3])
readout = GraphGather(
  batch_size=50,
  activation_fn=tf.nn.tanh,
  in_layers=[batch_norm3, degree_slice, membership] + deg_adjs)
costs = []
labels = []
regression = Dense( out_channels=1, activation_fn=None, in_layers=[readout])
model.add_output(regression)
label = Label(shape=(None, 1))
labels.append(label)
cost = L2Loss(in_layers=[label, regression])
costs.append(cost)
all_cost = Stack(in_layers=costs, axis=1)
weights = Weights(shape=(None, len(prediction_tasks)))
loss = WeightedError(in_layers=[all_cost, weights])
model.set_loss(loss)

#Define Training of neural network and the number of epochs for training
gene = default_generator(train_dataset, epochs=130, predict=True, pad_batches=True,
                         labels=labels, weights=weights, atom_features=atom_features, degree_slice=degree_slice,
                         membership=membership, deg_adjs=deg_adjs)
model.fit_generator(gene)
model.save()

#Predict with model
#Load the model, which was previously trained
model2 = TensorGraph.load_from_dir(model.model_dir)  
gene = default_generator(train_dataset, epochs=1, predict=True, pad_batches=True,
                         labels=labels, weights=weights, atom_features=atom_features, degree_slice=degree_slice,
                         membership=membership, deg_adjs=deg_adjs)
labels = layer_reference(model2, labels)
weights = layer_reference(model2, weights)
atom_features = layer_reference(model2, atom_features)
degree_slice = layer_reference(model2, degree_slice)
membership = layer_reference(model2, membership)
deg_adjs = layer_reference(model2, deg_adjs)
#Load the train_dataset for evaluation 
gene = default_generator(train_dataset, epochs=1, predict=True, pad_batches=True,
                         labels=labels, weights=weights, atom_features=atom_features, degree_slice=degree_slice,
                         membership=membership, deg_adjs=deg_adjs)
#Load the test_dataset for evaluation
gene1 = default_generator(test_dataset, epochs=1, predict=True, pad_batches=True,
                         labels=labels, weights=weights, atom_features=atom_features, degree_slice=degree_slice,
                         membership=membership, deg_adjs=deg_adjs)

metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean,  mode = "regression")
rms=dc.metrics.Metric(dc.metrics.rms_score,np.mean,mode='regression')

#Check performance on the training-dataset

print("Evaluating on train data")
train_predictions = model2.predict_on_generator(gene)
train_predictions = reshape_y_pred(train_dataset.y,train_predictions)
train_scores = metric.compute_metric(train_dataset.y, train_predictions,train_dataset.w)
train_scores2 = rms.compute_metric(train_dataset.y, train_predictions, train_dataset.w)
print("Train r²: %f" % train_scores)
print("Train rms: %f" % train_scores2)

#Check performance on the test_dataset

print("Evaluating on test data")
test_predictions = model2.predict_on_generator(gene1)
test_predictions = reshape_y_pred(test_dataset.y, test_predictions)
test_scores = metric.compute_metric(test_dataset.y, test_predictions, test_dataset.w)
test_scores2 = rms.compute_metric(test_dataset.y, test_predictions, test_dataset.w)
print("Test r²: %f" % test_scores)
print("Test rms: %f" % test_scores2)

