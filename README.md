# Molcule Classfication with Custom Tensorflow Multi Head Graph Attention Layer 
Aids activity classification of molcules of different size with custom tensorflow  masked multi head graph attention layer, custom model and training with custom tensorflow loop 
HIV active molecule , refers to chemical compounds that have been shown to have activity against the Human immunodeficiency Virus (HIV), these molecules can inhibit the replication of the virus , reduce its ability to infect cells , or interface with its lifecycle at various stages.
The dataset , is from a MoleculeNet.org benchmark  HIV data ,  it consists of 40426 HIV inactive molecules , and 1487 HIV active molecules , a very imbalanced data .

Each sample consists of SMILE representation of a molecule, to implement our Graph Attention Analysis on data , first we have to create the graph representation of each molecule , this is possible with rdkit python library . with this representation each atom in the molecule viewed at as a node, and each bond viewed at as edge between two nodes. Key thing to consider that there are three types of bonds, Single, Double and Aromatic , we use an edge weights [1,2,1.5] to represent strengths of these bond types .so each molecule will represented as a weighted graph . 
These weights will help our model to consider the information of bond types , helping it to better understand Atoms . 

Now that we have the graph representations, we must understand our data better. 
First step in dealing with molecular data is to know statistics of count of atoms of each molecule. 
With that we realize that the molecules have different sizes.  so, the first parameter we have to set is the maximum size of atoms in molecules (very similar to the ML tasks on text data), with help of statistics available in the code, we choose 44.  
Also, we do extract statistics of other structural Features like betweenness centrality of each node(atom), number of simple cycles in each molecule, degree of each node, most important number of atoms in each molecule. 
As mentioned previously, we are dealing with graphs (molecules) of different sizes, which means they have adjacency matrices of different sizes, which is a problem for parallel computations of TensorFlow models. so, we pad all these adjacency matrices to be in size (44, 44). 
Using padding in learning algorithms means we need a mask to mask the outputs of the model, so we create a suitable graph mask for each graph. 
The GNN models overall are powerful tools in graph machine learning, but they have some weaknesses in understanding some structural features of graphs. To help these models overcome this issue, usually we do graph Augmentation technique, in which we add some manually extracted structural features to node features . in this project we add node degree and betweenness centrality of each node . 

To apply Graph attention network , there are pre-implemented GAT layers in python libraries like PyGeometric , However , I implemented the Multi head GAT layer in TensorFlow as a custom TensorFlow layer class from scratch , Based on mathematics provided in Velickovic GAT paper .
Model :
In NLP tasks, when dealing with sentences, first we compute or assign an embedding to each word , as said before with this technique , we transform the text to a set feature vectors connected to  each other in a 1d structure . in deep NLP models this is usually done with an embedding layer in the model that assigns a embedding to each word type . 
Similar to NLP , when dealing with molecules as graphs , we can compute an embedding for each atom type with an embedding layer, then our molecule can be viewed as multi-dimensional arrangement of these vectors . 
Then this is the time we add structural node features to each node embedding to do the graph augmentation. 
In the next step we apply our custom  multi head GAT layers on  node embeddings matrix.
The main idea behind most of GNNs as GAT layer is aggregation of information from one-hop neighbors with message passing in each layer.
So based on the insights from structures of molecules , I chose three layers of GAT , so each node shares its information with all the nodes in the graph . 
The final output of this three layers, is a matrix with final embedding representation vector for each node. but as  we are doing graph classification we need to compute a overall vector representation for the graph . to do this , first we have to mask the padding node outputs then we do the sum aggregation of final node embeddings . 
Sum Aggregation can be beneficial , as it considers number of atoms in the molecule , this important , as hiv active and inactive  sample molecules in our dataset, have a different size distribution . 
Then We apply dense layers to the classification . 
To better implement and have better control  in  all the mentioned steps in the model , I gather all these layers in a custom tensorflow model class inherited from tf.Model class . 
Then i create an instance of this class (building the model ).


Training and test :
First we split the train and test data with 0.1 percent of samples of each class  as test .
As explained before our dataset is highly imbalanced and the data is complex , to overcome this issue , we do oversampling on the smaller class training samples. 
For having a better control and flexibility on training process instead of using tensorflow model.fit( ) , I created a custom training loop , and applied a the gradients in training with tf.gradient tape on learnable parameters of the model . 
The result was 71% Recall on smaller class , and 73% Recall on larger class . 

Future work:
First, the results of this classification task can be improved , by changes in parameters , number of layers  and better augmentations. 
Also in the near future I will implement graph transformer model in this data and compare it with GAT and GCN model results . 
Lastly , This dataset is very suitable for Drug discovery and graph generation tasks , and  my plan is to implement famous algorithm of the paper , “GraphRNN :Generating realistic graphs with deep auto regressive models “ by Jure Leskovec .






 

