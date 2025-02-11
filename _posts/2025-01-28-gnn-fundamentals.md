---
layout: post
comments: true
title: Graph Neural Networks Fundamentals
author: Shaodong Wang
---

Graph Neural Networks (GNNs) have become a fundamental tool in machine learning for processing data structured as graphs. Unlike traditional deep learning models, which operate on grid-like data such as images or sequences, GNNs excel at learning representations from relational data, where entities (nodes) are interconnected by relationships (edges). These networks have found applications in various domains, including social network analysis, molecular biology, recommendation systems, and knowledge graphs. This learning note explores the key concepts, methodologies, challenges, and advancements in GNNs, providing a comprehensive overview of their mechanisms and improvements.

## Representing Nodes, Edges and Graphs
### Node Representation
Nodes in a graph require a meaningful numerical representation that captures their attributes and relationships. The simplest approach is one-hot encoding, where each node is represented as a unique binary vector. Another approach is using node degree, which considers the number of connections a node has. However, these methods often fail to capture the rich contextual relationships present in graphs.

More advanced techniques involve learning node embeddings using random walks. Algorithms such as **DeepWalk** and **Node2Vec** generate sequences of nodes by performing random walks on the graph and then apply techniques similar to word embeddings (e.g., Word2Vec) to learn vector representations. These methods encode structural information about a node’s neighborhood, allowing the model to generalize better.

### Edge Representations
The simplest way to represent an edge is to concatenate or aggregate the two representations of its two nodes.

We can also learn edge embeddings, where a separate feature vector is assigned to each edge and trained alongside node embeddings. This is particularly useful in tasks requiring edge-specific information, such as link prediction and relational reasoning. 

### Graph Representation
To represent an entire graph or subgraph as a vector, one straightforward method is to compute embeddings for individual nodes and aggregate them using operations such as summation, averaging, or max pooling.

Another approach introduces a **super-node**, which is a virtual node connected to all nodes in the subgraph. This super-node is then embedded to represent the graph structure.

A more sophisticated technique is **Anonymous Walk Embeddings**, which involve capturing structural patterns in a graph by performing random walks without considering specific node identities. These walks can be counted to create a frequency distribution, or their embeddings can be concatenated to form a fixed-size representation of the graph.

## Fundamental GNN Architectures
### Graph Convolutional Networks (GCN)
Graph Convolutional Networks (GCN) generalize the concept of convolution from CNNs to graphs. Instead of applying filters over regular grid structures, GCNs aggregate information from a node’s neighbors using a weighted sum. The key idea is to propagate and transform information across the graph while maintaining structural dependencies.

### GraphSAGE (Sample and Aggregate)
GraphSAGE improves upon GCN by introducing a sampling strategy to handle large-scale graphs. Instead of using the entire adjacency matrix, GraphSAGE samples a fixed number of neighbors for each node and aggregates their features using operations such as mean, LSTM-based aggregation, or pooling. This makes the model more scalable and robust to changes in graph structure.

### Graph Attention Networks (GAT)
Graph Attention Networks (GAT) enhance traditional GNNs by incorporating an attention mechanism. Instead of treating all neighbors equally, GAT assigns different importance scores to each neighbor based on learned attention coefficients. This allows the network to focus more on influential nodes while reducing the impact of irrelevant ones.

## Over-Smoothing Problem in Deep GNNs
When we design our graphs we keep the limitation of the GNN in mind, the over-smoothing problem, where node embeddings become indistinguishable as the number of layers increases. This occurs because deeper networks aggregate information from increasingly distant neighbors, leading to highly similar representations across all nodes.

Several strategies mitigate over-smoothing:

- **Shallow but Expressive GNNs:** Instead of stacking many GNN layers, deeper feature transformations can be performed within each layer using Multi-Layer Perceptrons (MLPs).
- **Skip Connections:** Introducing residual connections allows earlier node embeddings to influence the final representation, preventing excessive smoothing.
- **Regularization Techniques:** Applying dropout or normalization methods helps maintain diversity in embeddings.

## Graph Augmentation
Not all graphs are ideal for GNN models, and graph augmentation can be an effective technique to enhance the performance. Augmentation strategies generally fall into two categories: feature augmentation and structural augmentation.

### Graph Feature Augmentation
In cases where the input graph lacks sufficient features, feature augmentation can provide meaningful enhancements. One simple approach is to assign constant values or unique identifiers to nodes, allowing GNNs to extract useful structural information from the graph. Assigning constant values is inductive and generalizable to unseen or new nodes, making it a practical solution.

Additional feature-based enhancements include incorporating statistical properties of nodes, such as node degree, clustering coefficient, PageRank scores (which indicate node importance), and centrality measures. These attributes provide GNNs with more discriminative information, leading to better performance in various graph-based tasks.

### Graph Structure Augmentation
#### Sparse Graphs
When a graph is too sparse, adding virtual edges can improve connectivity and message passing. For example, connecting two-hop neighbors via virtual edges can create more informative neighborhood structures. Mathematically, this can be represented as modifying the adjacency matrix from A to A + A², effectively linking nodes that previously had only indirect relationships. In real-world applications, such as author-paper networks, these virtual edges may represent potential collaborations.

Another approach for sparse graphs is to introduce virtual nodes. A common strategy is to add a parent node that connects all other nodes, serving as an intermediary that facilitates more efficient information flow across the network.

#### Dense Graphs
For graphs that are too dense, performing message passing on all neighbors can introduce redundancy and computational inefficiency. A viable solution is to sample neighbors dynamically when executing message passing. By selecting different subsets of neighbors in different GNN layers, the model can improve robustness and focus on the most informative connections while reducing redundancy.

#### Large Graphs
When dealing with exceptionally large graphs, processing the entire structure can be computationally prohibitive. One effective strategy is subgraph sampling, where smaller, localized subgraphs are extracted to compute embeddings. This method enables scalable training and inference while preserving the essential structural properties of the original graph.

By employing these augmentation strategies, GNN models can be more effective in learning representations from graphs, even when the underlying data structure presents challenges such as sparsity, excessive density, or large-scale complexity.

## Supervised Learning for Graph Neural Networks (GNNs)

Supervised learning in GNNs involves training the model using labeled data, where nodes, edges, or entire graphs are assigned specific labels. Depending on the task, predictions can be made at different levels: node-level, edge-level, or graph-level.

### Node Prediction

Node classification aims to assign a label  y_v  to each node  v  in the graph. This is typically done by applying a final transformation to the learned node representation after multiple GNN layers:

$$
y_v = 	head_{node}(h_v^L) = W^H h_v^L
$$

where $$h_v^L$$  is the final node embedding after  L  GNN layers, $$W^H$$  is a learnable weight matrix that maps node embeddings to label probabilities.

### Edge Prediction

Edge prediction (or link prediction) involves determining the likelihood of an edge existing between two nodes  u  and  v . Several approaches exist:

A simple approach is to **concatenate** the embeddings of both nodes and pass them through a linear layer:

$$
y_{uv} = 	Linear({Concat}(h_u^L, h_v^L))
$$

where $$h_u^L$$ and $$h_v^L$$ are the final embeddings of nodes  u  and  v, respectively.

Alternatively, a **dot product** can be used to compute a similarity score between two node embeddings:

$$
y_{uv} = (h_u^L)^T h_v^L
$$

This method effectively captures similarity-based link formation but may not be flexible enough for complex relationships.

For multi-class edge prediction, a **weighted dot product** is applied where different weights are learned for each class k:

$$
y_{uv}^k = (h_u^L)^T W^k h_v^L
$$

where $$W^k$$ is a learnable weight matrix for class  k .

### Graph Prediction
For tasks requiring graph-level classification, the node embeddings must be aggregated into a single representation.

A straightforward method is to apply mean, max, or sum pooling over all node embeddings, followed by a Multi-Layer Perceptron (MLP):

$$
y_G = 	MLP(Pooling(\{h_v^L | v \in G\}))
$$

However, this approach may lose significant structural information.

A better way involves hierarchical aggregation, where node embeddings are clustered before graph-level representation is computed. This can be done using:

- **Community detection algorithms** to define hierarchical groupings.
- **Graph partitioning methods** to identify meaningful substructures.
- **DiffPool (Differentiable Pooling)**, a model that learns hierarchical graph structure during training. DiffPool consists of two GNNs:
  - **GNN A** computes node embeddings.
  - **GNN B** determines node clusters for hierarchical pooling.

### Best Practices for Supervised Learning in GNNs

When working with node, edge, or graph labels, it is often beneficial to reduce the problem to a simpler form. If clusters of nodes exist, treating the cluster assignment as a node label can simplify the task. Similarly, multi-hop relationships between nodes can sometimes be converted into an edge classification problem instead of learning higher-order graph structures directly.


## Self-Supervised Learning for GNN

One of the primary challenges in training GNNs arises when there are no external labels available for nodes, edges, or entire graphs. In such cases, **self-supervised learning** provides a compelling solution by leveraging the **intrinsic structure** of graphs to generate supervision signals. 

### Node-Level Tasks  
At the node level, GNNs can be trained to predict node-specific statistics. These statistics include **clustering coefficients** (cluster id), and **PageRank scores**, which reflect the relative importance of nodes within a graph. By predicting such properties, the model learns to capture node significance and local structures.

### Edge-Level Tasks
The **link prediction** is a typical edge-level task. This involves hiding existing edges during training and asking the model to predict whether a link should exist between two nodes. This technique is useful in scenarios such as **social network analysis**, **recommendation systems**, and **biological network modeling**, where understanding connections between entities is crucial.

### Graph-Level Tasks
At the graph level, GNNs can learn to analyze **entire graph characteristics**. One example is determining whether two graphs are **isomorphic**, meaning they have the same structural composition despite differences in labeling. We can adjust the graphs in the dataset to create positive(isomorphic) graph pairs, while the other graph pairs are negatives. Learning such representations enables the model to generalize across different graphs, making it useful for tasks like **molecular property prediction** and **chemical compound classification**.

The key advantage of **self-supervised learning** in GNNs is that it eliminates the dependency on externally provided labels, making the learning process more scalable and adaptable across various domains. Since the model relies solely on internal graph structures, it can be effectively applied to large-scale networks where manual labeling is infeasible. 

