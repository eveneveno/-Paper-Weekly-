### [Space-Time Correspondence as a Contrastive Random Walk](https://arxiv.org/pdf/2006.14613.pdf)
Allan Jabri, Andrew Owens, Alexei A. Efros
(UC Berkeley, University of Michigan)

**Aim:**
> Learn a similarity representation for correspondence from unlabeled video.

**Focus**
> Simplicity & Scalability

#### Palindrome Sequence
<p align="center">
<img src="img/palindrome.png" alt="drawing" width="600"/>
</p>

#### Video as a Graph
<p align="center">
<img src="img/graph.png" alt="drawing" width="150"/>
<img src="img/transition.png" alt="drawing" width="170"/>
</p>

##### One Step (Contrastive Learning)
<p align="center">
<img src="img/one-step.png" alt="drawing" width="150"/>
<img src="img/sphere.png" alt="drawing" width="150"/>
</p>

Maximize
$P\left(X_{t+1}=\operatorname{pos} \mid X_{t}=q u e r y\right)=\frac{e^{\phi(q u e r y)^{\top} \phi(p o s)}}{\sum_{q_{l}} e^{\phi(q u e r y)^{\top} \phi\left(q_{l}\right)}}$
$k$-step Transition Matrix
$\bar{A}_{t}^{t+k}=\prod_{i=0}^{k-1} A_{t+i}^{t+i+1}=P\left(X_{t+k} \mid X_{t}\right)$

##### Chaining Correspondences
<p align="center">
<img src="img/chaining.png" alt="drawing" width="200"/>
</p>

##### Supervised -> Self-Supervised 
<p align="center">
<img src="img/cycle.png" alt="drawing" width="200"/>
</p>

Train on Palindromes: $\mathcal{L}_{c y c}^{k}=\mathcal{L}_{C E}\left(\bar{A}_{t}^{t+k} \bar{A}_{t+k}^{t}, I\right)$

#### Implementation
<p align="center">
<img src="img/implement.png" alt="drawing" width="220"/>
<img src="img/algorithm.png" alt="drawing" width="220"/>
</p>

#### Training Details
* Trained on the unlabeled Kinetics-400 dataset (Kay et al. 2017)
* Adam optimizer, $lr = 0.0001$
* One million weight updates
* 3x GPUs for 24-sequence batches
  * Can train with 1 GPU, 8-sequence batches

#### Evalutaion (use $\phi$ for Label Propagation)

**Label Propagation**
Weighted average of labels from K nearest neighbors
<p align="center">
<img src="img/knn.png" alt="drawing" width="180"/>
</p>

##### "Common Fate" by Edge Dropout
