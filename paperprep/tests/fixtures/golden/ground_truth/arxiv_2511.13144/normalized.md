# PERSONALIZED FEDERATED LEARNING WITH BIDIRECTIONAL COMMUNICATION COMPRESSION VIA ONE-BIT RANDOM SKETCHING

Anonymous Submission

Anonymous Institute

# ABSTRACT

Federated Learning (FL) enables collaborative training across decentralized data, but faces key challenges of bidirectional communication overhead and client-side data heterogeneity. To address communication costs while embracing data heterogeneity, we propose pFed1BS, a novel personalized federated learning framework that achieves extreme communication compression through one-bit random sketching. In personalized FL, the goal shifts from training a single global model to creating tailored models for each client. In our framework, clients transmit highly compressed one-bit sketches, and the server aggregates and broadcasts a global one-bit consensus. To enable effective personalization, we introduce a sign-based regularizer that guides local models to align with the global consensus while preserving local data characteristics. To mitigate the computational burden of random sketching, we employ the Fast Hadamard Transform for efficient projection. Theoretical analysis guarantees that our algorithm converges to a stationary neighborhood of the global potential function. Numerical simulations demonstrate that pFed1BS substantially reduces communication costs while achieving competitive performance compared to advanced communication-efficient FL algorithms.

# INTRODUCTION

Federated Learning (FL) is an increasingly popular paradigm in deep learning, designed to train machine learning models on distributed client data while preserving privacy  $(\ref{eq:1})$ . Although FL performs well under ideal conditions of i.i.d. data and unconstrained communication, real-world deployments face two fundamental challenges inherent to its decentralized nature.

First, data across clients is typically non-i.i.d., reflecting diverse user behaviors and environments, which can severely degrade the performance of a single global model. Second, the communication overhead is often prohibitive, as repeatedly transmitting high-dimensional models between a central server and numerous clients is infeasible in bandwidth-limited networks. These challenges are particularly acute in critical application domains such as the massive Internet of Things (IoT), Vehicle-to-Everything (V2X) communications, and remote sensing networks. In these settings, where

devices operate under extremely constrained bandwidth, efficient communication is not merely an optimization but a fundamental necessity for the system to be viable.

To overcome the challenge of non-i.i.d. data, Personalized Federated Learning (PFL)  $(\ref{eq:1})$  has been proposed. Here, it is crucial to distinguish the goal of "personalization" from merely "addressing data heterogeneity." While the latter often aims to improve a single global model, PFL explicitly seeks to produce bespoke, customized models for individual clients that capture the unique characteristics of their local data. However, most PFL methods still incur significant communication costs by transmitting full-precision, high-dimensional model parameters or updates. This leads to a more precise and challenging research question: can we design a federated learning algorithm that not only provides high-quality personalized models for each client but also operates efficiently under extreme bidirectional communication constraints?

To reduce this burden, communication-efficient FL (CEFL) techniques have been developed, employing methods like prototype-learning (?), sparsification (??) and quantization (??). Among the most aggressive are one-bit compression strategies. For instance, OBDA (?) applies symmetric one-bit quantization for bidirectional communication, OBCSAA (?) combines a one-bit compressed sensing uplink with an uncompressed downlink, and zSignFed (?) stabilizes sign-based compression through a noisy perturbation scheme.

While these methods achieve remarkable compression rates, they primarily focus on training a single global model and thus overlook the critical challenge of data heterogeneity. To systematically analyze the limitations of existing work and motivate our contribution, we provide a comparison of representative algorithms in Table 1. The table reveals a clear research gap: no existing framework offers both extreme, bidirectional communication efficiency and native support for personalization.

In this paper, we bridge this gap by introducing pFed1BS, a novel PFL framework designed for extreme communication constraints based on one-bit random sketching. We formulate a joint optimization problem where clients learn personalized models by minimizing a local loss augmented with a sign-based regularizer. This regularizer encourages alignment with a global consensus vector derived by the server.

Table 1: Comparisons of communication-efficient schemes and personalization Capabilities in Federated Learning Algorithms.

Critically, our framework achieves bidirectional compression: clients upload one-bit sketches of their local models, and the server broadcasts a compact one-bit consensus vector. This contrasts sharply with prior CEFL methods that either compress only the uplink or require a full-model downlink. Figure 1 provides an overview of our proposed framework.

# MAIN CONTRIBUTIONS

This paper proposes pFed1BS, as shown in Figure 1, a personalized federated learning framework designed for settings with extreme communication constraints. Our main contributions are as follows:

# RELATED WORKS

Our work is closely related to the following topics:

Personalized Federated Learning. To address the challenge of heterogeneous datasets, a rich body of work has emerged in Personalized Federated Learning (PFL). These approaches can be broadly categorized into local adaptation (????), multi-task learning (??), and architecture-based methods (??). For instance, regularization-based methods like pFedMe (?) and Ditto (?) learn personalized models by augmenting the local objective with a proximal term that regularizes it towards a global model. Architecture-based methods, such as FedRep (?) learn a shared feature representation while personalizing the final model layers. More recently, DisPFL (?) learns personalized sparse masks for each client. However, these advanced PFL methods typically inherit the communication bottlenecks of standard FL, as they still presuppose the transmission of full-precision, high-dimensional model parameters or updates.

Communication-Efficient Federated Learning. In a parallel research thrust, numerous methods have been proposed to alleviate the communication burden in FL. One prominent approach is update sparsification, where only a fraction of the model update is transmitted, using techniques like Top-k selection (?) or identifying parameters with high-magnitude changes (?) Another major direction is quantization, which reduces the numerical precision of the transmitted updates (????). A critical limitation of these popular techniques, however, is that they typically compress only the uplink (client-to-server) channel, still requiring the server to broadcast a full-precision, high-dimensional model.

More aggressive strategies leverage techniques from signal processing. Some works employ Compressed Sensing (CS) to project sparse updates into a low-dimensional subspace  $(\ref{eq:1})$ . Others explore one-bit quantization schemes, sometimes combined with over-the-air computation in wireless settings, to achieve extreme compression  $(\ref{eq:2})$ . While highly efficient, these methods are fundamentally designed to learn a single global model and lack mechanisms to handle data heterogeneity.

pFed1BS uniquely bridges these two disparate lines of research. It is the first work to integrate a bidirectional, one-bit sketching mechanism within a principled PFL formulation, thereby explicitly and simultaneously tackling the dual challenges of communication efficiency and data heterogeneity.

![](images/95c834764fb94b091c04ad4351ce55e2be8b040aeacc27d9a2dfa7cb14f807a2.jpg)

# THE PROPOSED METHOD

This section presents our proposed method, pFed1BS. We begin by formulating the overall optimization framework that governs the collaborative learning process by introducing a sign-based regularizer and random sketching. Subsequently, we describe the iterative algorithm, specifying distinct client-server procedures for optimizing this objective. Finally, to address the computational bottleneck posed by the high-dimensional random sketching in local training, we introduce an efficient sketching method based on the Fast Hadamard Transform.

# OPTIMIZATION FRAMEWORK

In a federated learning system with  $K$  clients, our goal is to move beyond learning a single global model and instead learn a personalized model  $\boldsymbol{w}_k \in \mathbb{R}^n$  for each client  $k \in \{1, \dots, K\}$ . The system aggregates client contributions using weights  $p_k$ , which are typically set based on the client's dataset size, e.g.,  $p_k = N_k / \sum_{i=1}^{K} N_i$ , where  $N_k$  is the number of local samples. The local objective for each client is to minimize the expected loss over its private data distribution  $\mathcal{P}_k$

$$

$$
f _ {k} (\boldsymbol {w} _ {k}) = \mathbb {E} _ {\xi_ {k} \sim \mathcal {P} _ {k}} [ \hat {f} _ {k} (\boldsymbol {w} _ {k}; \xi_ {k}) ], \tag {1}
$$

$$

where  $\xi_{k}$  is a data sample drawn from  $\mathcal{P}_k$  and  $\hat{f}_k(\pmb {w}_k;\xi_k)$  is the loss function for a single sample  $\xi_{k}$ . In practice, the true gradient  $\nabla f_{k}(\pmb{w}_{k})$  is intractable, so we rely on stochastic gradients computed on mini-batches to approximate it.

To drastically reduce communication costs, our core technical idea is to replace the transmission of high-dimensional

models  $\pmb{w}_k$  with low-dimensional one-bit sketches. Specifically, each client transmits only  $\mathrm{sign}(\Phi \pmb{w}_k)$ , where  $\Phi \in \mathbb{R}^{m\times n}$  is a random projection matrix.

This radical compression necessitates a new way for the server and clients to interact. To this end, we introduce a global consensus vector  $\pmb{v} \in \{\pm 1\}^m$ , which the server aggregates from the clients' one-bit sketches. This vector  $\pmb{v}$  is then broadcast back to clients and incorporated into their local optimization, which acts as a target, guiding the local model  $\pmb{w}_k$  to produce a projection  $\Phi \pmb{w}_k$  whose signs align with  $\pmb{v}$ .

To enforce this guidance, we introduce a sign-based regularizer, denoted by  $g(\pmb{v}, \pmb{\Phi}\pmb{w}_k)$ , which measures the disagreement between the signs of the projected model  $\pmb{\Phi}\pmb{w}_k$  and the global consensus  $\pmb{v}$ . We define this regularizer using the one-sided  $\ell_1$ -norm:

$$

$$
g (\boldsymbol {x}, \boldsymbol {y}) = \| [ \boldsymbol {x} \odot \boldsymbol {y} ] _ {-} \| _ {1}, \text {w h e r e} [ x _ {i} ] _ {-} = \min  (x _ {i}, 0). \tag {2}
$$

$$

For the client-side objective, where  $\pmb{v}$  is a sign vector and  $\Phi \pmb{w}_k$  is real-valued, this regularizer is equivalent to:

$$

$$
g (\boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k}) = \frac {1}{2} \left(\| \boldsymbol {\Phi} \boldsymbol {w} _ {k} \| _ {1} - \langle \boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} \rangle\right). \tag {3}
$$

$$

Furthermore, to prevent the personalized models  $\boldsymbol{w}_k$  from diverging or growing unbounded during local training due to the absence of global data constraints, we introduce an  $\ell_2$  penalty term.

This leads to the following regularized client objective:

$$

$$
F _ {k} \left(\boldsymbol {w} _ {k}; \boldsymbol {v}\right) = f _ {k} \left(\boldsymbol {w} _ {k}\right) + \lambda g (\boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k}) + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2}, \tag {4}
$$

$$

where  $\lambda$  controls the strength of the sign alignment and  $\mu$  controls the norm of model parameters.

This form, however, is non-smooth due to the  $\ell_1$ -norm in (3), posing a challenge for gradient-based optimization. To enable gradient-based optimization, we employ a continuously differentiable approximation for the  $\ell_1$ -norm. A standard approach is to approximate  $\|z\|_1$  with  $h_\gamma(z)$ , where  $h_\gamma(z) = \frac{1}{\gamma} \sum_{i=1}^m \log(\cosh(\gamma z_i))$ . So we obtain a smoothed regularizer

$$

$$
\tilde {g} (\boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k}) = h _ {\gamma} (\boldsymbol {\Phi} \boldsymbol {w} _ {k}) - \langle \boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} \rangle , \tag {5}
$$

$$

and a smoothed client-side objective  $\tilde{F}_k(\pmb {w}_k;\pmb {v})$

$$

$$
\begin{array}{l} \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = f _ {k} (\boldsymbol {w} _ {k}) + \lambda \tilde {g} (\boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k}) + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2} \\ = f _ {k} \left(\boldsymbol {w} _ {k}\right) + \lambda \left(h _ {\gamma} \left(\boldsymbol {\Phi} \boldsymbol {w} _ {k}\right) - \langle \boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} \rangle\right) + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2}, \tag {6} \\ \end{array}
$$

$$

where the factor of  $\frac{1}{2}$  can be absorbed into the hyperparameter  $\lambda$ .

The gradient of the smoothed penalty term with respect to  $\boldsymbol{w}_k$  is then given by:

$$

$$
\nabla \tilde {g} (\boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k}) = \boldsymbol {\Phi} ^ {\top} (\tanh  (\gamma \boldsymbol {\Phi} \boldsymbol {w} _ {k}) - \boldsymbol {v}). \tag {7}
$$

$$

As the smoothing parameter  $\gamma \rightarrow \infty$ , the term  $\tanh (\gamma \Phi w_k) \approx \mathrm{sign}(\Phi w_k)$ . Therefore, the gradient effectively penalizes the misalignment between the signs of the projected local model and the global consensus vector  $\mathbf{v}$ , driving the local updates towards alignment.

Having defined how each client utilizes the global vector  $\pmb{v}$ , we now address how the server generates it. At the end of each round, the server receives a one-bit sketch  $z_{k} = \mathrm{sign}(\Phi \pmb{w}_{k})$  from each participating client  $k$ . The server's task is to aggregate these sketches into a new consensus vector  $\pmb{v}^{t + 1}$  that best represents the collective information.

We formulate this as an optimization problem where the server seeks to find a vector  $\boldsymbol{v} \in \{\pm 1\}^m$  that minimizes the total weighted disagreement with the received client sketches:

$$

$$
\min  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \sum_ {k = 1} ^ {K} p _ {k} \tilde {g} (\boldsymbol {v}, \boldsymbol {z} _ {k}). \tag {8}
$$

$$

Overall, we formulate a bilevel optimization problem as follows

$$

$$
\text {S e r v e r :} \quad \min  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \sum_ {k = 1} ^ {K} p _ {k} \tilde {g} (\boldsymbol {v}, \operatorname {s i g n} (\boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {\star} (\boldsymbol {v})))
$$

$$

$$

$$
\begin{array}{l} \text {C l i e n t s :} \quad \boldsymbol {w} _ {k} ^ {\star} (\boldsymbol {v}) \in \arg \min  _ {\boldsymbol {w} _ {k} \in \mathbb {R} ^ {n}} F _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = f _ {k} (\boldsymbol {w} _ {k}) \\ + \lambda \tilde {g} (\boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k}) + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2}, \tag {9} \\ \end{array}
$$

$$

where  $\pmb{w}_k^{\star}(\pmb{v})$  denotes the optimal solution of the  $k$ -th low-level problem for a given upper-level variable  $\pmb{v}$ .

# ALGORITHM

To solve the joint optimization problem defined in Eq. (9), we propose an alternating optimization scheme named pFed1BS. The core idea is to iteratively perform (i) local optimization on the client side to update the personalized models  $\boldsymbol{w}_k$  and (ii) global aggregation on the server side

Algorithm 1: pFed1BS: Personalized Federated Learning via One-Bit Random Sketching

to update the consensus vector  $\pmb{v}$ . The overall procedure is presented in Algorithm 1. We describe the specifics of each component below.

At the start of round  $t$ , each participating client  $k \in S^t$  receives the global consensus vector  $\pmb{v}^t$ . The client's goal is to update its local model  $\pmb{w}_k^t$  by approximately minimizing its smoothed objective  $\tilde{F}_k(\pmb{w}_k; \pmb{v}^t)$  (from Eq. (6)). As shown in Line 16 of Algorithm 1, the client performs  $R$  steps of stochastic gradient descent. For local step  $r$ , the update is

$$

$$
\boldsymbol {w} _ {k, r + 1} = \boldsymbol {w} _ {k, r} - \eta \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r}; \boldsymbol {v} ^ {t}\right). \tag {10}
$$

$$

The gradient  $\nabla \tilde{F}_k$  is composed of the standard task gradient and our regularization terms:

$$

$$
\begin{array}{l} \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t}; \boldsymbol {v} ^ {t}\right) = \nabla f _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t}\right) + \lambda \boldsymbol {\Phi} ^ {\top} \left[ \tanh  \left(\gamma \boldsymbol {\Phi} \boldsymbol {w} _ {k, r} ^ {t}\right) - \boldsymbol {v} ^ {t} \right] \\ + \mu \boldsymbol {w} _ {k, r} ^ {t}. \tag {11} \\ \end{array}
$$

$$

Since the calculation of  $\nabla f_{k}(\boldsymbol{w}_{k,r}^{t})$  is intractable, we use the mean over a mini-batch of data  $\mathcal{B}_{k,r}$  from  $\mathcal{P}_k$

$$

$$
\nabla \hat {f} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}; \mathcal {B} _ {k, r}\right) = \frac {1}{\left| \mathcal {B} _ {k , r} \right|} \sum_ {\xi_ {k} \in \mathcal {B} _ {k, r}} \nabla \hat {f} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}; \xi_ {k}\right). \tag {12}
$$

$$

After  $R$  steps, the client transmits its new one-bit sketch  $\pmb{z}_k^{t + 1} = \mathrm{sign}(\Phi \pmb{w}_k^{t + 1})$  to the server. Upon receiving the sketches  $\{\pmb{z}_k^{t + 1}\}_{k\in S^t}$ , the server generates the next consensus vector  $\pmb{v}^{t + 1}$ . As defined in Eq. (8), the server aims to solve:

$$

$$
\min  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \sum_ {k \in \mathcal {S} ^ {t}} p _ {k} g (\boldsymbol {v}, \boldsymbol {z} _ {k} ^ {t + 1}). \tag {13}
$$

$$

Crucially, this discrete optimization problem admits an exact, closed-form solution. The following lemma states that the optimal aggregation is a simple weighted majority vote.

Lemma 1 (Optimal Server Aggregation). The unique minimizer of the server objective in Eq. (13) is given by:

$$

$$
\boldsymbol {v} ^ {*} = \operatorname {s i g n} \left(\sum_ {k \in S ^ {t}} p _ {k} \boldsymbol {z} _ {k} ^ {t + 1}\right). \tag {14}
$$

$$

This result is a straightforward but significant application of optimization principles, ensuring that our aggregation step (Line 8 in Algorithm 1) is not a heuristic but is guaranteed to be optimal given the information available to the server.

# EFFICIENT PROJECTION VIA FAST HADAMARD TRANSFORM

A naive implementation of the projection  $\Phi w$  using a dense Gaussian matrix requires  $\mathcal{O}(mn)$  computation and memory, which causes a computational burden for large models ( $n \gg 10^6$ ). To ensure scalability, we employ a structured projection based on the Subsampled Randomized Hadamard Transform (SRHT) (?), which reduces the complexity to  $\mathcal{O}(n \log n)$ .

![](images/c0afbf2282840cc6972f5442f38031fa2d6fdd132210410d38b4b3088f5471e1.jpg)

![](images/44541c300128107567b5e892d7e30a420876ae28068e786bbf1598b170e68256.jpg)

Forward Projection. The forward projection  $z = \Phi w \in \mathbb{R}^m$  is computed through a sequence of efficient operations,

as illustrated in Figure 2 (b) (Up). Let  $n$  denote the model dimension and  $m \ll n$  be the target dimension. First, the input vector  $\boldsymbol{w} \in \mathbb{R}^n$  is zero-padded to the next power-of-two dimension,  $n' = 2^{\lceil \log_2 n \rceil}$ . This can be represented as:

$$

$$
\tilde {\boldsymbol {w}} = \boldsymbol {P} _ {\text {p a d}} (\boldsymbol {w}) = \left[ \begin{array}{l} \boldsymbol {w} \\ \boldsymbol {0} \end{array} \right], \tag {15}
$$

$$

where  $P_{\mathrm{pad}}$  can be viewed as an  $n' \times n$  matrix consisting of the identity matrix  $I_n$  stacked on top of an  $(n' - n) \times n$  zero matrix, and  $\tilde{\boldsymbol{w}} \in \mathbb{R}^{n'}$ . This step ensures compatibility with the Fast Hadamard Transform (FHT), which requires input lengths that are powers of 2.

Then the structured random projection is performed as follows

$$

$$
\Phi \boldsymbol {w} = S ^ {\prime} H D \tilde {\boldsymbol {w}}, \tag {16}
$$

$$

where  $D \in \mathbb{R}^{n' \times n'}$  is a diagonal matrix where each diagonal entry  $D_{ii}$  is an independent random variable drawn uniformly from  $\{-1, +1\}$ ,  $H \in \mathbb{R}^{n' \times n'}$  is the normalized Walsh-Hadamard matrix, and  $S' = \sqrt{\frac{n'}{m}} S \in \mathbb{R}^{m \times n'}$ , where  $S$  is a subsampling matrix that uniformly selects  $m$  rows at random from the  $n'$ -dimensional identity matrix. This structured operator avoids forming any dense matrices. The total computational complexity is dominated by the FHT, resulting in an efficient  $\mathcal{O}(n \log n)$  computation.

Backward Computation. In the backward pass, we compute the adjoint projection  $\Phi^{\top} \pmb{v}$ , where  $\pmb{v} \in \mathbb{R}^{m}$  denotes a low-dimensional one-bit vector.

Specifically, we first lift  $\pmb{v} \in \mathbb{R}^m$  to  $\tilde{\pmb{v}} \in \mathbb{R}^{n'}$  by zeropadding as follows

$$

$$
\tilde {\boldsymbol {v}} = \boldsymbol {S} ^ {\prime \top} \boldsymbol {v}. \tag {17}
$$

$$

This operation places the elements of  $\boldsymbol{v}$  into the coordinates that were selected by  $S$ , filling the rest with zeros. We then apply the FHT to the padded vector, and apply the same random sign flips as in the forward pass as  $DH\tilde{v}$ .

Finally, we truncate the resulting  $n^{\prime}$  -dimensional vector back to the original dimensions. This is the action of  $\pmb{P}_{\mathrm{pad}}^{\top}$ . The operator  $\pmb{P}_{\mathrm{pad}}^{\top}:\mathbb{R}^{n^{\prime}}\to \mathbb{R}^{n}$  is a truncation operator, which we denote as  $\pmb{P}_{\mathrm{trunc}}$ . Its action is to select the first  $n$  coordinates of a vector in  $\mathbb{R}^{n^{\prime}}$  and discard the remaining  $n^{\prime}-n$  coordinates.

The complete operation is formulated as

$$

$$
\boldsymbol {\Phi} ^ {\top} \boldsymbol {v} = \left(\boldsymbol {S} ^ {\prime} \boldsymbol {H} \boldsymbol {D} \boldsymbol {P} _ {\text {p a d}}\right) ^ {\top} \boldsymbol {v} = \boldsymbol {P} _ {\text {p a d}} ^ {\top} \boldsymbol {D} ^ {\top} \boldsymbol {H} ^ {\top} \boldsymbol {S} ^ {\prime \top} \boldsymbol {v}. \tag {18}
$$

$$

Given that  $P_{\mathrm{trunc}} = P_{\mathrm{pad}}^{\top}$  and  $D$  is diagonal, this simplifies to  $\Phi^{\top}v = P_{\mathrm{trunc}}DH^{\top}S^{\prime \top}v$ .

Similar to the forward projection, the adjoint projection is matrix-free and has a computational complexity of  $\mathcal{O}(n\log n)$ , making our algorithm practical and efficient.

# THEORETICAL ANALYSIS

We provide a rigorous convergence guarantee for pFed1BS, showing that the algorithm converges to a neighborhood of a stationary point.

# ASSUMPTIONS AND PRELIMINARIES

Our analysis relies on a set of standard assumptions common in FL literature.

Assumption 1 (L-smoothness). The local objective function  $f_{k}(\cdot)$  is L-smooth, i.e., its gradient is L-Lipschitz continuous.

Assumption 2 (Bounded Below). The global potential function  $\Psi(\pmb{w}_1, \ldots, \pmb{w}_K; \pmb{v}) = \sum_{k=1}^{K} p_k \tilde{F}_k(\pmb{w}_k; \pmb{v})$  is bounded below by some value  $F^*$ .

Assumption 3 (Bounded Gradient Variance). The variance of the stochastic gradients is uniformly bounded, i.e., for any client  $k$  and model  $\mathbf{w}$ ,  $\mathbb{E}_{\mathcal{B}}\left[\| \nabla \hat{f}_k(\mathbf{w};\mathcal{B}) - \nabla f_k(\mathbf{w})\|^2\right] \leq \sigma^2$  for some constant  $\sigma^2 \geq 0$ .

Assumption 4 (Bounded Task Gradient Variance). The stochastic gradient of the task loss  $f_{k}$  has a bounded second moment, i.e., there exists a constant  $G > 0$  such that  $\mathbb{E}_{\mathcal{B}}\left[\| \nabla \hat{f}_k(\boldsymbol {w};\mathcal{B})\|^2\right]\leq G^2$  for all  $\boldsymbol{\mathsf{w}}$  and  $k$

Our analysis of partial client participation also relies on the following standard lemmas.

Lemma 2 (Bounded Projection Norm). Let the projection matrix  $\Phi \in \mathcal{R}^{m\times n}$  be constructed as described in the "Efficient Projection" section, involving a normalized Hadamard matrix  $\pmb{H}$ , a random sign matrix  $\pmb{D}$ , and a subsampling matrix  $\pmb{S}$ . The resulting operator has an exact spectral norm given by:

$$

$$
\left\| \Phi \right\| = \sqrt {\frac {n ^ {\prime}}{m}}. \tag {19}
$$

$$

For our analysis, we formally define  $C_{\Phi} = \|\Phi\|$ , where  $C_{\Phi} = \mathcal{O}\left(\sqrt{\frac{n}{m}}\right)$ .

Lemma 3 (Client-Side Objective and Gradient). The client-side objective  $\tilde{F}_k$  for a given client  $k$  and server message  $\mathbf{v}$  is defined as:

$$

$$
\begin{array}{l} \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = f _ {k} (\boldsymbol {w} _ {k}) + \lambda \left(h _ {\gamma} \left(\boldsymbol {\Phi} \boldsymbol {w} _ {k}\right) - \langle \boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} \rangle\right) \\ + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2}, \tag {20} \\ \end{array}
$$

$$

where  $h_{\gamma}(z) = \frac{1}{\gamma}\sum_{i=1}^{m}\log(\cosh(\gamma z_i))$  is a differentiable surrogate for the  $\ell_1$ -norm. The objective  $\tilde{F}_k(\boldsymbol{w}_k)$  is differentiable with respect to  $\boldsymbol{w}_k$ , and its gradient is given by:

$$

$$
\begin{array}{l} \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = \nabla f _ {k} (\boldsymbol {w} _ {k}) + \lambda \boldsymbol {\Phi} ^ {\top} [ \tanh  (\gamma \boldsymbol {\Phi} \boldsymbol {w} _ {k}) - \boldsymbol {v} ] \\ + \mu \boldsymbol {w} _ {k}. \tag {21} \\ \end{array}
$$

$$

Lemma 4 (Smoothness of Client Objective). Under Assumptions 1 and 1, the client-side objective  $\tilde{F}_k(\boldsymbol{w}_k; \boldsymbol{v})$  is  $L_F$ -smooth with respect to  $\boldsymbol{w}_k$  with high probability, where the smoothness constant is given by:

$$

$$
L _ {F} = L + \lambda \gamma C _ {\Phi} ^ {2} + \mu . \tag {22}
$$

$$

Lemma 5 (Bounded Model Norm). Let Assumption 1-4 hold. For a learning rate  $\eta$  satisfying  $\eta < \frac{1}{3\mu}$ , the expected

squared norm of the client weights is uniformly bounded across all rounds  $t$  with high probability:

$$

$$
\mathbb {E} \left[ \| \boldsymbol {w} _ {k} ^ {t} \| _ {2} ^ {2} \right] \leq W ^ {2}, \quad \forall t \geq 0, \tag {23}
$$

$$

where the bound  $W$  is defined as

$$

$$
W ^ {2} \triangleq \max  \left(\| \boldsymbol {w} _ {k} ^ {0} \| _ {2} ^ {2}, \frac {C ^ {\prime}}{(1 - \alpha) (1 - \alpha^ {R})}\right) \tag {24}
$$

$$

with constants  $\alpha = 1 - \eta \mu (1 - 3\eta \mu)$  and  $C^\prime$  given by:

$$

$$
C ^ {\prime} \triangleq \left(\frac {\eta}{\mu} + 3 \eta^ {2}\right) G ^ {2} + 3 \eta^ {2} \lambda^ {2} \left(2 C _ {\Phi} \sqrt {m}\right) ^ {2}. \tag {25}
$$

$$

Lemma 6 (Variance of Client Sampling). Let  $\{z_k^t\}_{k=1}^K$  be the set of client sketches at round  $t$ . If a subset  $S^t$  of size  $S$  is sampled uniformly at random without replacement, then the variance of the sample mean is bounded by:

$$

$$
\mathbb {E} _ {\mathcal {S} ^ {t}} \left[ \left\| \frac {1}{S} \sum_ {k \in \mathcal {S} ^ {t}} z _ {k} - \bar {z} ^ {t} \right\| _ {2} ^ {2} \right] \leq \frac {K - S}{S K (K - 1)} \left\| z _ {k} - \bar {z} ^ {t} \right\| _ {2} ^ {2}, \tag {26}
$$

$$

where  $\bar{z}^t\triangleq \frac{1}{K}\sum_{k = 1}^{K}z_k^t$

Lemma 7 (Client-Side Objective Descent). After  $R$  local steps of subgradient descent with learning rate  $\eta$  on the smoothed objective  $\tilde{F}_k(\cdot ;\pmb {v}^t)$ , starting from  $\pmb{w}_{k,0}^{t + 1} = \pmb{w}_k^t$ , we have

$$

$$
\begin{array}{l} \mathbb {E} \left[ \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, R} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) \right] \leq \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t}; \boldsymbol {v} ^ {t}\right) + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2} \\ \left. - \eta R \left(1 - \frac {\eta L _ {F}}{2}\right) \cdot \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \left\| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) \right\| _ {2} ^ {2}. \right. \tag {27} \\ \end{array}
$$

$$

# MAIN CONVERGENCE RESULT

Our proof relies on several key intermediate results (proofs in the Appendix). These are combined to analyze the evolution of a global potential function:

$$

$$
\Psi^ {t} \triangleq \sum_ {k = 1} ^ {K} p _ {k} \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t}; \boldsymbol {v} ^ {t}\right). \tag {28}
$$

$$

Our main result bounds the time-averaged expected squared norm of the local gradients.

Theorem 1 (Local Convergence). Under standard assumptions for federated learning analysis (detailed in the Appendix), if the learning rate satisfies  $\eta \leq \frac{1}{L_F}$ , after  $T$  communication rounds with partial client participation, Algorithm 1 guarantees the following convergence

$$

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \mathbb {E} \left[ \sum_ {k = 1} ^ {K} p _ {k} \| \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}) \| _ {2} ^ {2} \right] \\ \leq \frac {\Psi^ {0} - F ^ {*}}{c _ {1} T} + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2 c _ {1}} + \frac {\Delta_ {\operatorname* {m a x}}}{c _ {1}} + \frac {\lambda E _ {S}}{c _ {1}}, \tag {29} \\ \end{array}
$$

$$

with high probability, where  $c_{1} = \eta R(1 - \eta L_{F} / 2)$ . The term  $\Psi^{0}$  is the initial value of the potential function in (28),  $F^{*}$  is its lower bound,  $L_{F} = L + \lambda \gamma C_{\Phi}^{2} + \mu$  is the smoothness

constant of the client objective,  $\Delta_{\mathrm{max}} = 2\lambda (\sqrt{m} WC_{\Phi} + m)$  bounds the error from the one-bit server update, and  $E_{S}$  bounds the error from client sampling, defined as

$$

$$
E _ {S} = \frac {2 \sqrt {m}}{T} \sum_ {t = 0} ^ {T - 1} \sqrt {\frac {K - S}{S K (K - 1)} \sum_ {k = 1} ^ {K} \left\| \boldsymbol {z} _ {k} ^ {t + 1} - \bar {\boldsymbol {z}} ^ {t + 1} \right\| _ {2} ^ {2}}. \tag {30}
$$

$$

Remark 1. Theorem 1 shows that as  $T$  grows, the average squared gradient norm converges to a neighborhood of zero at an  $\mathcal{O}(1 / (RT))$  rate, which means that the algorithm converges to a stationary point of the global potential function. The neighborhood (convergence error) of pFed1BS is governed by stochastic noise  $\mathcal{O}(\eta L_F\sigma^2)$ , communication error  $\mathcal{O}(\Delta_{\max} / (\eta R))$  and client sampling error  $\mathcal{O}(\lambda E_S / (\eta R))$ . To bound the convergence error, the regularization parameter  $\lambda$  must satisfy  $\lambda = \mathcal{O}(1/n)$ , which simultaneously controls  $L_F$ ,  $\Delta_{\max}$  and  $E_S$ .

Remark 2. Note that the client sampling error  $E_S$  vanishes when  $S = K$ , i.e., full client participation. In this case, our convergence bound recovers the result for the full participation setting.

# EXPERIMENTS

We empirically evaluate pFed1BS against state-of-the-art baselines on several benchmarks under challenging non-i.i.d. conditions.

# EXPERIMENTAL SETUP

Datasets and Models: Our experiments use standard image classification benchmarks: MNIST, FMNIST, CIFAR-10, CIFAR-100, and SVHN. To further assess generalizability, we also include experiments on SVHN. We use a two-layer MLP for MNIST and FMNIST, and VGG architectures for the other datasets. We simulate a highly non-i.i.d. environment by partitioning data among 20 clients based on labels.

Baselines: We compare pFed1BS against FedAvg and several one-bit CEFL methods: OBDA, OBCSAA, and zSignFed. To provide a comprehensive evaluation, we also include state-of-the-art PFL and communication-efficient baselines: EDEN (?) and FedBAT(?).

Implementation Details: We run experiments for 100-300 communication rounds with multiple local epochs. Key hyperparameters were set via a grid search to  $\lambda = 0.0005$ ,  $\mu = 0.00001$ , and  $\gamma = 10000$ . The compression ratio is fixed at  $m/n = 0.1$ . All experiments are implemented in PyTorch and run on an NVIDIA RTX 3090 Ti GPU, with results averaged over 10 independent runs.

# EVALUATION METRICS

We use the following metrics for evaluation:

round. For pFed1BS, this is the sum of all uplink one-bit sketches (size  $m$ ) and the downlink one-bit consensus vector (size  $m$ ).

# MAIN RESULTS

Table 2 presents the final test accuracy and per-round communication cost. pFed1BS establishes a new state-of-the-art for communication-constrained FL. On all datasets, it achieves accuracy that is highly competitive with or superior to all baselines, including full-precision FedAvg and advanced communication-efficient methods like OBDA, while reducing communication costs by over  $96\%$ . For instance, on CIFAR-10, pFed1BS achieves  $85.21\%$  accuracy with only  $0.13\mathrm{MB}$  per round, whereas OBDA requires  $1.34\mathrm{MB}$  for a similar  $73.26\%$  accuracy.

![](images/8020c14f24d96df9e280d607a29400951a1617d9f30bdbffda7dcfd655ff0ae1.jpg)

![](images/dbb2f982c1b5b07ef6550cfe7e7e015c4a9fc956622c39e452e6b5586c4326fd.jpg)

On more challenging datasets like CIFAR-100, the advantage is even more pronounced: one-bit baselines suffer a performance collapse, while pFed1BS maintains high accuracy, demonstrating the critical role of personalization in enabling extreme compression. The convergence plots in Figures 3 and 4 further show that pFed1BS achieves faster convergence to a better and more stable solution.

# CONCLUSION

In this work, we proposed pFed1BS, a novel personalized federated learning framework that successfully reconciles

Table 2: Top-1 accuracy (%) and one-round communication cost (MB) of FL algorithms on various datasets under a Non-IID setting. Best results in each column are highlighted in bold. The row for our proposed method, pFed1BS, is highlighted with a gray background for emphasis.

the dual challenges of extreme communication compression and data heterogeneity. By integrating a bidirectional one-bit sketching mechanism with a principled sign-based regularizer, our method reduces communication costs by over  $99\%$  while simultaneously achieving state-of-the-art accuracy. Crucially, pFed1BS achieves this extreme compression with only a minimal trade-off in model accuracy compared to full-precision methods, while decisively outperforming other one-bit baselines that suffer a significant performance collapse in non-i.i.d. settings. Our work demonstrates that a carefully designed personalization strategy is the key to making extreme compression schemes viable, preventing catastrophic performance loss and establishing a new, practical frontier for deploying powerful federated models in real-world, resource-constrained environments.

# APPENDIX

In this appendix, we provide the proofs for the theorems and lemmas in the main paper, as well as additional experimental settings and results that further validate our proposed framework, pFed1BS.

# APPENDIX CONTENTS:

- A. Additional Experimental Results: We present ablation studies on the MNIST and CIFAR-10 datasets to analyze the impact of key components and hyperparameters of pFed1BS.

- B. Theoretical Analysis and Proofs: We provide detailed proofs for all lemmas and the main convergence theorem.

# A. ADDITIONAL EXPERIMENTAL RESULTS

We conduct further analysis on the MNIST dataset under the non-i.i.d. setting to investigate the impact of our method's key components and hyperparameters.

A.1 Effect of the Number of Participating Clients  $(S)$ . To understand how the degree of client participation affects performance, we vary the number of clients  $(S)$  sampled in each communication round, from sparse participation to full participation  $(S = 20)$ . As shown in Figure 1, model performance improves directly with the number of participating clients. A larger  $S$  provides the server with a more accurate and stable estimate of the global consensus, leading to faster convergence and higher final accuracy. Even with a small fraction of participating clients (e.g.,  $S = 5$ ), pFed1BS maintains robust performance, demonstrating its efficacy in settings with limited client availability.

A.2. Effect of Local Epochs  $(R)$ . The number of local epochs,  $R$ , determines the amount of computation each client performs between communication rounds. We evaluate the impact of varying  $R$  from 5 to 30. As shown in Figure 2, increasing the amount of local work generally accelerates convergence in terms of communication rounds. For instance, increasing  $R$  from 5 to 20 leads to a noticeable improvement in convergence speed. However, this benefit saturates quickly; performance for  $R = 20$  is nearly identical. This suggests that while sufficient local training is beneficial, excessive local updates provide diminishing returns and may not be the most efficient use of computational resources, making  $R = 20$  or  $R = 25$  a practical choice for this task.

A.3. Performance with the Fast Hadamard Transform (FHT). A key claim of our work is that the computational efficiency gained by using a structured projection (FHT) does not come at the cost of model performance. To validate this, we compared our FHT-based implementation against a baseline using a dense Gaussian projection matrix. The results in Figure 3 are decisive. The accuracy and loss curves for both methods are nearly identical throughout the training process. This empirically confirms that our use of FHT provides its significant computational advantages (from  $(O)(mn)$  to  $(O)(n\log n)$ ) with no discernible

![](images/e6b28c1b8fb64611eb589258e9e43042bad5d08db88b5238566e7a2aceafd214.jpg)

![](images/7c41e62eac9564717ac6ef072a3523c7c9b1d9f70258f19d5ce776ea0575ff3c.jpg)

impact on convergence or final model quality, making it a critical component for the scalability of pFed1BS.

A.4. Hyperparameter Sensitivity Analysis. To validate the robustness of pFed1BS, we conduct a comprehensive sensitivity analysis of its key hyperparameters on the CIFAR-10 dataset under non-i.i.d. conditions. The results are presented in Table 1.

The analysis demonstrates that pFed1BS is remarkably robust to the precise settings of its key hyperparameters. For instance, the sign-alignment parameter  $\lambda$  can be varied across six orders of magnitude with a performance fluctuation of less than  $0.4\%$ . Similarly, the model performs consistently well for the  $\ell_2$  penalty  $\mu$  and the smoothing parameter  $\gamma$  across very wide ranges, with performance degrading only at extreme values. This stability reduces the burden of meticulous hyperparameter tuning, making our method more practical for real-world deployment. These results confirm that our sign-based regularizer and  $\ell_2$  penalty are essential components that contribute to the final performance without introducing fragility.

# B. THEORETICAL ANALYSIS AND PROOFS

In this subsection, we provide the detailed theoretical analysis for our proposed Algorithm 1, including all supporting lemmas and the full proof for our main convergence result, Theorem 1, which is stated in the main paper.

B.1. Assumptions. Our analysis relies on the following assumptions. Assumption 1 and Assumption 2 are standard assumptions in the analysis of large-scale optimization and federated learning (?). Assumptions 3 (????) and 4 (??) are widely used in FL context.

![](images/b94ca148c1e3733cebb1109b4d9cafa679e1fd096d9b78160c074e640e3a19ea.jpg)

![](images/cf58a19493e8a40790e280d3b2ccb522f9b90ab004ec3be3468948ca9e836b92.jpg)

Assumption 1 (L-smoothness). The local objective function  $f_{k}(\cdot)$  is differentiable and  $L$ -smooth for all clients  $k \in 1, \ldots, K$ , i.e., for any  $\mathbf{w}_{1}, \mathbf{w}_{2} \in \mathbb{R}^{n}$ , there exists a constant  $L > 0$  such that

$$

$$
\left\| \nabla f _ {k} \left(\boldsymbol {w} _ {1}\right) - \nabla f _ {k} \left(\boldsymbol {w} _ {2}\right) \right\| \leq L \| \boldsymbol {w} _ {1} - \boldsymbol {w} _ {2} \|. \tag {1}
$$

$$

Assumption 2 (Bounded Below). The global potential function  $\Psi(\boldsymbol{w}_1, \ldots, \boldsymbol{w}_K; \boldsymbol{v}) = \sum_{k=1}^{K} p_k \tilde{F}_k(\boldsymbol{w}_k)$  is bounded below by some value  $F^*$ .

Assumption 3 (Bounded Stochastic Gradient Variance). The variance of the stochastic gradients computed on minibatches is bounded as follows

$$

$$
\mathbb {E} _ {\mathcal {B}} \left[ \| \nabla \hat {f} _ {k} (\boldsymbol {w}; \mathcal {B}) - \nabla f _ {k} (\boldsymbol {w}) \| ^ {2} \right] \leq \sigma^ {2}, \quad \forall \boldsymbol {w}. \tag {2}
$$

$$

Assumption 4 (Bounded Task Gradient Variance). The stochastic gradient of the task loss  $f_{k}$  has a bounded second moment, i.e., there exists a constant  $G > 0$  such that  $\mathbb{E}_{\mathcal{B}}\left[\| \nabla \hat{f}_k(\boldsymbol {w};\mathcal{B})\|^2\right]\leq G^2$  for all  $\boldsymbol{\mathsf{w}}$  and  $k$ .

B.2. Supporting Lemmas and Proofs. We now establish several key lemmas concerning the properties of the projection matrix norm, the client objective, model boundedness, and server aggregation.

Lemma 1 (Bounded Projection Norm). Let the projection matrix  $\Phi \in \mathcal{R}^{m\times n}$  be constructed as described in the "Efficient Projection" section, involving a normalized Hadamard matrix  $\pmb{H}$ , a random sign matrix  $D$ , and a subsampling matrix  $S$ . The resulting operator has an exact spectral norm

(c) Impact of  $\gamma$

Table 1: Sensitivity analysis of the key hyperparameters of pFed1BS ( $\lambda$ ,  $\mu$ , and  $\gamma$ ) on the CIFAR-10 (Non-IID) dataset. We report the Top-1 test accuracy (\%) as 'mean ± standard deviation' over multiple independent runs. The best result in each subtable is shown in bold.

![](images/bef17ef07c22b62c1eb99e4f238a46c68e13425586ab06fd6597dd5c8c49976c.jpg)

![](images/c43c69322921cc9acc268a6505873b61d0b8513a2849a57564a472c44627ecc0.jpg)

given by:

$$

$$
\left\| \Phi \right\| = \sqrt {\frac {n ^ {\prime}}{m}}, \tag {3}
$$

$$

For our analysis, we formally assume that  $\| \Phi \| = C_{\Phi}$  where  $C_{\Phi} = \mathcal{O}\left(\sqrt{\frac{n}{m}}\right)$ .

Proof. The proof relies on analyzing the spectral norm of the operator  $\Phi$ . Let us define the projection matrix as  $\Phi = \sqrt{\frac{n'}{m}}\mathbf{S}\mathbf{H}\mathbf{D}$ , where the dimensions and properties of the matrices are as described in the lemma and the "Efficient Projection" section. Specifically,  $\pmb{D}$  is a diagonal sign matrix, satisfying  $D^{\top}D = I_{n}$ .  $\pmb{H}$  is a normalized Hadamard matrix, satisfying  $\pmb{H}^{\top}\pmb{H} = \pmb{I}_{n}^{\prime}$ .  $\pmb{S}$  is a subsampling matrix that

selects  $m$  rows from  $n^{\prime}$  satisfying  $\boldsymbol {S}\boldsymbol{S}^{\top} = \boldsymbol{I}_{m}$  .Let us define an intermediate matrix  $\mathbf{Q} = \mathbf{SHD}$  . Then  $\Phi = \sqrt{\frac{n^{\prime}}{m}}\mathbf{Q}$  The spectral norm of  $\Phi$  is given by  $\| \Phi \| = \sqrt{\frac{n^{\prime}}{m}}\| \mathbf{Q}\|$

The core of the proof is to compute the spectral norm of  $\mathbf{Q}$ . We can do this by analyzing the matrix  $\mathbf{QQ}^{\top}$ :

$$

$$
\begin{array}{l} \mathbf {Q} \mathbf {Q} ^ {T} = (\mathbf {S H D}) (\mathbf {S H D}) ^ {T} \\ = (\mathbf {S H D}) \left(\mathbf {D} ^ {T} \mathbf {H} ^ {T} \mathbf {S} ^ {T}\right) \\ = \mathbf {S} \mathbf {H} (\mathbf {D} \mathbf {D} ^ {T}) \mathbf {H} ^ {T} \mathbf {S} ^ {T} \\ = \mathbf {S H I n} ^ {\prime} \mathbf {H} ^ {T} \mathbf {S} ^ {T} \quad (\text {s i n c e} \mathbf {D D} ^ {T} = \mathbf {I}) \tag {4} \\ = \mathbf {S} (\mathbf {H} \mathbf {H} ^ {T}) \mathbf {S} ^ {T} \\ = \mathbf {S} \mathbf {I} n ^ {\prime} \mathbf {S} ^ {T} \quad (\text {s i n c e} \mathbf {H} \mathbf {H} ^ {T} = \mathbf {I}) \\ = \mathbf {S} \mathbf {S} ^ {T} = \mathbf {I} _ {m}. \\ \end{array}
$$

$$

The eigenvalues of an identity matrix  $\mathbf{I}_m$  are all 1. The eigenvalues of  $\mathbf{Q}\mathbf{Q}^{\top}$  are the squared singular values of  $\mathbf{Q}$ . Therefore, all non-zero singular values of  $\mathbf{Q}$  must be 1. The spectral norm  $\| \mathbf{Q}\|$  is defined as the largest singular value, so we have:

$$

$$
\left\| \mathbf {Q} \right\| = \sigma_ {\max } (\mathbf {Q}) = 1. \tag {5}
$$

$$

Substituting this result back into the expression for the norm of  $\Phi$ , we get:

$$

$$
\left\| \boldsymbol {\Phi} \right\| = \sqrt {\frac {n ^ {\prime}}{m}} \cdot \left\| \mathbf {Q} \right\| = \sqrt {\frac {n ^ {\prime}}{m}} \cdot 1 = \sqrt {\frac {n ^ {\prime}}{m}}. \tag {6}
$$

$$

Finally, since  $n'$  is the smallest power of two greater than or equal to  $n$ , we can express the norm in asymptotic notation for our analysis:

$$

$$
C _ {\Phi} = \mathcal {O} \left(\sqrt {\frac {n}{m}}\right). \tag {7}
$$

$$

This completes the proof.

Lemma 2 (Client-Side Objective and Gradient). The client-side objective  $\tilde{F}_k$  for a given client  $k$  and server message  $\mathbf{v}$  is defined as:

$$

$$
\begin{array}{l} \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = f _ {k} (\boldsymbol {w} _ {k}) + \lambda \left(h _ {\gamma} \left(\boldsymbol {\Phi} \boldsymbol {w} _ {k}\right) - \langle \boldsymbol {v}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} \rangle\right) \\ + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2}, \tag {8} \\ \end{array}
$$

$$

where  $h_{\gamma}(z) = \frac{1}{\gamma}\sum_{i=1}^{m}\log(\cosh(\gamma z_i))$  is a differentiable surrogate for the  $\ell_1$ -norm. The objective  $\tilde{F}_k(\boldsymbol{w}_k)$  is differentiable with respect to  $\boldsymbol{w}_k$ , and its gradient is given by:

$$

$$
\begin{array}{l} \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = \nabla f _ {k} (\boldsymbol {w} _ {k}) + \lambda \boldsymbol {\Phi} ^ {\top} [ \tanh  (\gamma \boldsymbol {\Phi} \boldsymbol {w} _ {k}) - \boldsymbol {v} ] \\ \quad + \mu \boldsymbol {w} _ {k}. \end{array} \tag {9}
$$

$$

Proof. The proof follows by direct differentiation of Equation 8. The gradient of the standard loss  $f_{k}(\pmb{w}_{k})$  and the  $\ell_2$  regularization term  $\frac{\mu}{2}\| \pmb{w}_k\| _2^2$  are  $\nabla f_k(\pmb {w}_k)$  and  $\mu \pmb {w}_k$ , respectively. The key is the gradient of the term  $\lambda (h_{\gamma}(\Phi \pmb{w}_k) - \langle \pmb {v},\pmb {\Phi}\pmb{w}_k\rangle)$ . Using the chain rule, the gradient of  $h_\gamma (\Phi \pmb {w}_k)$  with respect to  $\pmb{w}_k$  is

$$

$$
\begin{array}{l} \nabla_ {\boldsymbol {w} _ {k}} h _ {\gamma} (\Phi \boldsymbol {w} _ {k}) = \Phi^ {\top} \nabla_ {\boldsymbol {z}} h _ {\gamma} (\boldsymbol {z}) \Big | _ {\boldsymbol {z} = \Phi \boldsymbol {w} _ {k}} \\ = \boldsymbol {\Phi} ^ {\top} \tanh  (\gamma \boldsymbol {\Phi} \boldsymbol {w} _ {k}). \tag {10} \\ \end{array}
$$

$$

The gradient of  $-\lambda \langle \pmb{v}, \pmb{\Phi} \pmb{w}_k \rangle$  is  $-\lambda \pmb{\Phi}^\top \pmb{v}$ . Combining all terms yields the expression in Equation (9).

Remark 3. As the smoothing parameter  $\gamma \rightarrow \infty$ , the function  $\tanh (\gamma z_i)$  converges pointwise to  $\mathrm{sign}(z_i)$ , and the smooth surrogate  $h_\gamma (\Phi \pmb {w}_k)$  approaches the non-smooth  $\ell_1$  norm  $\| \Phi \pmb {w}_k\| _1$ . Consequently, for a large  $\gamma$ , the gradient descent on  $\tilde{F}_k(\pmb {w}_k)$  closely approximates subgradient descent on the non-smooth objective:

$$

$$
\nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) \approx \nabla f _ {k} (\boldsymbol {w} _ {k}) + \lambda \boldsymbol {\Phi} ^ {\top} (\operatorname {s i g n} (\boldsymbol {\Phi} \boldsymbol {w} _ {k}) - \boldsymbol {v}) + \mu \boldsymbol {w} _ {k}. \tag {11}
$$

$$

Lemma 3 (Smoothness of Client Objective). Under Assumptions 1 and 1, the client-side objective  $\tilde{F}_k(\boldsymbol{w}_k; \boldsymbol{v})$  is  $L_F$ -smooth with respect to  $\boldsymbol{w}_k$  with high probability, where the smoothness constant is given by:

$$

$$
L _ {F} = L + \lambda \gamma C _ {\Phi} ^ {2} + \mu . \tag {12}
$$

$$

Proof. A function is  $L_{F}$ -smooth if its Hessian  $\nabla^2\tilde{F}_k(\boldsymbol{w}_k;\boldsymbol{v})$  satisfies  $\| \nabla^2\tilde{F}_k(\boldsymbol{w}_k;\boldsymbol{v}) \| \leq L_F$  for all  $\boldsymbol{w}_k$ . We analyze the Hessian of  $\tilde{F}_k(\boldsymbol{w}_k;\boldsymbol{v})$  by decomposing it into its three constituent parts:

$$

$$
\begin{array}{l} \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) = f _ {k} (\boldsymbol {w} _ {k}) + \lambda \left(h _ {\gamma} \left(\Phi \boldsymbol {w} _ {k}\right) - \langle \boldsymbol {v}, \Phi \boldsymbol {w} _ {k} \rangle\right) \\ + \frac {\mu}{2} \| \boldsymbol {w} _ {k} \| _ {2} ^ {2}. \tag {13} \\ \end{array}
$$

$$

The Hessian of the sum is the sum of the Hessians. We bound the spectral norm of the Hessian for each part as follows.

$$

$$
\nabla^ {2} \left(\lambda h _ {\gamma} (\boldsymbol {\Phi} \boldsymbol {w})\right) = \lambda \gamma \boldsymbol {\Phi} ^ {\top} \operatorname {d i a g} \left(\operatorname {s e c h} ^ {2} \left(\gamma \left(\boldsymbol {\Phi} \boldsymbol {w} _ {k}\right) _ {i}\right)\right) \boldsymbol {\Phi}. \tag {14}
$$

$$

The spectral norm of this Hessian is bounded as follows:

$$

$$
\begin{array}{l} \| \lambda \gamma \pmb {\Phi} ^ {\top} \mathrm {d i a g} (\mathrm {s e c h} ^ {2} (\gamma (\pmb {\Phi} \pmb {w} _ {k}) _ {i}) \pmb {\Phi} \| \\ \leq \lambda \gamma \| \boldsymbol {\Phi} ^ {\top} \| \| \mathrm {d i a g} (\mathrm {s e c h} ^ {2} (\gamma (\boldsymbol {\Phi} \boldsymbol {w} _ {k}) _ {i}) \| \| \boldsymbol {\Phi} \| \\ = \lambda \gamma \| \operatorname {d i a g} \left(\operatorname {s e c h} ^ {2} \left(\gamma \left(\boldsymbol {\Phi} \boldsymbol {w} _ {k}\right) _ {i}\right) \| \| \boldsymbol {\Phi} \| ^ {2}. \right. \tag {15} \\ \end{array}
$$

$$

Since  $0 \leq \operatorname{sech}^2(z) \leq 1$  for any  $z \in \mathbb{R}$ , the spectral norm of the diagonal matrix  $\operatorname{diag}(\operatorname{sech}^2(\gamma(\Phi w_k)_i))$  is  $\| \operatorname{diag}(\operatorname{sech}^2(\gamma(\Phi w_k)_i)) \| \leq 1$ . From Assumption 1, we have  $\| \Phi \| \leq C_\Phi$ . Substituting these bounds gives:

$$

$$
\left\| \nabla^ {2} \left(\lambda h _ {\gamma} \left(\boldsymbol {\Phi} \boldsymbol {w} _ {k}\right)\right) \right\| \leq \lambda \gamma C _ {\Phi} ^ {2}. \tag {16}
$$

$$

By the triangle inequality, the norm of the total Hessian is bounded by the sum of the norms of the individual Hessians:

$$

$$
\left\| \nabla^ {2} \tilde {F} _ {k} (\boldsymbol {w} _ {k}; \boldsymbol {v}) \right\| \leq L + \lambda \gamma C _ {\Phi} ^ {2} + \mu . \tag {17}
$$

$$

![](images/01b3d476f7712801c0540204d87c01883b071cf23238ec8cc18fc46711a8db3d.jpg)

Lemma 4 (Bounded Model Norm). Let Assumption 1-4 hold. For a learning rate  $\eta$  satisfying  $\eta < \frac{1}{3\mu}$ , the expected squared norm of the client weights is uniformly bounded across all rounds  $t$  with high probability:

$$

$$
\mathbb {E} \left[ \| \boldsymbol {w} _ {k} ^ {t} \| _ {2} ^ {2} \right] \leq W ^ {2}, \quad \forall t \geq 0, \tag {18}
$$

$$

where the bound  $W$  is defined as

$$

$$
W ^ {2} \triangleq \max  \left(\| \boldsymbol {w} _ {k} ^ {0} \| _ {2} ^ {2}, \frac {C ^ {\prime}}{(1 - \alpha) (1 - \alpha^ {R})}\right) \tag {19}
$$

$$

with constants  $\alpha = 1 - \eta \mu (1 - 3\eta \mu)$  and  $C^\prime$  given by:

$$

$$
C ^ {\prime} \triangleq \left(\frac {\eta}{\mu} + 3 \eta^ {2}\right) G ^ {2} + 3 \eta^ {2} \lambda^ {2} \left(2 C _ {\Phi} \sqrt {m}\right) ^ {2}. \tag {20}
$$

$$

Proof. For simplicity, we analyze the local updates for a single client  $k$  within a communication round  $t$ . We denote  $\boldsymbol{w}_r \triangleq \boldsymbol{w}_{k,r}^{t+1}$ . The update rule for the client model is:

$$

$$
\boldsymbol {w} _ {r + 1} = \boldsymbol {w} _ {r} - \eta \boldsymbol {d} _ {r}, \tag {21}
$$

$$

where  $\pmb{d}_r = \nabla \hat{f}_k(\pmb{w}_r; \mathcal{B}_r) + \lambda \nabla \tilde{g}(\pmb{v}^t, \pmb{\Phi}\pmb{w}_r) + \mu \pmb{w}_r$  is the stochastic subgradient of the client objective.

We analyze the evolution of the squared  $\ell_2$ -norm of  $\pmb{w}_{r + 1}$ :

$$

$$
\mathbb {E} [ \| \boldsymbol {w} _ {r + 1} \| _ {2} ^ {2} ] = \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] - 2 \eta \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \boldsymbol {d} _ {r} \rangle ] + \eta^ {2} \mathbb {E} [ \| \boldsymbol {d} _ {r} \| _ {2} ^ {2} ]. \tag {22}
$$

$$

First, we bound the inner product term. The inner product term  $\langle \pmb{w}_r, \pmb{d}_r \rangle$  could be written as:

$$

$$
\begin{array}{l} \langle \boldsymbol {w} _ {r}, \boldsymbol {d} _ {r} \rangle = \langle \boldsymbol {w} _ {r}, \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) + \lambda \nabla \tilde {g} (\boldsymbol {v} ^ {t}, \boldsymbol {\Phi} \boldsymbol {w} _ {r}) + \mu \boldsymbol {w} _ {r} \rangle \\ = \langle \boldsymbol {w} _ {r}, \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \rangle + \lambda \langle \boldsymbol {w} _ {r}, \nabla \tilde {g} (\boldsymbol {v} ^ {t}, \boldsymbol {\Phi} \boldsymbol {w} _ {r}) \rangle \\ + \mu \| \boldsymbol {w} _ {r} \| _ {2} ^ {2}. \tag {23} \\ \end{array}
$$

$$

The term  $\mu \| \pmb{w}_r\| _2^2$  provides the key dissipative effect. For the second term, we have:

$$

$$
\begin{array}{l} \langle \boldsymbol {w} _ {r}, \nabla \tilde {g} \rangle = \langle \boldsymbol {w} _ {r}, \boldsymbol {\Phi} ^ {\top} (\tanh  (\gamma \boldsymbol {\Phi} \boldsymbol {w} _ {r}) - \boldsymbol {v} ^ {t}) \rangle \\ \approx \langle \boldsymbol {w} _ {r}, \boldsymbol {\Phi} ^ {\top} (\operatorname {s i g n} (\boldsymbol {\Phi} \boldsymbol {w} _ {r}) - \boldsymbol {v} ^ {t}) \rangle \\ = \left\langle \boldsymbol {\Phi} \boldsymbol {w} _ {r}, \operatorname {s i g n} \left(\boldsymbol {\Phi} \boldsymbol {w} _ {r}\right) - \boldsymbol {v} ^ {t} \right\rangle \\ = \| \boldsymbol {\Phi} \boldsymbol {w} _ {r} \| _ {1} - \left\langle \boldsymbol {\Phi} \boldsymbol {w} _ {r}, \boldsymbol {v} ^ {t} \right\rangle \geq 0, \tag {24} \\ \end{array}
$$

$$

where the inequality follows from the property that  $\pmb{v}^t$  has entries in  $\{-1,0,1\}$ , and thus by the Hölder's inequality,  $\langle \Phi \pmb{w}_r, \pmb{v}^t \rangle \leq \| \Phi \pmb{w}_r \|_1 \| \pmb{v}^t \|_{\infty} \leq \| \Phi \pmb{w}_r \|_1$ . Therefore, the second term in Eq. (23) is non-negative. Taking expectation of Eq. (23) over the mini-batch  $\mathcal{B}_r$ :

$$

$$
\begin{array}{l} \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \boldsymbol {d} _ {r} \rangle ] = \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \rangle ] + \lambda \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \nabla \tilde {g} \rangle ] \\ + \mu \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] \\ \geq \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \rangle ] + \mu \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ], \tag {25} \\ \end{array}
$$

$$

Applying Cauchy-Schwarz and Young's inequality with a parameter  $\epsilon > 0$  to the remaining term:

$$

$$
\begin{array}{l} - \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \rangle ] \\ \leq \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} \| \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \| _ {2} ] \\ \leq \frac {\epsilon}{2} \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \frac {1}{2 \epsilon} \mathbb {E} [ \| \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \| _ {2} ^ {2} ] \\ \leq \frac {\epsilon}{2} \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \frac {G ^ {2}}{2 \epsilon}, \tag {26} \\ \end{array}
$$

$$

where we use the Assumption 4  $(\mathbb{E}[\| \nabla \hat{f}_k(\pmb {w}_r;\mathcal{B}_r)\| _2^2\leq G^2)$  Combining these, we get:

$$

$$
- 2 \eta \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \boldsymbol {d} _ {r} \rangle ] \leq - 2 \eta \mu \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \eta \epsilon \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \frac {\eta}{\epsilon} G ^ {2}. \tag {27}
$$

$$

Next, we bound the squared norm of the stochastic gradient,  $\mu \| \pmb{d}_r\| _2^2$

$$

$$
\begin{array}{l} \mathbb {E} \left[ \| \boldsymbol {d} _ {r} \| _ {2} ^ {2} \right] = \mathbb {E} \left[ \| \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) + \lambda \nabla \tilde {g} + \mu \boldsymbol {w} _ {r} \| _ {2} ^ {2} \right] \\ \leq 3 (\mathbb {E} [ \| \nabla \hat {f} _ {k} (\boldsymbol {w} _ {r}; \mathcal {B} _ {r}) \| _ {2} ^ {2} ] + \lambda^ {2} \mathbb {E} [ \| \nabla \tilde {g} \| _ {2} ^ {2} \\ + \mu^ {2} \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2}) ] \\ \leq 3 \left(G ^ {2} + \lambda^ {2} C _ {g} ^ {2} + \mu^ {2} \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ]\right), \tag {28} \\ \end{array}
$$

$$

where we define the constant  $C_g \triangleq 2C_{\Phi}\sqrt{m}$ , since  $\| \nabla \tilde{g} \|_2^2 = \| \Phi^\top(\mathrm{sign}(\Phi \pmb{w}_r) - \pmb{v}^t) \|_2^2 \leq \| \Phi^\top \|_2^2 \| (\mathrm{sign}(\Phi \pmb{w}_r) - \pmb{v}^t) \|_2^2 \leq C_{\Phi} \cdot 2\sqrt{m}$ .

Finally, we combine the bounds. Substituting everything back into the main expansion:

$$

$$
\begin{array}{l} \mathbb {E} [ \| \boldsymbol {w} _ {r + 1} \| _ {2} ^ {2} ] \\ = \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] - 2 \eta \mathbb {E} [ \langle \boldsymbol {w} _ {r}, \boldsymbol {d} _ {r} \rangle ] + \eta^ {2} \mathbb {E} [ \| \boldsymbol {d} _ {r} \| _ {2} ^ {2} ] \\ \leq \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \left(- 2 \eta \mu \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \eta \epsilon \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] + \frac {\eta}{\epsilon} G ^ {2}\right) \\ + 3 \eta^ {2} \left(G ^ {2} + \lambda^ {2} C _ {g} ^ {2} + \mu^ {2} \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ]\right) \\ = (1 - 2 \eta \mu + \eta \epsilon + 3 \eta^ {2} \mu^ {2}) \mathbb {E} [ \| \boldsymbol {w} _ {r} \| ^ {2} ] \\ + \left(\frac {\eta}{\epsilon} + 3 \eta^ {2}\right) G ^ {2} + 3 \eta^ {2} \lambda^ {2} C _ {g} ^ {2}. \tag {29} \\ \end{array}
$$

$$

This inequality holds for any  $\epsilon > 0$ . We make a strategic choice to simplify the coefficient of  $\mathbb{E}[\| \pmb{w}_r\|_2^2]$ . Let  $\epsilon = \mu$ .

$$

$$
\begin{array}{l} \mathbb {E} [ \| \boldsymbol {w} _ {r + 1} \| _ {2} ^ {2} ] \leq (1 - 2 \eta \mu + \eta \mu + 3 \eta^ {2} \mu^ {2}) \mathbb {E} [ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} ] \\ + \left(\frac {\eta}{\mu} + 3 \eta^ {2}\right) G ^ {2} + 3 \eta^ {2} \lambda^ {2} C _ {g} ^ {2} \\ = \left(1 - \eta \mu + 3 \eta^ {2} \mu^ {2}\right) \mathbb {E} \left[ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} \right] + C ^ {\prime}, \tag {30} \\ \end{array}
$$

$$

where we define  $C' \triangleq (\frac{\eta}{\mu} + 3\eta^2)G^2 + 3\eta^2\lambda^2 C_q^2$ .

To ensure contraction, we require the coefficient of  $\mathbb{E}[\| \pmb{w}_r\|_2^2]$  to be strictly less than 1. We need  $1 - \eta \mu + 3\eta^2\mu^2 < 1$ , which simplifies to  $-\eta \mu + 3\eta^2\mu^2 < 0$ . Since  $\eta, \mu > 0$ , we can divide by  $\eta \mu$ , yielding  $-1 + 3\eta \mu < 0$ . This gives the condition  $\eta < \frac{1}{3\mu}$ .

By choosing  $\eta < \frac{1}{3\mu}$ , we have  $3\eta \mu < 1$ . The coefficient can be written as  $1 - \eta \mu (1 - 3\eta \mu)$ . Let  $\alpha = 1 - \eta \mu (1 - 3\eta \mu) < 1$ . We obtain the recursive relationship:

$$

$$
\mathbb {E} \left[ \| \boldsymbol {w} _ {r + 1} \| _ {2} ^ {2} \right] \leq \alpha \mathbb {E} \left[ \| \boldsymbol {w} _ {r} \| _ {2} ^ {2} \right] + C ^ {\prime}. \tag {31}
$$

$$

This recursive inequality holds for each local step  $r$  within any communication round  $t + 1$ . By applying this inequality recursively  $R$  times from  $r = 0$  to  $r = R - 1$ , starting with the model from the previous round  $\boldsymbol{w}_k^t$ :

$$

$$
\begin{array}{l} \mathbb {E} [ \| \boldsymbol {w} _ {k, R} ^ {t + 1} \| _ {2} ^ {2} ] \leq \alpha \mathbb {E} [ \| \boldsymbol {w} _ {k, R - 1} ^ {t + 1} \| _ {2} ^ {2} ] + C ^ {\prime} \\ \leq \alpha^ {R} \mathbb {E} [ \| \boldsymbol {w} _ {k, 0} ^ {t + 1} \| _ {2} ^ {2} ] + C ^ {\prime} \sum_ {i = 0} ^ {R - 1} \alpha^ {i} \\ \leq \alpha^ {R} \mathbb {E} [ \| \boldsymbol {w} _ {k} ^ {t} \| _ {2} ^ {2} ] + \frac {C ^ {\prime}}{1 - \alpha} \\ = \alpha^ {R} \mathbb {E} [ \| \boldsymbol {w} _ {k} ^ {t} \| _ {2} ^ {2} ] + \frac {C ^ {\prime}}{\eta \mu (1 - 3 \eta \mu)}. \tag {32} \\ \end{array}
$$

$$

Since  $\pmb{w}_k^{t + 1}$  is used as the starting point for the next round, we have established a recursive relationship across communication rounds:

$$

$$
\mathbb {E} \left[ \| \boldsymbol {w} _ {k} ^ {t + 1} \| _ {2} ^ {2} \right] \leq \alpha^ {R} \mathbb {E} \left[ \| \boldsymbol {w} _ {k} ^ {t} \| _ {2} ^ {2} \right] + \frac {C ^ {\prime}}{\eta \mu (1 - 3 \eta \mu)}. \tag {33}
$$

$$

As  $\alpha \in (0,1)$ , the term  $\alpha^R$  is also a constant contraction factor less than 1. This relationship ensures that the sequence of expected squared norms  $\{\mathbb{E}[\| \pmb{w}_k^t\|_2^2]\}_{t = 0}^{T - 1}$  cannot diverge.

Therefore, the sequence is uniformly bounded. A simple upper bound for all  $t \geq 0$  is given by the maximum of its starting value and the fixed-point value of the recursion:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \| \boldsymbol {w} _ {k} ^ {t} \| _ {2} ^ {2} \right] \leq \max  \left(\| \boldsymbol {w} _ {k} ^ {0} \| _ {2} ^ {2}, \frac {C ^ {\prime}}{\eta \mu (1 - 3 \eta \mu) (1 - \alpha^ {R})}\right) \\ = \max  \left(\| \boldsymbol {w} _ {k} ^ {0} \| _ {2} ^ {2}, \frac {C ^ {\prime}}{(1 - \alpha) (1 - \alpha^ {R})}\right). \tag {34} \\ \end{array}
$$

$$

Since all terms on the right-hand side are constants independent of  $t$ , this proves the uniform boundedness and completes the proof.

Lemma 5 (Variance of Client Sampling). Let  $\{z_k^t\}_{k=1}^K$  be the set of client sketches at round  $t$ . If a subset  $S^t$  of size  $S$  is sampled uniformly at random without replacement, then the variance of the sample mean is bounded by:

$$

$$
\begin{array}{l} \mathbb {E} _ {\mathcal {S} ^ {t}} \left[ \left\| \frac {1}{S} \sum_ {k \in \mathcal {S} ^ {t}} z _ {k} - \bar {z} ^ {t} \right\| _ {2} ^ {2} \right] \\ \leq \frac {K - S}{S K (K - 1)} \sum_ {k = 1} ^ {K} \left\| \boldsymbol {z} _ {k} - \bar {\boldsymbol {z}} ^ {t} \right\| _ {2} ^ {2}, \tag {35} \\ \end{array}
$$

$$

where  $\bar{z}^t \triangleq \frac{1}{K}\sum_{k=1}^{K} z_k^t$  is the true mean of all client sketches.

Proof. This is a standard result from sampling theory. For completeness, we provide a sketch of the proof. Let  $\alpha_{k} = z_{k}^{t} - \bar{z}^{t}$ . Note that  $\sum_{k=1}^{K} \alpha_{k} = 0$ . The left-hand side becomes  $\mathbb{E}[\| \frac{1}{S} \sum_{k \in S^{t}} \alpha_{k} \|^{2}]$ . We expand the squared norm:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \left\| \frac {1}{S} \sum_ {k \in \mathcal {S} ^ {t}} \boldsymbol {\alpha} _ {k} \right\| _ {2} ^ {2} \right] \\ = \frac {1}{S ^ {2}} \mathbb {E} \left[ \sum_ {k \in \mathcal {S} ^ {t}} \| \boldsymbol {\alpha} _ {k} \| _ {2} ^ {2} + \sum_ {i, j \in \mathcal {S} ^ {t}, i \neq j} \langle \boldsymbol {\alpha} _ {i}, \boldsymbol {\alpha} _ {j} \rangle \right] \\ = \frac {1}{S ^ {2}} \left(\sum_ {k = 1} ^ {K} p _ {k \in \mathcal {S} ^ {t}} \| \boldsymbol {\alpha} _ {k} \| _ {2} ^ {2} + \sum_ {i \neq j} p _ {i, j \in \mathcal {S} ^ {t}} \langle \boldsymbol {\alpha} _ {i}, \boldsymbol {\alpha} _ {j} \rangle\right) \\ = \frac {1}{S ^ {2}} \left(\frac {S}{K} \sum_ {k = 1} ^ {K} \| \boldsymbol {\alpha} _ {k} \| _ {2} ^ {2} + \frac {S (S - 1)}{K (K - 1)} \sum_ {i \neq j} \langle \boldsymbol {\alpha} _ {i}, \boldsymbol {\alpha} _ {j} \rangle\right). \tag {36} \\ \end{array}
$$

$$

Using the property  $\sum_{i\neq j}\langle \pmb {\alpha}_i,\pmb {\alpha}_j\rangle = -\sum_{k = 1}^{K}\| \pmb {\alpha}_k\| _2^2$  , we substitute and simplify:

$$

$$
\begin{array}{l} = \frac {1}{S ^ {2}} \left(\frac {S}{K} - \frac {S (S - 1)}{K (K - 1)}\right) \sum_ {k = 1} ^ {K} \| \boldsymbol {\alpha} _ {k} \| _ {2} ^ {2} \\ = \frac {K - S}{S \cdot K (K - 1)} \sum_ {k = 1} ^ {K} \| \boldsymbol {\alpha} _ {k} \| _ {2} ^ {2}, \tag {37} \\ \end{array}
$$

$$

which is equivalent to the stated result.

![](images/118fa6311a61175c6d6895a84dde3abe9a5fa3a06e58dca4f96840df4613456f.jpg)

We now establish two key lemmas regarding the server and client updates before presenting the main convergence theorem.

Lemma 6 (Optimal Server Aggregation). For any set of client sign vectors  $\{z_k\}_{k \in S^t} \subset \{\pm 1\}^m$ , the server update rule

$$

$$
\boldsymbol {v} = \operatorname {s i g n} \left(\sum_ {k \in S ^ {t}} p _ {k} \boldsymbol {z} _ {k}\right) \tag {38}
$$

$$

is the exact minimizer of the server's objective  $\min_{\pmb{v} \in \{\pm 1\}^m} \sum_{k \in S^t} p_k g(\pmb{v}, \pmb{z}_k)$ .

Proof. Let  $\pmb{z}_k = \mathrm{sign}(\Phi \pmb{w}_k^{t + 1})$ . Since both  $\pmb{v}$  and  $\pmb{z}_k$  are in  $\{\pm 1\}^m$ , we can write  $g(\pmb{v}, \pmb{z}_k) = \frac{1}{2} (\| \pmb{z}_k \|_1 - \langle \pmb{v}, \pmb{z}_k \rangle)$ , where  $m$  is the dimension of  $\pmb{z}_k$ . The objective becomes:

$$

$$
\begin{array}{l} \min  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \sum_ {k \in \mathcal {S} ^ {t}} p _ {k} g (\boldsymbol {v}, \boldsymbol {z} _ {k}) \\ = \min  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \sum_ {k \in \mathcal {S} ^ {t}} \frac {p _ {k}}{2} \left(\| \boldsymbol {z} _ {k} \| _ {1} - \langle \boldsymbol {v}, \boldsymbol {z} _ {k} \rangle\right) \\ = \min  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \left(\frac {1}{2} \sum_ {k \in S ^ {t}} p _ {k} \| \boldsymbol {z} _ {k} \| _ {1} - \frac {1}{2} \sum_ {k \in S ^ {t}} p _ {k} \langle \boldsymbol {v}, \boldsymbol {z} _ {k} \rangle\right). \tag {39} \\ \end{array}
$$

$$

Since the first term is constant with respect to  $\pmb{v}$ , and  $\| z_k\| _1 = m$ , the problem is equivalent to:

$$

$$
\max  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \sum_ {k \in \mathcal {S} ^ {t}} p _ {k} \langle \boldsymbol {v}, \boldsymbol {z} _ {k} \rangle = \max  _ {\boldsymbol {v} \in \{\pm 1 \} ^ {m}} \left\langle \boldsymbol {v}, \sum_ {k \in \mathcal {S} ^ {t}} p _ {k} \boldsymbol {z} _ {k} \right\rangle . \tag {40}
$$

$$

This is a dot product between a variable vector  $\pmb{v}$  and a fixed vector  $\bar{\pmb{z}} = \sum_{k\in S^t}p_k\pmb{z}_k$ . To maximize the dot product under the constraint that each component  $v_{j}\in \{\pm 1\}$ , we must choose  $v_{j}$  to have the same sign as the corresponding component  $\bar{z}_j$ . Therefore, the optimal solution is:

$$

$$
\boldsymbol {v} = \operatorname {s i g n} (\bar {z}) = \operatorname {s i g n} \left(\sum_ {k \in \mathcal {S} ^ {t}} p _ {k} z _ {k}\right). \tag {41}
$$

$$

![](images/ede91789f3f4f46461263db99a3a4fd7bb9cec3fbd1b59391730d9a1b84ce91f.jpg)

Lemma 7 (Client-Side Objective Descent). After  $R$  local steps of subgradient descent with learning rate  $\eta$  on the smoothed objective  $\tilde{F}_k(\cdot; \mathbf{v}^t)$ , starting from  $\mathbf{w}_{k,0}^{t+1} = \mathbf{w}_k^t$ , we have

$$

$$
\begin{array}{l} \mathbb {E} \left[ \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, R} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) \right] \leq \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t}; \boldsymbol {v} ^ {t}\right) + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2} \\ - \eta R \left(1 - \frac {\eta L _ {F}}{2}\right) \cdot \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \left\| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) \right\| ^ {2}. \tag {42} \\ \end{array}
$$

$$

Proof. This follows from the standard analysis of SGD on an  $L_{F}$ -smooth function. By smoothness, we have:

$$

$$
\begin{array}{l} \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r + 1} ^ {t + 1}\right) \leq \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) \\ + \langle \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k, r} ^ {t + 1}), \boldsymbol {w} _ {k, r + 1} ^ {t + 1} - \boldsymbol {w} _ {k, r} ^ {t + 1} \rangle \\ + \frac {L _ {F}}{2} \| \boldsymbol {w} _ {k, r + 1} ^ {t + 1} - \boldsymbol {w} _ {k, r} ^ {t + 1} \| ^ {2} \\ = \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) - \eta \langle \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right), \boldsymbol {d} _ {k, r} \rangle \\ + \frac {\eta^ {2} L _ {F}}{2} \| \boldsymbol {d} _ {k, r} \| ^ {2}, \tag {43} \\ \end{array}
$$

$$

where  $\pmb{d}_{k,r} = \nabla \hat{f}_k(\pmb{w}_{k,r}^{t+1};\mathcal{B}_{k,r}) + \lambda \nabla \tilde{g}_\gamma(\pmb{v}^t,\pmb{w}_{k,r}^{t+1}) + \mu \pmb{w}_{k,r}^{t+1}$  is the stochastic gradient. Using Assumption 3,  $\mathbb{E}[\| \pmb{d}_{k,r}\|^2] \leq \| \nabla \tilde{F}_k\|^2 + \sigma^2$ . Substituting this in gives:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r + 1} ^ {t + 1}\right) \right] \leq \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) - \eta \| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) \| ^ {2} \\ + \frac {\eta^ {2} L _ {F}}{2} \mathbb {E} [ \| \boldsymbol {d} _ {k, r} \| ^ {2} ] \\ \leq \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) - \eta \| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) \| ^ {2} \\ + \frac {\eta^ {2} L _ {F}}{2} \left(\| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) \| ^ {2} + \sigma^ {2}\right) \\ = \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) \\ - \eta \left(1 - \frac {\eta L _ {F}}{2}\right) \| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}\right) \| ^ {2} \\ + \frac {\eta^ {2} L _ {F} \sigma^ {2}}{2}. \tag {44} \\ \end{array}
$$

$$

Telescoping this inequality over  $r = 0, \dots, R - 1$  and taking the total expectation yields the desired result.

B.3. Main Convergence Proof (Theorem 1). We now present our main convergence result by analyzing the evolution of a carefully chosen Lyapunov function. In our personalized federated setting, a single global model does not exist. Instead, our goal is to show that the distributed optimization process is stable and converges to a meaningful equilibrium. To this end, we analyze the evolution of a global smoothed potential function

$$

$$
\Psi^ {t} = \sum_ {k = 1} ^ {K} p _ {k} \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t}; \boldsymbol {v} ^ {t}\right). \tag {45}
$$

$$

This function tracks the state of the entire distributed system, encompassing all personalized models  $\boldsymbol{w}_k^t$  and the server's consensus vector  $\boldsymbol{v}^t$ .

Theorem 1 (Convergence to a Stationary Neighborhood). Let assumptions 1-4 hold. For a learning rate  $\eta \leq \frac{1}{L_F}$ , after  $T$  rounds of Algorithm 1 where  $S$  out of  $K$  clients are sampled each round, we have

$$

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \mathbb {E} \left[ \sum_ {k = 1} ^ {K} p _ {k} \| \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}) \| _ {2} ^ {2} \right] \\ \leq \frac {\Psi^ {0} - F ^ {*}}{c _ {1} T} + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2 c _ {1}} + \frac {\Delta_ {\operatorname* {m a x}}}{c _ {1}} + \frac {\lambda E _ {S}}{c _ {1}} \tag {46} \\ \end{array}
$$

$$

with high probability, where  $c_{1} = \eta R(1 - \eta L_{F} / 2)$ . The error terms are defined as follows:

$$

$$
\begin{array}{l} E _ {S} = \\ \left(\frac {1}{T} \sum_ {t = 0} ^ {T - 1} 2 \sqrt {m} \sqrt {\frac {K - S}{S K (K - 1)} \sum_ {k = 1} ^ {K} \left\| z _ {k} ^ {t + 1} - \bar {z} ^ {t + 1} \right\| _ {2} ^ {2}}\right). \tag {47} \\ \end{array}
$$

$$

Remark 4. Note that the client sampling error  $E_{S}$  vanishes when  $S = K$  (full client participation). In this case, our convergence bound recovers the result for the full participation setting.

Proof. We analyze the one-round change in the potential function,  $\Psi^{t + 1} - \Psi^t$ . We decompose this change as:

$$

$$
\begin{array}{l} \mathbb {E} [ \Psi^ {t + 1} - \Psi^ {t} ] \\ = \mathbb {E} \left[ \underbrace {\sum_ {k} p _ {k} \left(\tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t + 1} ; \boldsymbol {v} ^ {t + 1}\right) - \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t + 1} ; \boldsymbol {v} ^ {t}\right)\right)} _ {=: \Gamma_ {1}} \right] \\ + \mathbb {E} \left[ \underbrace {\sum_ {k} p _ {k} \left(\tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t + 1} ; \boldsymbol {v} ^ {t}\right) - \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t} ; \boldsymbol {v} ^ {t}\right)\right)} _ {=: \Gamma_ {2}} \right]. \tag {48} \\ \end{array}
$$

$$

(1) Bounding the Client Progress. From Lemma 7, after taking expectations and weighting by  $p_k$ , the client progress is bounded by:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \Gamma_ {2} \right] = \mathbb {E} \left[ \sum_ {k} p _ {k} \left(\tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) - \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t}; \boldsymbol {v} ^ {t}\right)\right) \right] \\ \leq - c _ {1} \sum_ {k} p _ {k} \left(\frac {1}{R} \sum_ {r = 0} ^ {R - 1} \| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) \| ^ {2}\right) \\ + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2}, \tag {49} \\ \end{array}
$$

$$

where  $c_{1} = \eta R(1 - \frac{\eta L_{F}}{2})$

(2) Bounding the Server Progress. The server progress term reflects the change in the potential function due solely to the server updating its vector from  $\pmb{v}^t$  to  $\pmb{v}^{t + 1}$ . Based on the definition of  $\tilde{F}_k$ , we define:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \Gamma_ {1} \right] = \mathbb {E} \left[ \sum_ {k} p _ {k} \left(\tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t + 1}; \boldsymbol {v} ^ {t + 1}\right) - \tilde {F} _ {k} \left(\boldsymbol {w} _ {k} ^ {t + 1}; \boldsymbol {v} ^ {t}\right)\right) \right] \\ = \lambda \mathbb {E} \left[ \sum_ {k} p _ {k} \left(\langle \boldsymbol {v} ^ {t}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} \rangle - \langle \boldsymbol {v} ^ {t + 1}, \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} \rangle\right) \right] \\ = \lambda \mathbb {E} \left[ \left\langle \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1}, \sum_ {k} p _ {k} \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} \right\rangle \right]. \tag {50} \\ \end{array}
$$

$$

Here, we must acknowledge the mismatch:  $\tilde{F}_k$  uses  $\Phi \pmb{w}$  but the server update rule  $\pmb{v}^{t + 1}$  uses  $\mathrm{sign}(\Phi \pmb {w})$ . We bridge this gap by introducing the quantization error  $\delta_k^{t + 1} = \Phi \pmb{w}_k^{t + 1} - \mathrm{sign}(\Phi \pmb{w}^{t + 1})$ . Let  $\pmb{z}_k^{t + 1} = \mathrm{sign}(\Phi \pmb{w}_k^{t + 1})$  and  $\delta_k^{t + 1}$  be the quantization error. Let  $\hat{\pmb{z}}^{t + 1} = \sum_{k\in \mathcal{S}^t}p_k\pmb{z}_k^{t + 1}$  be the sampled aggregate and  $\bar{z}^{t + 1} = \sum_{k = 1}^{K}p_{k}\pmb{z}_{k}^{t + 1}$  be the ideal aggregate. The server update is  $\pmb{v}^{t + 1} = \mathrm{sign}(\hat{\pmb{z}}^{t + 1})$ . We decompose the inner product by introducing the sampled aggregate  $\hat{\pmb{z}}^{t + 1}$  and the quantization error:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \Gamma_ {1} \right] / \lambda = \mathbb {E} \left[ \left\langle \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1}, \sum_ {k = 1} ^ {K} p _ {k} \boldsymbol {z} _ {k} ^ {t + 1} + \sum_ {k = 1} ^ {K} p _ {k} \boldsymbol {\delta} _ {k} ^ {t + 1} \right\rangle \right] \\ = \mathbb {E} [ \underbrace {\left\langle \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1} , \hat {\boldsymbol {z}} ^ {t + 1} \right\rangle} _ {=: \Theta_ {1}} ] \\ + \mathbb {E} [ \underbrace {\left\langle \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1} , \bar {\boldsymbol {z}} ^ {t + 1} - \hat {\boldsymbol {z}} ^ {t + 1} \right\rangle} _ {=: \Theta_ {2}} ] \\ + \mathbb {E} \left[ \underbrace {\left\langle \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1} , \sum_ {k = 1} ^ {K} p _ {k} \boldsymbol {\delta} _ {k} ^ {t + 1} \right\rangle} _ {=: \Theta_ {3}} \right]. \tag {51} \\ \end{array}
$$

$$

Now we bound each term:

simplicity, assuming  $p_k = 1 / K$ :

$$

$$
\begin{array}{l} \mathbb {E} [ \Theta_ {2} ] \leq \mathbb {E} [ \| \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1} \| _ {2} \cdot \| \bar {\boldsymbol {z}} ^ {t + 1} - \hat {\boldsymbol {z}} ^ {t + 1} \| _ {2} ] \\ \leq 2 \sqrt {m} \sqrt {\mathbb {E} [ \| \bar {z} ^ {t + 1} - \hat {z} ^ {t + 1} \| _ {2} ^ {2} ]} \\ \leq 2 \sqrt {m} \sqrt {\frac {K - S}{S K (K - 1)} \sum_ {k = 1} ^ {K} \left\| z _ {k} ^ {t + 1} - \bar {z} ^ {t + 1} \right\| _ {2} ^ {2}}. \tag {52} \\ \end{array}
$$

$$

Since  $\| z_k\| _2^2 = m$ , the term  $\| z_{k} - \bar{z}\|^{2}$  is bounded.

- Term  $\Theta_3$ : Since the entries of  $v^t$  and  $v^{t+1}$  are  $\pm 1$ , the entries of their difference are in  $\{-2,0,2\}$ . This gives:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \Theta_ {3} \right] \leq \mathbb {E} \left[ \| \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1} \| _ {\infty} \left\| \sum_ {k} p _ {k} \delta_ {k} ^ {t + 1} \right\| _ {1} \right] \\ \leq 2 \cdot \mathbb {E} \left[ \left\| \sum_ {k} p _ {k} \boldsymbol {\delta} _ {k} ^ {t + 1} \right\| _ {1} \right] \\ (s i n c e \| \boldsymbol {v} ^ {t} - \boldsymbol {v} ^ {t + 1} \| _ {\infty} \leq 2) \\ \leq 2 \sum_ {k} p _ {k} \mathbb {E} \left[ \| \boldsymbol {\delta} _ {k} ^ {t + 1} \| _ {1} \right]. \tag {53} \\ \end{array}
$$

$$

As established in Lemma 4, the expected squared norm of the weights  $\mathbb{E}[\| \pmb{w}_k^{t + 1}\| _2^2 ]$  is uniformly bounded by a constant  $W^2$ . This allows us to bound the expected quantization error for each client:

$$

$$
\begin{array}{l} \mathbb {E} [ \| \boldsymbol {\delta} _ {k} ^ {t + 1} \| _ {1} ] = \mathbb {E} [ \| \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} - \operatorname {s i g n} (\boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1}) \| _ {1} ] \\ \leq \mathbb {E} [ \| \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} \| _ {1} ] + m \\ \leq \sqrt {m} \mathbb {E} [ \| \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} \| _ {2} ] + m \\ \leq \sqrt {m} \sqrt {\mathbb {E} [ \| \boldsymbol {\Phi} \boldsymbol {w} _ {k} ^ {t + 1} \| _ {2} ^ {2} ]} + m \\ \leq \sqrt {m} \sqrt {C _ {\Phi} ^ {2} \mathbb {E} [ \| \boldsymbol {w} _ {k} ^ {t + 1} \| _ {2} ^ {2} ]} + m \\ \leq \sqrt {m} C _ {\Phi} W + m. \tag {54} \\ \end{array}
$$

$$

Then we have

$$

$$
\mathbb {E} \left[ \Theta_ {3} \right] \leq 2 (\sqrt {m} C _ {\Phi} W + m). \tag {55}
$$

$$

We define  $\Delta_{\mathrm{max}} \triangleq 2\lambda (\sqrt{m} C_{\Phi}W + m)$ , so that this term is bounded by  $\Delta_{\mathrm{max}} / \lambda$ .

Combining these bounds, the total expected server progress is bounded by:

$$

$$
\begin{array}{l} \mathbb {E} [ \Gamma_ {1} ] \\ \leq 2 \lambda \sqrt {m} \sqrt {\frac {K - S}{S (K - 1)} \frac {1}{K} \sum_ {k = 1} ^ {K} \left\| z _ {k} ^ {t + 1} - \bar {z} ^ {t + 1} \right\| _ {2} ^ {2}} \\ + 2 \lambda (\sqrt {m} C _ {\Phi} W + m). \tag {56} \\ \end{array}
$$

$$

(3) Combining and Telescoping. Now, combining the bounds for the expected client and server progress, we get

the one-round recurrence:

$$

$$
\begin{array}{l} \mathbb {E} \left[ \Psi^ {t + 1} \right] \\ \leq \mathbb {E} [ \Psi^ {t} ] - c _ {1} \mathbb {E} \left[ \sum_ {k} p _ {k} \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \| \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}) \| _ {2} ^ {2} \right] \\ + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2} + \Delta_ {\max} \\ + \lambda \left(2 \sqrt {m} \sqrt {\frac {K - S}{S (K - 1)} \frac {1}{K} \sum_ {k = 1} ^ {K} \left\| z _ {k} ^ {t + 1} - \bar {z} ^ {t + 1} \right\| _ {2} ^ {2}}\right), \tag {57} \\ \end{array}
$$

$$

where  $c_{1}\triangleq \eta R(1 - \eta L_{F} / 2)$

Rearranging and summing over  $t = 0,\dots ,T - 1$  ..

$$

$$
\begin{array}{l} \sum_ {t = 0} ^ {T - 1} c _ {1} \mathbb {E} \left[ \sum_ {k} p _ {k} \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \| \nabla \tilde {F} _ {k} (\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}) \| _ {2} ^ {2} \right] \\ \leq \mathbb {E} [ \Psi^ {0} ] - \mathbb {E} [ \Psi^ {T} ] + T \left(\frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2} + \Delta_ {\max }\right) \\ + \lambda \sum_ {t = 0} ^ {T - 1} \left(2 \sqrt {m} \sqrt {\frac {K - S}{S K (K - 1)} \sum_ {k = 1} ^ {K} \left\| z _ {k} ^ {t + 1} - \bar {z} ^ {t + 1} \right\| _ {2} ^ {2}}\right). \tag {58} \\ \end{array}
$$

$$

Using the fact that  $\Psi^T \geq F^*(\mathrm{Assumption}2)$  and dividing by  $c_1 T$ , we arrive at the final result:

$$

$$
\begin{array}{l} \frac {1}{T} \sum_ {t = 0} ^ {T - 1} \mathbb {E} \left[ \sum_ {k} p _ {k} \frac {1}{R} \sum_ {r = 0} ^ {R - 1} \| \nabla \tilde {F} _ {k} \left(\boldsymbol {w} _ {k, r} ^ {t + 1}; \boldsymbol {v} ^ {t}\right) \| _ {2} ^ {2} \right] \\ \leq \frac {\Psi^ {0} - F ^ {*}}{c _ {1} T} + \frac {\eta^ {2} R L _ {F} \sigma^ {2}}{2 c _ {1}} + \frac {\Delta_ {\max}}{c _ {1}} \\ + \frac {\lambda}{c _ {1} T} \sum_ {t = 0} ^ {T - 1} 2 \sqrt {m} \sqrt {\frac {K - S}{S K (K - 1)} \sum_ {k = 1} ^ {K} \left\| z _ {k} ^ {t + 1} - \bar {z} ^ {t + 1} \right\| _ {2} ^ {2}}, \tag {59} \\ \end{array}
$$

$$

which completes the proof.

![](images/0fb742a45710e0bbb29e52e1e82bce1326e19decf578a1bf869311af543fd9ec.jpg)