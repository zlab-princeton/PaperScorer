
# ROBUST TRAINING WITH ENSEMBLE CONSENSUS


Anonymous authors


Paper under double-blind review


# ABSTRACT


Since deep neural networks are over-parametrized, they may memorize noisy examples. We address such memorizing issue in the presence of annotation noise. From the fact that deep neural networks cannot generalize neighborhoods of the features acquired via memorization, we observe that noisy examples do not consistently incur small losses on the network under a certain perturbation. Based on this observation, we propose a novel training method called Learning with Ensemble Consensus (LEC) that prevents overfitting noisy examples by eliminating them using the consensus of an ensemble of perturbed networks. One of the proposed LECs, LTEC outperforms the current state-of-the-art methods on MNIST, CIFAR-10, and CIFAR-100 with efficient memory usage.


# 1 INTRODUCTION


Deep neural networks (DNNs) have shown excellent performance (Krizhevsky et al., 2012; He et al., 2016) on visual recognition datasets (Deng et al., 2009). However, it is difficult to obtain annotated datasets of such high quality in practice (Wang et al., 2018a). Even worse, DNNs may not generalize training data in the presence of noisy examples (Zhang et al., 2016). Therefore, there is an increasing demand for robust training methods. In general, DNNs trained on noisy datasets first generalize clean examples (Arpit et al., 2017). Based on this, recent studies consider examples that incur small training losses in the early stage of training as being clean (Han et al., 2018; Shen & Sanghavi, 2019; Yu et al., 2019). However, small-loss examples may be highly corrupted especially when the training set contains a high level of annotation noise. Hence, choosing safe examples from the noisy training set with small-loss criteria can be impractical.


To address this, we find the method of screening out noisy examples among small-loss examples by focusing on well-known observations: (i) noisy examples are learned via memorization rather than via generalization and (ii) under a certain perturbation, network predictions for memorized features easily fluctuate, while those for generalized features do not. Based on these two observations, we hypothesize that out of small-loss examples, training losses of noisy examples would change dynamically by injecting certain perturbation, while those of clean examples would not. This suggests that clean examples can be found out of small-loss examples by selecting examples consistently incurring small losses on an ensemble of perturbed networks, i.e., ensemble consensus on small-loss examples. Since this idea comes from the difference between generalization and memorization, it can be applied to any architecture optimized with the gradient-based method.


In this work, we introduce simple perturbation methods to generate the ensemble for discriminating clean and noisy examples. By embedding those perturbation methods into training, we propose a new robust training scheme termed learning with ensemble consensus (LEC). In LEC, the network is first trained on the entire training set for a while and then trained on examples selected based on whether training losses are consistently small across the ensemble of perturbed networks. We present three LECs with different perturbations and evaluate their effectiveness on three benchmark datasets with random label noise (Goldberger & Ben-Reuven, 2016; Ma et al., 2018), open-set noise (Wang et al., 2018b), and semantic noise. The proposed LEC outperforms existing robust training methods by efficiently removing noisy examples from training batches.


# 2 RELATED WORK


Generalization of DNNs. Although DNNs are over-parametrized, they have impressive generalization ability (Krizhevsky et al., 2012; He et al., 2016). Some studies argue that gradient-based optimization plays an important role in regularizing DNNs (Neyshabur et al., 2014; Zhang et al., 2016). As a result, DNNs optimized with SGD first generalize and then memorize the training set (Krueger et al., 2017; Arpit et al., 2017). Since mislabeling reduces the correlation with other training examples, it is likely that noisy examples are learned via memorization. Therefore, we analyze the difference between generalized and memorized features to discriminate clean and noisy examples.


Training DNNs with Noisy datasets. Label noise issues can be addressed by reducing the impact of noisy examples. One direction is to train with a modified loss function based on the noise distribution. Most studies of this direction estimate the noise distribution prior to training as it is not accessible in general (Sukhbaatar et al., 2014; Goldberger & Ben-Reuven, 2016; Patrini et al., 2017; Hendrycks et al., 2018). Another direction is to train with modified labels using the current model prediction (Reed et al., 2014; Ma et al., 2018). Aside from these directions, recent work suggests a method of exploiting small-loss examples (Jiang et al., 2017; Han et al., 2018; Yu et al., 2019; Shen & Sanghavi, 2019). Their effectiveness comes from an artifact of SGD that a considerable portion of small-loss examples during generalization is well-annotated (Han et al., 2018). However, it is still hard to find clean examples based on training losses. This study presents a simple method to overcome such a problem of small-loss criteria.


# 3 ROBUST TRAINING WITH ENSEMBLE CONSENSUS


# 3.1 PROBLEM STATEMENT


In a situation where  $\epsilon \%$  of examples in a dataset  $\mathcal{D} \coloneqq \mathcal{D}_{clean} \cup \mathcal{D}_{noisy}$  are noisy, let  $S_{\epsilon, \mathcal{D}, \theta}$  denote the set of  $(100 - \epsilon)\%$  small-loss examples of the network parametrized by  $\theta$  out of examples in  $\mathcal{D}$ . Since it is generally hard to learn only all clean examples especially on the highly corrupted training set, it may be problematic to regard all examples in  $S_{\epsilon, \mathcal{D}, \theta}$  as being clean. To mitigate this, we suggest a simple idea: to find noisy examples among examples in  $S_{\epsilon, \mathcal{D}, \theta}$ .


# 3.2 LEARNING WITH ENSEMBLE CONSENSUS (LEC)


Since noisy examples are little correlated with other training examples, they are likely to be learned via memorization. However, DNNs cannot generalize neighborhoods of the memorized features. This means that even if training losses of noisy examples are small, they can be easily increased under a certain perturbation  $\delta$ , i.e., for  $(x,y) \in \mathcal{D}_{\text{noisy}}$ ,


$$

$$
(x, y) \in \mathcal {S} _ {\epsilon , \mathcal {D}, \theta} \Rightarrow (x, y) \notin \mathcal {S} _ {\epsilon , \mathcal {D}, \theta + \delta}.
$$

$$


Unlike noisy examples, some clean examples can be generalized even in the early stage of training with the entire set  $\mathcal{D}$ . Thus, training losses of those clean examples are consistently small in the presence of the certain perturbation  $\delta$ , i.e., for  $(x,y) \in \mathcal{D}_{clean}$ ,


$$

$$
(x, y) \in \mathcal {S} _ {\epsilon , \mathcal {D}, \theta} \Rightarrow (x, y) \in \mathcal {S} _ {\epsilon , \mathcal {D}, \theta + \delta}.
$$

$$


This suggests that noisy examples can be identified by using the inconsistency of losses under certain perturbation  $\delta$ . Based on this, we regard examples in the intersection of  $(100 - \epsilon)\%$  small-loss examples of an ensemble of  $M$  networks generated by perturbing parameters  $M$  times, i.e.,


$$

$$
\cap_ {m = 1} ^ {M} \mathcal {S} _ {\epsilon , \mathcal {D}, \theta + \delta_ {m}}
$$

$$


as being clean. We call it ensemble consensus filtering because examples are selected via ensemble consensus. Here, we can obtain such  $\theta$  by training on the entire set for a while. Considering all this, we propose a new robust training method termed learning with ensemble consensus (LEC) as described in Algorithms 1 and 2. Both algorithms consist of warming-up and filtering processes. The difference between these two lies in the selection of small-loss examples during the filtering process, i.e., a mini batch vs. a full batch. In Algorithm 1, the network is trained on the intersection of  $(100 - \epsilon)\%$  small-loss examples of  $M$  networks within a mini batch, thus the number of examples updated


at once is changing. We can encourage more stable training with fixed batchsize as described in Algorithm 2. At each epoch, we obtain the intersection of small-loss examples of  $M$  networks within a full batch. At each update, we sample a subset of batchsize from the intersection and train them like a normal SGD.


# ALGORITHM 1 LEC


Require: noisy dataset  $\mathcal{D}$  with noise ratio  $\epsilon \%$  duration of warming-up  $T_{m}$ , # of networks used for filtering  $M$ , perturbation  $\delta$


1: Initialize  $\theta$  randomly


2: for epoch  $t = 1 : {T}_{w}$  do


3: for mini-batch index  $b = 1 : \frac{|\mathcal{D}|}{\text{batchsize}}$  do


4: Sample a subset of batchsize  $\mathcal{B}_b$  from a full batch  $\mathcal{D}$


5:  $\theta \gets \theta -\alpha \nabla_{\theta}\frac{1}{|\mathcal{B}_b|}\sum_{(x,y)\in \mathcal{B}_b}CE(f_\theta (x),y)$


6: end for


7:er


8: for epoch  $t = T_w + 1:T_{end}$  do Filtering process


9: for mini-batch index  $b = 1 : \frac{|\mathcal{D}|}{\text{batchsize}}$  do


10: Sample a subset of batchsize  $\mathcal{B}_b$  from a full batch  $\mathcal{D}$


11: for  $m = 1:M$  do


12:  $\theta_{m} = \theta +\delta_{m,b,t}$   Adding perturbation


13:  $S_{\epsilon ,\mathcal{B}_b,\theta_m}\coloneqq (100 - \epsilon)\%$  small-loss examples of  $f_{\theta_m}$  within a mini batch  $\mathcal{B}_b$


14: end for


15:  $\mathcal{B}_b^{\prime} = \cap_{m = 1}^{M}S_{e,\mathcal{B}_b,\theta_m}$ $\triangleright$  Ensemble consensus


16: filtering  $\theta \gets \theta -\alpha \nabla_{\theta}\frac{1}{|\mathcal{B}_b^{\prime}|}\sum_{(x,y)\in \mathcal{B}_b^{\prime}}CE(f_{\theta}(x),y)$


17: end for


18: end for


# 3.3 PERTURBATION TO IDENTIFY NOISY EXAMPLES


Now the goal is to find a certain perturbation  $\delta$  that will be injected to distinguish between generalized and memorized features. We present three LECs with different perturbations in the following. The pseudocodes can be found in Section A.1.3.

- - Network-Ensemble Consensus (LNEC): Inspired by the observation that an ensemble of networks with the same architecture is correlated during generalization and is decorrelated during memorization (Morphos et al., 2018), the perturbation of prediction  $\delta$  comes from the difference between  $M$  networks. During the warming-up process,  $M$  networks are trained independently. During filtering process,  $M$  networks are trained on the intersection of  $(100 - \epsilon)\%$  small-loss examples of  $M$  networks.

- - Self-Ensemble Consensus (LSEC): We focus on the relationship between Morcos et al. (2018) and Lakshminarayanan et al. (2017): network predictions for memorized features are uncertain and those for generalized features are certain. Since the uncertainty of predictions also can be captured by multiple stochastic predictions (Gal & Ghahramani, 2016), the perturbation of prediction  $\delta$  comes from the difference between  $M$  stochastic predictions of a single network. $^{1}$  During the filtering process, the network is trained on the intersection of  $(100 - \epsilon)\%$  small-loss examples obtained with  $M$  stochastic predictions.

- - Temporal-Ensemble Consensus (LTEC): Inspired by the observation that during training, atypical features are more easily forgetful compared to typical features (Toneva et al., 2018), the perturbation of prediction  $\delta$  comes from the difference between predictions at the current and the preceding epochs. During the filtering process, the network is trained on the intersection of  $(100 - \epsilon)\%$  small-loss examples at the current epoch  $t$  and preceding  $\min(M - 1, t - 1)$  epochs. We collect  $(100 - \epsilon)\%$  small-loss examples at the preceding epochs, rather than network parameters to reduce memory usage.


# 4 EXPERIMENTS


In this section, we will show (i) the effectiveness of three perturbations at removing noisy examples from small-loss examples and (ii) the comparison of LEC and other existing methods under various annotation noises.


# 4.1 EXPERIMENTAL SETUP


Annotation noise. We study random label noise (Goldberger & Ben-Reuven, 2016; Ma et al., 2018), open-set noise (Wang et al., 2018b), and semantic noise. To generate these noises, we use MNIST (LeCun et al., 1998), CIFAR-10/100 (Krizhevsky et al., 2009) that are commonly used to assess robustness. For each benchmark dataset, we only corrupt its training set, while leaving its test set intact for testing. The details can be found in Section A.1.1.

- - Random label noise. Annotation issues can occur in easy images as well as hard images (Wang et al., 2018a). This is simulated in two ways: sym- $\epsilon\%$  and asym- $\epsilon\%$ . For sym- $\epsilon\%$ ,  $\epsilon\%$  of the entire set are randomly mislabeled to one of the other labels and for asym- $\epsilon\%$ , each label  $i$  of  $\epsilon\%$  of the entire set is changed to  $i + 1$ . We study four types: sym- $20\%$  and asym- $20\%$  for simulating a low level of noise, and sym- $60\%$  and asym- $40\%$  for simulating a high level of noise.

- - Open-set noise. In reality, annotated datasets may contain out-of-distribution (OOD) examples. As in Yu et al. (2019), to make OOD training examples, images of  $\epsilon \%$  examples randomly sampled from the original dataset are replaced with images from another dataset, while labels are left intact. SVHN (Netzer et al., 2011) is used to generate open-set noise for CIFAR-100, and ImageNet-32 (Chrabaszcz et al., 2017) and CIFAR-100 are used to generate open-set noise for CIFAR-10. Here, the similarity between original and open-set datasets is considered. We study two types:  $20\%$  and  $40\%$  open-set noise.

- - Semantic noise. In general, images with easy patterns are correctly labeled, while images with ambiguous patterns are obscurely mislabeled. To simulate this, we select the top  $\epsilon \%$  most uncertain images and then flip their labels to the confusing ones. Here, the uncertainty of each image is computed by the amount of disagreement between the predictions of five networks trained with clean dataset as in Lakshminarayanan et al. (2017). Then, the label of each image is assigned to the label with the highest value of averaged softmax outputs of the five networks trained with a clean dataset except for its groundtruth label. We study two types: 20% and 40% semantic noise.


Architecture and optimization. Unless otherwise specified, we use a variant of 9-convolutional layer architecture (Laine & Aila, 2016; Han et al., 2018). All parameters are trained for 200 epochs with Adam (Kingma & Ba, 2014) with a batch size of 128. The details can be found in Section A.1.2.


Hyperparameter. The proposed LEC involves three hyperparameters: duration of warming-up  $T_{w}$ , noise ratio  $\epsilon \%$ , and the number of predictions  $M$ . Unless otherwise specified,  $T_{w}$  is set to 10, and  $M$  is set to 5 for random label noise and open-set noise, and 10 for semantic noise. We assume that a noise ratio of  $\epsilon \%$  is given. Further study on hyperparameters can be found in Section 5.2.


Evaluation. We use two metrics: test accuracy and label precision (Han et al., 2018). At the end of each epoch, test accuracy is measured as the ratio of correctly predicted test examples to all test examples, and label precision is measured as the ratio of clean examples used for training to examples used for training. Thus, for both metrics, higher is better. For methods with multiple networks, the averaged values are reported. We report peak as well as final accuracy because a small validation set may be available in reality.


For each noise type, every method is run four times with four random seeds, e.g., four runs of Standard on CIFAR-10 with sym-20%. For each run of both random label noise and open-set noise, both a noisy dataset and network initialization are randomized. Here, four noisy datasets generated in four runs are the same for all methods. On the other hand, semantic noise is generated in a deterministic way. Thus, only network initialization is randomized for each run of semantic noise.


![Figure 1: Label precision (\%) of Self-training and three LECs on CIFAR-10 with random label noise. We plot the average as a solid line and the standard deviation as a shadow around the line.](images/5b2be97e873e213417013e4d8d7ec906f5505ab2c27e1c4d24242e01dae5f280.jpg)


Table 1: Average of final/peak test accuracy (%) of Self-training and three LECs on CIFAR-10 with random label noise. The best is highlighted in bold.


<table><tr><td>Dataset</td><td>Noise type</td><td>Self-training</td><td>LNEC</td><td>LSEC</td><td>LTEC</td></tr><tr><td rowspan="4">CIFAR-10</td><td>sym-20%</td><td>84.96/85.02</td><td>86.72/86.78</td><td>85.42/85.63</td><td>88.18/88.28</td></tr><tr><td>sym-60%</td><td>73.99/74.35</td><td>79.61/79.64</td><td>76.73/76.92</td><td>80.38/80.52</td></tr><tr><td>asym-20%</td><td>85.02/85.24</td><td>86.90/87.11</td><td>85.44/85.64</td><td>88.86/88.93</td></tr><tr><td>asym-40%</td><td>78.84/79.66</td><td>84.01/84.48</td><td>80.74/81.49</td><td>86.36/86.50</td></tr></table>


![Figure 2: Label precision (\%) of small-loss examples of the current network (in green) and the intersection of small-loss examples of the current and preceding networks (in red) during running LTEC on CIFAR-10 with random label noise. We report from epoch 11 when the filtering process starts.](images/a771664100513fabca0dea7543d122e8d96206fcf22d7129cb10ebe04bbd4641.jpg)


# 4.2 EFFECTIVENESS OF LECS AT IDENTIFYING NOISY EXAMPLES


Comparison with Self-training. In Section 3.1, we argue that  $(100 - \epsilon)\%$  small-loss examples may be corrupted. To show this, we run LEC with  $M = 1$  that is a method of training on  $(100 - \epsilon)\%$  small-loss examples. Note that this method is similar to the idea of Jiang et al. (2017); Shen & Sanghavi (2019). We call it Self-training for simplicity. Figure 1 shows the label precision of Self-training is low especially on the high level of noise, i.e., sym-60%. Compared to Self-training, three LECs are trained on a higher label precision of examples through the filtering process, achieving higher test accuracy as shown in Table 1. Out of these three, LTEC performs the best in both label precision and test accuracy.


Noisy examples are removed through ensemble consensus. In LTEC, we first obtain  $(100 - \epsilon)\%$  small-loss examples of the current network and then train only on the intersection of small-loss examples of the current and preceding networks. We plot label precisions of small-loss examples of the current network (in green) and the intersection (in red) at each epoch during running LTEC on CIFAR-10 with random noise in Figure 2. We observe that label precision of the intersection is always higher, indicating that noisy examples are removed through ensemble consensus filtering.


# 4.3 COMPARISON WITH STATE-OF-THE-ART METHODS


Competing methods. The competing methods include a regular training method: Standard, a method of training with corrected labels: D2L (Ma et al., 2018), a method of training with modified loss function based on the noise distribution: Forward (Patrini et al., 2017), and a method of exploiting small-loss examples: Co-teaching (Han et al., 2018). We tune all the methods individually as described in Section A.1.4.


![Figure 3: Test accuracy  $(\%)$  of different algorithms on MNIST/CIFAR with random label noise.](images/9537d75037f40d1c48465ddd2d516cca49d997841a33bce7eb9819b698ab04ed.jpg)


![Figure 4: Label precision  $(\%)$  of different algorithms on MNIST/CIFAR with random label noise.](images/e001127db46550bb60147dfcd55ada8edca5d087d1d7d83f98daf366b5108bb3.jpg)


Results on MNIST/CIFAR with random label noise. The overall results can be found in Figures 3 and 4, and Table 2. We plot the average as a solid line and the standard deviation as a shadow around the line. Figure 3 states that the test accuracy of D2L increases at the low level of label noise as training progresses, but it does not increase at the high level of label noise. This is because D2L puts large weights on given labels in the early stage of training even under the high level of noise. Forward shows its strength only in limited scenarios such as MNIST. Co-teaching does not work well on CIFAR-100 with asym-  $40\%$  , indicating that its cross-training scheme is vulnerable to smallloss examples of a low label precision (see Figure 4). Unlike Co-teaching, our methods attempt to remove noisy examples in small-loss examples. Thus, on CIFAR-100 with asym-  $40\%$  noise, both LTEC and LTEC-full surpass Co-teaching by a wide margin of about  $6 \%$  and  $5 \%$  , respectively.


Table 2: Average of final/peak test accuracy (%) of different algorithms on MNIST/CIFAR with random label noise. The best is highlighted in bold.


<table><tr><td>Dataset</td><td>Noise type</td><td>Standard</td><td>D2L</td><td>Forward</td><td>Co-teaching</td><td>LTEC</td><td>LTEC-full</td></tr><tr><td rowspan="4">MNIST</td><td>sym-20%</td><td>95.21/99.36</td><td>98.38/99.35</td><td>96.88/99.29</td><td>97.84/99.24</td><td>99.52/99.58</td><td>99.58/99.64</td></tr><tr><td>sym-60%</td><td>55.88/98.50</td><td>59.40/98.37</td><td>64.03/98.26</td><td>91.52/98.53</td><td>99.16/99.25</td><td>99.38/99.44</td></tr><tr><td>asym-20%</td><td>89.74/99.32</td><td>92.88/99.41</td><td>97.71/99.52</td><td>96.11/99.40</td><td>99.49/99.59</td><td>99.61/99.66</td></tr><tr><td>asym-40%</td><td>65.13/96.58</td><td>66.44/96.99</td><td>95.76/99.51</td><td>91.10/98.81</td><td>98.47/99.32</td><td>99.40/99.48</td></tr><tr><td rowspan="4">CIFAR-10</td><td>sym-20%</td><td>79.50/80.74</td><td>84.60/84.68</td><td>80.29/80.91</td><td>85.46/85.52</td><td>88.18/88.28</td><td>88.16/88.31</td></tr><tr><td>sym-60%</td><td>41.91/65.06</td><td>44.10/65.26</td><td>44.38/61.89</td><td>75.01/75.19</td><td>80.38/80.52</td><td>79.13/79.26</td></tr><tr><td>asym-20%</td><td>79.24/81.39</td><td>84.27/84.40</td><td>79.89/82.08</td><td>85.24/85.44</td><td>88.86/88.93</td><td>89.04/89.14</td></tr><tr><td>asym-40%</td><td>57.50/68.77</td><td>60.63/67.46</td><td>58.53/67.19</td><td>79.53/80.19</td><td>86.36/86.50</td><td>84.56/84.69</td></tr><tr><td rowspan="4">CIFAR-100</td><td>sym-20%</td><td>50.28/50.89</td><td>55.47/55.58</td><td>50.01/50.58</td><td>57.87/57.94</td><td>59.73/59.82</td><td>59.91/59.98</td></tr><tr><td>sym-60%</td><td>20.79/34.26</td><td>23.72/34.89</td><td>21.78/34.01</td><td>43.36/43.68</td><td>46.24/46.43</td><td>45.77/45.89</td></tr><tr><td>asym-20%</td><td>52.40/52.42</td><td>57.31/57.53</td><td>52.44/52.56</td><td>55.88/55.91</td><td>58.72/58.86</td><td>58.05/58.16</td></tr><tr><td>asym-40%</td><td>37.64/37.66</td><td>40.12/40.37</td><td>36.95/37.61</td><td>40.99/41.01</td><td>47.70/47.82</td><td>45.49/45.55</td></tr></table>


Table 3: Average of final/peak test accuracy (%) of different algorithms on CIFAR with open-set noise. The best is highlighted in bold.


<table><tr><td>Dataset + Open-set</td><td>Noise type</td><td>Standard</td><td>D2L</td><td>Forward</td><td>Co-teaching</td><td>LTEC</td><td>LTEC-full</td></tr><tr><td rowspan="2">CIFAR-10 + CIFAR-100</td><td>20%</td><td>86.74/86.83</td><td>89.42/89.49</td><td>86.87/86.96</td><td>88.58/88.61</td><td>88.69/88.82</td><td>89.07/89.11</td></tr><tr><td>40%</td><td>82.64/82.71</td><td>85.32/85.41</td><td>82.57/82.68</td><td>86.18/86.22</td><td>86.37/86.41</td><td>86.26/86.33</td></tr><tr><td rowspan="2">CIFAR-10 + ImageNet-32</td><td>20%</td><td>88.27/88.36</td><td>90.60/90.64</td><td>88.24/88.29</td><td>88.99/89.06</td><td>89.15/89.24</td><td>89.34/89.42</td></tr><tr><td>40%</td><td>85.90/85.99</td><td>87.91/87.95</td><td>85.84/85.99</td><td>86.99/87.03</td><td>86.63/86.78</td><td>87.00/87.12</td></tr><tr><td rowspan="2">CIFAR-100 + SVHN</td><td>20%</td><td>59.08/59.19</td><td>62.89/62.98</td><td>58.99/59.08</td><td>60.69/60.75</td><td>61.65/61.78</td><td>61.87/61.98</td></tr><tr><td>40%</td><td>53.32/53.35</td><td>56.30/56.38</td><td>53.18/53.30</td><td>56.45/56.52</td><td>56.95/57.18</td><td>57.77/57.90</td></tr></table>


Table 4: Average of final/peak test accuracy (%) of different algorithms on CIFAR with semantic noise. The best is highlighted in bold.


<table><tr><td>Dataset</td><td>Noise type</td><td>Standard</td><td>D2L</td><td>Forward</td><td>Co-teaching</td><td>LTEC</td><td>LTEC-full</td></tr><tr><td rowspan="2">CIFAR-10</td><td>20%</td><td>81.29/81.36</td><td>83.96/83.99</td><td>81.10/81.23</td><td>83.53/83.56</td><td>84.48/84.66</td><td>84.48/84.58</td></tr><tr><td>40%</td><td>71.64/74.36</td><td>74.72/74.94</td><td>71.38/73.47</td><td>76.61/76.89</td><td>75.52/76.52</td><td>76.57/78.06</td></tr><tr><td rowspan="2">CIFAR-100</td><td>20%</td><td>56.88/56.96</td><td>60.23/60.40</td><td>56.60/56.74</td><td>58.45/58.50</td><td>58.75/58.78</td><td>58.73/58.80</td></tr><tr><td>40%</td><td>49.56/49.69</td><td>53.04/53.19</td><td>49.57/49.69</td><td>52.96/52.98</td><td>52.58/52.78</td><td>53.15/54.18</td></tr></table>


Results on CIFAR with open-set noise. The overall results can be found in Table 3. All the methods including LTEC and LTEC-full perform well under open-set noise. We speculate that this is due to a low correlation between open-set noisy examples. This is supported by the results on CIFAR-10 where all the methods perform better on ImageNet-32 noise than on CIFAR-100 noise, as ImageNet-32 has more classes than CIFAR-100. Similar to poorly annotated examples, out-of-distribution examples are difficult to be generalized during the warming-up process. Therefore, they can be removed from training batches through ensemble consensus filtering.


Results on CIFAR with semantic noise. The overall results can be found in Table 4. The semantically generated noisy examples are highly correlated with each other, making it difficult to filter out those examples through ensemble consensus. Intuitively, using a bigger  $M$  makes ensemble consensus more conservative, filtering out more examples. Thus, we use 10 as the value of  $M$  for semantic noise. On CIFAR with semantic noise, LTEC and LTEC-full perform comparably or best, compared to the other methods. Between these two, LTEC-full performs better on  $40\%$  semantic noise due to its training stability.


# 5 DISCUSSION


# 5.1 HARD-TO-CLASSIFY BUT CLEAN EXAMPLES


It is hard to learn all clean examples during the warming-up process. Therefore, clean examples with high losses may be excluded from training batches during the filtering process. However, we expect that the number of clean examples used for training would increase gradually as training proceeds


![Figure 5: Recall (\%) of LTEC and LTEC-full on CIFAR-10 with random label noise. We plot the average as a solid line and the standard deviation as a shadow around the line.](images/9a3520c1cf6af50c29d54c98d3a2f4464d89b08a180327a3d58e306eeff7c281.jpg)


Table 5: Average of final/peak test accuracy (%) of LTEC with varying the number of networks used for filtering  $M$  . The best is highlighted in bold.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Noise type</td><td colspan="6">LTEC</td></tr><tr><td>M=1</td><td>M=3</td><td>M=5</td><td>M=7</td><td>M=9</td><td>M=∞</td></tr><tr><td rowspan="4">CIFAR-10</td><td>sym-20%</td><td>84.96/85.02</td><td>87.68/87.78</td><td>88.18/88.28</td><td>88.63/88.77</td><td>88.79/88.87</td><td>86.57/86.62</td></tr><tr><td>sym-60%</td><td>73.99/74.35</td><td>79.73/79.80</td><td>80.38/80.52</td><td>80.39/80.45</td><td>80.28/80.39</td><td>71.63/71.86</td></tr><tr><td>asym-20%</td><td>85.02/85.24</td><td>87.85/88.15</td><td>88.86/88.93</td><td>88.96/89.07</td><td>88.99/89.11</td><td>85.55/85.59</td></tr><tr><td>asym-40%</td><td>78.84/79.66</td><td>85.44/85.59</td><td>86.36/86.50</td><td>86.78/86.82</td><td>86.59/86.63</td><td>77.30/77.40</td></tr></table>


Table 6: Average of final/peak test accuracy (%) of Co-teaching and LTEC with noise ratio estimates (simulated). The best is highlighted in bold.


<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Noise type</td><td colspan="2">under-estimated (0.9ε)</td><td colspan="2">correctly estimated (ε)</td><td colspan="2">over-estimated (1.1ε)</td></tr><tr><td>Co-teaching</td><td>LTEC</td><td>Co-teaching</td><td>LTEC</td><td>Co-teaching</td><td>LTEC</td></tr><tr><td rowspan="4">CIFAR-10</td><td>sym-20%</td><td>84.51/84.58</td><td>87.93/88.08</td><td>85.46/85.52</td><td>88.18/88.28</td><td>86.40/86.45</td><td>88.72/88.75</td></tr><tr><td>sym-60%</td><td>70.47/73.11</td><td>77.98/78.22</td><td>75.01/75.19</td><td>80.38/80.52</td><td>79.15/79.17</td><td>79.34/79.45</td></tr><tr><td>asym-20%</td><td>84.61/84.73</td><td>88.15/88.39</td><td>85.24/85.44</td><td>88.86/88.93</td><td>86.41/86.57</td><td>89.04/89.22</td></tr><tr><td>asym-40%</td><td>76.14/77.41</td><td>84.42/84.52</td><td>79.53/80.19</td><td>86.36/86.50</td><td>82.19/82.63</td><td>86.93/86.96</td></tr></table>


since LEC allows the network to generalize clean examples without overfitting. To confirm this, we measure recall defined by the ratio of clean examples used for training to all clean examples at the end of each epoch during running LTEC and LTEC-full. As expected, recalls of both LTEC and LTEC-full sharply increase in the first 50 epochs as described in Figure 5. Pre-training (Hendrycks et al., 2019) prior to the filtering process may help in reducing the number of clean examples removed from training batches.


# 5.2 ABLATION STUDY


The number of networks used for filtering. During the filtering process of LEC, we use only the intersection of small-loss examples of  $M$  perturbed networks for training. Thus, the number of examples used for training highly depends on  $M$ . To understand the effect of  $M$ , we run LTEC with varying  $M$  on CIFAR-10 with random label noise. In particular, the range of  $M$  is  $\{1, 3, 5, 7, 9, \infty\}$ . Table 5 shows that a larger  $M$  is not always lead to better performance. This is because too many examples may be removed from training batches as  $M$  increases. Indeed, the total number of examples used for training is critical for the robustness as claimed in Rolnick et al. (2017); Li et al. (2017).


Noise ratio. In reality, only a poorly estimated noise ratio may be accessible. To study the effect of poor noise estimates, we run LTEC on CIFAR-10 with random label noise using a bit lower and higher values than the actual noise ratio as in Han et al. (2018). We also run Co-teaching that requires the noise ratio for comparison. The overall results can be found in Table 6. Since it is generally difficult to learn all clean examples, training on small-loss examples selected using the over-estimated ratio (i.e.,  $1.1\epsilon$ ) is often helpful in both Co-teaching and LTEC. In contrast, small-loss examples selected using the under-estimated ratio may be highly corrupted. In this case, LTEC is robust to the estimation error of noise ratio, while Co-teaching is not. Such robustness of LTEC against noise estimation error comes from ensemble consensus filtering.


Table 7: Average of final/peak test accuracy (%) of Standard and LTEC with ResNet. The best is highlighted in bold.


<table><tr><td>Dataset</td><td>Noise Type</td><td>Standard (ResNet)</td><td>LTEC (ResNet)</td></tr><tr><td rowspan="4">CIFAR-10</td><td>sym-20%</td><td>81.31/85.30</td><td>89.01/89.12</td></tr><tr><td>sym-60%</td><td>61.94/72.80</td><td>81.46/81.66</td></tr><tr><td>asym-20%</td><td>81.93/87.32</td><td>88.90/89.04</td></tr><tr><td>asym-40%</td><td>62.76/77.10</td><td>86.62/86.85</td></tr></table>


Applicability to different architecture. The key idea of LEC is rooted in the difference between generalization and memorization. Since this is an artifact of SGD optimization (Arpit et al., 2017), we expect that LEC can be applied to any architecture optimized with SGD. To support this, we run Standard and LTEC with ResNet-20 (He et al., 2016). The architecture is optimized based on Chollet et al. (2015), achieving the final test accuracy of  $90.67\%$  on clean CIFAR-10. Here,  $T_w$  is set to 30. Table 7 shows LTEC (ResNet) beats Standard (ResNet) in both peak and final accuracies, as expected.


# 6 CONCLUSION


This work presents the method of generating and using the ensemble for robust training. We explore three simple perturbation methods to generate the ensemble and then develop the way of identifying noisy examples using the ensemble consensus on small-loss examples. Along with growing attention to the use of small-loss examples for robust training, we expect that our ensemble method will be useful for such training methods.
