
# REINFORCEMENT LEARNING WITH COMPETITIVE ENSEMBLES OF INFORMATION-CONSTRAINED PRIMITIVEIVES


Anonymous authors


Paper under double-blind review


# ABSTRACT


Reinforcement learning agents that operate in diverse and complex environments can benefit from the structured decomposition of their behavior. Often, this is addressed in the context of hierarchical reinforcement learning, where the aim is to decompose a policy into lower-level primitives or options, and a higher-level meta-policy that triggers the appropriate behaviors for a given situation. However, the meta-policy must still produce appropriate decisions in all states. In this work, we propose a policy design that decomposes into primitives, similarly to hierarchical reinforcement learning, but without a high-level meta-policy. Instead, each primitive can decide for themselves whether they wish to act in the current state. We use an information-theoretic mechanism for enabling this decentralized decision: each primitive chooses how much information it needs about the current state to make a decision and the primitive that requests the most information about the current state acts in the world. The primitives are regularized to use as little information as possible, which leads to natural competition and specialization. We experimentally demonstrate that this policy architecture improves over both flat and hierarchical policies in terms of generalization.


# 1 INTRODUCTION


Learning policies that generalize to new environments or tasks is a fundamental challenge in reinforcement learning. While deep reinforcement learning has enabled training powerful policies, which outperform humans on specific, well-defined tasks (Mnih et al., 2015), their performance often diminishes when the properties of the environment or the task change to regimes not encountered during training.


This is in stark contrast to how humans learn, plan, and act: humans can seamlessly switch between different aspects of a task, transfer knowledge to new tasks from remotely related but essentially distinct prior experience, and combine primitives (or skills) used for distinct aspects of different tasks in meaningful ways to solve new problems. A hypothesis hinting at the reasons for this discrepancy is that the world is inherently compositional, such that its features can be described by compositions of small sets of primitive mechanisms (Parascandolo et al., 2017). Since humans seem to benefit from learning skills and learning to combine skills, it might be a useful inductive bias for the learning models as well.


This is addressed to some extent by the hierarchical reinforcement learning (HRL) methods, which focus on learning representations at multiple spatial and temporal scales, thus enabling better exploration strategies and improved generalization performance (Dayan & Hinton, 1993; Sutton et al., 1999b; Dietterich, 2000; Kulkarni et al., 2016). However, hierarchical approaches rely on some form of learned high-level controller, which decides when to activate different components in the hierarchy. While low-level sub-policies can specialize to smaller portions of the state space, the top-level controller (or master policy) needs to know how to deal with any given state. That is, it should provide optimal behavior for the entire accessible state space. As the master policy is trained on a particular state distribution, learning it in a way that generalizes to new environments effectively becomes the bottleneck for such approaches (Sasha Vezhnevets et al., 2017; Andreas et al., 2017).


![Figure 1: Illustration of our model (Left): An intrinsic competition mechanism, based on the amount of information each primitive provides, is used to select a primitive to be active for a given input. Each primitive focuses on distinct features of the environment; in this case, one policy focuses on boxes, a second one on gates, and the third one on spheres. Right: The primitive-selection mechanism of our model. The primitive with most information acts in the environment, and gets the reward.](images/4f56dab1b60557ebe40f606b85b5093ceb83319733af7d3487364b064eee65dd.jpg)


![](images/ca0fe885ba2707aca60c50dd8427c911c0111967ea2008de5bc409fd02835a00.jpg)


We argue, and empirically show, that in order to achieve better generalization, the interaction between the low-level primitives and the selection thereof should itself be performed without requiring a single centralized network that understands the entire state space. We, therefore, propose a decentralized approach as an alternative to standard HRL, where we only learn a set of low-level primitives without learning an explicit high-level controller. In particular, we construct a factorized representation of the policy by learning simple "primitive" policies, which focus on distinct regions of the state space. Rather than being gated by a single meta-policy, the primitives directly compete with one another to determine which one should be active at any given time, based on the degree to which their state encoders "recognize" the current state input. While, technically, the competition between primitives implicitly realizes a global selection mechanism, we consider our model "decentralized" in the sense that individual primitives can function on their own, and can be combined in new ways, without relying on an explicit high-level controller.


We frame the problem as one of information transfer between the current state and a dynamically selected primitive policy. Each policy can by itself decide to request information about the current state, and the amount of information requested is used to determine which primitive acts in the current state. Since the amount of state information that a single primitive can access is limited, each primitive is encouraged to use its resources wisely. Constraining the amount of accessible information in this way naturally leads to a decentralized competition and decision mechanism where individual primitives specialize in smaller regions of the state space. We formalize this information-driven objective based on the variational information bottleneck. The resulting set of competing primitives achieves both a meaningful factorization of the policy and an effective decision mechanism for which primitives to use. Importantly, not relying on a centralized meta-policy means that individual primitive mechanisms can be recombined in a "plug-and-play" fashion, and can be transferred seamlessly to new environments.


Contributions: In summary, the contributions of our work are as follows: (1) We propose a method for learning and operating a set of functional primitives in a decentralized way, without requiring an explicit high-level meta-controller to select active primitives (see Fig. 1 for illustration). (2) We introduce an information-theoretic objective, the effects of which are twofold: a) it leads to the specialization of individual primitives to distinct regions of the state space, and b) it enables a competition mechanism, which is used to select active primitives in a decentralized manner. (3) We demonstrate the superior transfer learning performance of our model, which is due to the flexibility of the proposed framework regarding the dynamic addition, removal, and recombination of primitives. Decentralized primitives can be successfully transferred to larger or previously unseen environments, and outperform models with an explicit meta-controller for primitive selection.


# 2 PRELIMINARIES


We consider a Markov decision process (MDP) defined by the tuple  $(S, \mathcal{A}, P, r, \gamma)$ , where the state space  $S$  and the action space  $\mathcal{A}$  may be discrete or continuous. The environment emits a bounded reward  $r: S \times \mathcal{A} \to [r_{min}, r_{max}]$  on each transition and  $\gamma \in [0,1)$  is the discount factor.  $\pi(|s)$  denotes a policy over the actions given the current state  $s$ .  $R(\pi) = \mathbb{E}_{\pi}[\sum_t \gamma^t r(s_t)]$  denotes the expected total return when the policy  $\pi$  is followed. The standard objective in reinforcement learning is to maximize the expected total return  $R(\pi)$ . We use the concept of the information bottleneck (Tishby et al., 2000) to learn compressed representations. The information bottleneck objective is formalized as minimizing the mutual information of a bottleneck representation layer with the input while maximizing its mutual information with the corresponding output. This type of input compression has been shown to improve generalization (Achille & Soatto, 2016; Alemi et al., 2016).


# 3 INFORMATION-THEORETIC LEARNING OF DISTINCT PRIMITIVEIS


Our goal is to learn a policy, composed of multiple primitive sub-policies, to maximize average returns over  $T$ -step interactions for a distribution of tasks. Simple primitives which focus on solving a part of the given task (and not the complete task) should generalize more effectively, as they can be applied to similar aspects of different tasks (subtasks) even if the overall objective of the tasks are drastically different. Learning primitives in this way can also be viewed as learning a factorized representation of a policy, which is composed of several independent policies.


Our proposed approach consists of three components: 1) a mechanism for restricting a particular primitive to a subset of the state space; 2) a competition mechanism between primitives to select the most effective primitive for a given state; 3) a regularization mechanism to improve the generalization performance of the policy as a whole. We consider experiments with both fixed and variable sets of primitives and show that our method allows for primitives to be added or removed during training, or recombined in new ways. Each primitive is represented by a differentiable, parameterized function approximator, such as a neural network.


# 3.1 PRIMITIVES WITH AN INFORMATION BOTTLENECK


To encourage each primitive to encode information from a particular part of state space, we limit the amount of information each primitive can access from the state. In particular, each primitive has an information bottleneck with respect to the input state, preventing it from using all the information from the state.


We define the overall policy as a mixture of experts,


$$

$$
\pi (a \mid s) = \sum_ {k} c _ {k} \pi^ {k} (a \mid s),
$$

$$


where  $c_k = \delta_{kk'}$  for  $k' \sim p(k' \mid s)$ . We denote the probability of selecting primitive  $k$  as  $\alpha_k(s) \coloneqq p(k \mid s)$ .


Rather than learning an explicit model for  $p(k \mid s)$ , however, we impose an information-based mechanism for selecting primitives, wherein we limit the amount of information each primitive can contain and select the ones that request the most information about the state. To implement an information bottleneck, we design each of the  $K$  primitives to be composed of an encoder  $p_{\mathrm{enc}}(z_k \mid s)$  and a decoder  $p_{\mathrm{dec}}(a \mid z_k)$ , together forming the primitive policy,


$$

$$
\pi_ {\theta} ^ {k} (a \mid s) = \int_ {z} p _ {\mathrm {e n c}} (z _ {k} \mid s) p _ {\mathrm {d e c}} (a \mid z _ {k}) \mathrm {d} z _ {k}. ^ {1}
$$

$$


The encoder output  $z_{k}$  is meant to represent the information about the current state  $s$  that an individual primitive  $k$  believes is important to access in order to perform well. The decoder takes this encoded information and produces a distribution over the actions  $a$ . Following the variational information bottleneck objective (Alemi et al., 2016), we penalize the KL divergence of  $p(z_k|s)$  and a prior  $p(z)$


$$

$$
\mathcal {L} _ {k} = \mathrm {D} _ {\mathrm {K L}} \left(p _ {\text {e n c}} \left(z _ {k} \mid s\right) \| p (z)\right). \tag {1}
$$

$$


In other words, a primitive pays an "information cost" proportional to  $\mathcal{L}_k$  for accessing the information about the current state.


In the experiments below, we fix the prior to be a unit Gaussian. In the general case, we can learn the prior as well and include its parameters in  $\theta$ . The information bottleneck encourages each primitive to limit its knowledge about the current state, but it will not prevent multiple primitives from specializing to similar parts of the state space. To mitigate this redundancy, and to make individual primitives focus on different regions of the state space, we introduce an information-based competition mechanism to encourage diversity among the primitives.


# 3.2 COMPETING INFORMATION-CONSTRAINED PRIMITIVES


We can use the information measure from equation 1 to define a selection mechanism for the primitives without having to learn a centralized meta-policy. The idea is that the information content of an individual primitive encodes its effectiveness in a given state  $s$  such that the primitive with the highest value  $\mathcal{L}_k$  should be activated in that particular state.


In particular, we set  $\alpha_{k} = Z^{-1}\exp (\beta \mathcal{L}_{k})$  to obtain a distribution over  $k$  as a function of the information content, activating the primitives with the highest information content. Here,  $Z = \sum_{k}\exp (\beta \mathcal{L}_{k})$  is a normalization constant. This mechanism enables competition between primitives, leading them to focus on parts of the state space that they "understand" well and letting others act in other parts.


Trading reward and information. To perform proper credit assignment, the environment reward is distributed to primitives according to their participation in the global decision, i.e. the reward  $r_k$  given to the  $k^{th}$  primitive is weighted by its selection coefficient, such that  $r_k = \alpha_k r$ , with  $r = \sum_k r_k$ . Hence, a primitive can potentially get a higher reward when deciding to act, but it also pays a higher price for accessing more information about the current state. The information bottleneck and the competition mechanism, when combined with the overall reward maximization objective, will lead to specialization of individual primitives to distinct regions in the state space. That is, each primitive should specialize in a part of the state space that it can reliably associate rewards with. Since the entire ensemble still needs to understand all of the state space for the given task, different primitives need to encode and focus on different parts of the state space.


# 3.3 REGULARIZING PRIMITIVE SELECTION


The objective described above will optimize the expected return while minimizing the information content of individual primitives. This is not sufficient, however, as it might lead to highly unbalanced outcomes: some primitives might be more active initially and learn to become even more active, completely disabling other primitives.


Thus, in addition to minimizing each primitive's absolute information content, we need to normalize their activity with respect to each other. To do so, we penalize their information content in proportion to their activation by adding a regularization term of the form


$$

$$
\mathcal {L} _ {\text {r e g}} = \sum_ {k} \alpha_ {k} \mathcal {L} _ {k}. \tag {2}
$$

$$


Note that this can be rewritten (see Appendix A) as  $\mathcal{L}_{\mathrm{reg}} = -H(\alpha) + \mathrm{LSE}(\mathcal{L}_1,\ldots ,\mathcal{L}_K)$ , where  $H(\alpha)$  is the entropy of  $\alpha$ , and LSE is the LogSumExp function,  $\mathrm{LSE}(x) = \log (\sum_{j}e^{x_{j}})$ . Thus, minimizing  $\mathcal{L}_{\mathrm{reg}}$  increases the entropy of  $\alpha$ , leading to a diverse set of primitive selections in turn, ensuring that different combinations of the primitives are used. Similarly, LSE approximates the maximum of its arguments,  $\mathrm{LSE}(x)\approx \max_jx_j$ , and, therefore, penalizes the dominating  $\mathcal{L}_k$  terms, thus equalizing their magnitudes.


# 3.4 OBJECTIVE AND ALGORITHM SUMMARY


Our overall objective function consists of 3 terms,


1. The expected return from the standard RL objective,  $R(\pi)$  which is distributed to the primitives according to their participation,

- 2. The individual bottleneck terms leading the individual primitives to focus on specific parts of the state space,  $\mathcal{L}_k$  for  $k = 1,\dots ,K$

- 3. The regularization term applied to the combined model,  $\mathcal{L}_{\mathrm{reg}}$


The overall objective for the  $k^{th}$  primitive thus takes the form:


$$

$$
J _ {k} (\theta) \equiv \mathbb {E} _ {\pi_ {\theta}} \left[ r _ {k} \right] - \beta_ {\text {i n d}} \mathcal {L} _ {k} - \beta_ {\text {r e g}} \mathcal {L} _ {\text {r e g}}, \tag {3}
$$

$$


where  $\mathbb{E}_{\pi_{\theta}}$  denotes an expectation over the state trajectories generated by the agent's policy,  $r_k = \alpha_k r$  is the reward given to the  $k$ th primitive, and  $\beta_{\mathrm{ind}}, \beta_{\mathrm{reg}}$  are the parameters controlling the impact of the respective terms.


Implementation: In our experiments, the encoders  $p_{\mathrm{enc}}(z_k \mid s)$  and decoders  $p_{\mathrm{dec}}(a \mid z_k)$  (see. Fig. 1) are represented by neural networks, the parameters of which we denote by  $\theta$ . Actions are sampled through each primitive every step. While our approach is compatible with any RL method, we maximize  $J(\theta)$  computed on-policy from the sampled trajectories using a score function estimator (Williams, 1992; Sutton et al., 1999a) specifically A2C (Mnih et al., 2016) (unless otherwise noted). Every experimental result reported has been averaged over 5 random seeds. Our model introduces 2 extra hyper-parameters  $\beta_{\mathrm{ind}}, \beta_{\mathrm{reg}}$ .


# 4 RELATED WORK


There are a wide variety of hierarchical reinforcement learning approaches(Sutton et al., 1998; Dayan & Hinton, 1993; Dietterich, 2000). One of the most widely applied HRL framework is the Options framework ((Sutton et al., 1999b)). An option can be thought of as an action that extends over multiple timesteps thus providing the notion of temporal abstraction or subroutines in an MDP. Each option has its own policy (which is followed if the option is selected) and the termination condition (to stop the execution of that option). Many strategies are proposed for discovering options using task-specific hierarchies, such as pre-defined sub-goals (Heess et al., 2017), hand-designed features (Florensa et al., 2017), or diversity-promoting priors (Daniel et al., 2012; Eysenbach et al., 2018). These approaches do not generalize well to new tasks. Bacon et al. (2017) proposed an approach to learn options in an end-to-end manner by parameterizing the intra-option policy as well as the policy and termination condition for all the options. Eigen-options (Machado et al., 2017) use the eigenvalues of the Laplacian (for the transition graph induced by the MDP) to derive an intrinsic reward for discovering options as well as learning an intra-option policy.


In this work, we consider a sparse reward setup with high dimensional action spaces. In such a scenario, performing unsupervised pretraining or using auxiliary rewards leads to much better performance (Frans et al., 2017; Florensa et al., 2017; Heess et al., 2017). Auxiliary tasks such as motion imitation have been applied to learn motor primitives that are capable of performing a variety of sophisticated skills (Liu & Hodgins, 2017; Peng et al., 2017; Merel et al., 2019b;a). Our work is also related to the Neural Module Network family of architectures (Andreas et al., 2017; Johnson et al., 2017; Rosenbaum et al., 2019) where the idea is to learn modules that can perform some useful computation like solving a subtask and a controller that can learn to combine these modules for solving novel tasks. More recently, Wu et al. (2019) proposed a framework for using diverse suboptimal world models to learn primitive policies. The key difference between our approach and all the works mentioned above is that we learn functional primitives without requiring any explicit high-level meta-controller or master policy.


# 5 EXPERIMENTAL RESULTS


In this section, we briefly outline the tasks that we used to evaluate our proposed method and direct the reader to the appendix for the complete details of each task along with the hyperparameters used for the model. The code is provided with the supplementary material. We designed experiments to address the following questions: a) Learning primitives – Can an ensemble of primitives be learned over a distribution of tasks? b) Transfer Learning using primitives – Can the learned primitives be transferred to unseen/unsolvable sparse environments? c) Comparison to centralized methods – How does our method compare to approaches where the primitives are trained using an explicit meta-controller, in a centralized way?


![Figure 2: Snapshots of motions learned by the policy. Top: Reference motion clip. Middle: Simulated character imitating the reference motion. Bottom: Probability of selecting each primitive.](images/b5e21f21fd0adf0fdc0df2d8b391ab3a16f49e1ebb588cc9123a544845df1cc5.jpg)


Baselines. We compare our proposed method to the following baselines: a) Option Critic (Bacon et al., 2017) - We extended the author's implementation $^2$  of the Option Critic architecture and experimented with multiple variations in the terms of hyperparameters and state/goal encoding. None of these yielded reasonable performance in partially observed tasks, so we omit it from the results. b) MLSH (Meta-Learning Shared Hierarchy) (Frans et al., 2017) - This method uses meta-learning to learn sub-policies that are shared across tasks along with learning a task-specific high-level master. It also requires a phase-wise training schedule between the master and the sub-policies to stabilize training. We use the MLSH implementation provided by the authors $^3$ . c) Transfer A2C: In this method, we first learn a single policy on the one task and then transfer the policy to another task, followed by fine-tuning in the second task.


# 5.1 LEARNING ENSEMBLES OF FUNCTIONAL PRIMITIVES


We evaluate our approach on a number of RL environments to show that we can indeed learn sets of primitive policies focusing on different aspects of a task and collectively solving it.


![Figure 3: Convergence of 4 primitives on four room: Left: We trained four primitives on 4 rooms task, where the goal was sampled from one of the two fixed goals. We see that the proposed algorithm is able to learn 4 primitives. Right: We transfer the learned primitives to the scenario where the goal is sampled from one of the four possible goals. The checkpointed model is ran on 100 different episodes (after fixed number of steps/updates) and the normalized frequency of activation of the different primitives is plotted.](images/c8d3642e836da2da7b6bb7fcf4d4ab5d13be4d7356d5c619c21cdf5a8f6d6406.jpg)


![](images/0998734617d3f1e8063452e7c5df75eb24134d155bd15711c6863202d81144a6.jpg)


Four Room Maze: We consider the Four-rooms gridworld environment (Sutton et al., 1999c) where the agent has to navigate its way through a grid of four interconnected rooms to reach a goal position within the grid. We consider the scenario where the starting position of the agent is fixed, but the goal is sampled from a discrete set. Fig. 3 shows that the proposed algorithm can learn four primitives. Refer to Appendix F for more details.


Motion Imitation. To evaluate the proposed method in terms of scalability, we present a series of tasks from the motion imitation domain, showing that we can use a set of distinct primitives for imitation learning. In these tasks, we train a simulated 2D biped character to perform a variety of highly dynamic skills by imitating motion capture clips recorded from human actors. Each mocap


clip is represented by a target state trajectory  $\tau^{*} = \{s_{0}^{*}, s_{1}^{*}, \dots, s_{T}^{*}\}$ , where  $s_{t}^{*}$  denotes the target state at timestep  $t$ . The input to the policy is augmented with a goal  $g_{t} = \{s_{t+1}^{*}, s_{t+2}^{*}\}$ , which specifies the target states for the next two timesteps. Both the state  $s_{t}$  and goal  $g_{t}$  are then processed by the encoder  $p_{\mathrm{enc}}(z_{t}|s_{t}, g_{t})$ . The repertoire of skills consists of 8 clips depicting different types of walks, runs, jumps, and flips. The motion imitation approach closely follows Peng et al. (2018). To analyze the specialization of the various primitives, we computed 2D embeddings of states and goals which each primitive is active in, and the actions proposed by the primitives. Fig. 4 illustrates the embeddings computed with t-SNE (van der Maaten & Hinton, 2008). The embeddings show distinct clusters for the primitives, suggesting a degree of specialization of each primitive to certain states, goals, and actions.


# 5.2 MULTI-TASK TRAINING


We evaluate our model in a partially-observable 2D multi-task environment called Minigrid, similar to the one introduced in (Chevalier-Boisvert et al., 2018). The environment is a two-dimensional grid with a single agent, impassable walls, and many objects scattered in the environment. The agent is provided with a natural language string that specifies the task that the agent needs to complete. The setup is partially observable and the agent only gets the small, egocentric view of the grid (along with the natural language task description). We consider three tasks here: the Pickup task (A), where the agent is required to pick


up an object specified by the goal string, the Unlock task (B) where the agent needs to unlock the door (there could be multiple keys in the environment and the agent needs to use the key which matches the color of the door) and the UnlockPickup task (C), where the agent first needs to unlock a door that leads to another room. In this room, the agent needs to find and pick up the object specified by the goal string. Additional implementation details of the environment are provided in appendix D. Details on the agent model can be found in appendix D.3.


We train agents with varying numbers of primitives on various tasks – concurrently, as well as in transfer settings. The different experiments are summarized in Figs. 5 and 7. An advantage of the multi-task setting is that it allows for quantitative interpretability as to when and which primitives are being used. The results indicate that a system composed of multiple primitives generalizes more easily to a new task, as compared to a single policy. We further demonstrate that several primitives can be combined dynamically and that the individual primitives respond to stimuli from new environments when trained on related environments.


# 5.3 DO LEARNED PRIMITIVES HELP IN TRANSFER LEARNING?


We evaluate our approach in the settings where the adaptation to the changes in the task is vital. The argument in favor of modularity is that it enables better knowledge transfer between related tasks. Naturally, transfer is easier when the tasks are closely related, as the model will only need to learn how to compose the already-learned primitives. In general, it is difficult to determine how closely related two tasks are, however, and the inductive bias of modularity could even be harmful if the two tasks are very different. In such cases, we could add new primitives (which would need to be learned) and still obtain a sample-efficient transfer, as some part of the task structure would already have been captured by the pretrained primitives. This approach can be extended towards adding primitives during training, providing a seamless way to combine knowledge about different tasks to solve more complex tasks. We investigate here the transfer properties of a primitive trained in one environment, and transferred to a different one. Results are shown in Fig. 5.


Continuous control for ant maze We evaluate the transfer performance of pretrained primitives on the cross maze environment (Haarnoja et al., 2018). Here, a quadrupedal robot must walk to the different goals along the different paths (see Appendix G for details). The goal is randomly chosen


![Figure 4: Embeddings visualizing the states (S) and goals (G) which each primitive is active in, and the actions (A) proposed by the primitives for the motion imitation tasks. A total of four primitives are trained. The primitives produce distinct clusters.](images/f18fbded05ad23044ee2120c1bb5e74174d2a09bdeafd1a992a9778030e32677.jpg)


![](images/fd67ecf2dc62f8a389af6a0710a88742689f1a255de47bec8cc08bd2cbffc717.jpg)


![](images/2ecbad1711af346b1e1d38a98397d2a458ecfb037744086fec9ede5857dabb03.jpg)


![Figure 5: Multitask training. Each panel corresponds to a different training setup, where different tasks are denoted A, B, C, ..., and a rectangle with  $n$  circles corresponds to an agent composed of  $n$  primitives trained on the respective tasks. Top row: activation of primitives for agents trained on single tasks. Bottom row: Retrain: two primitives are trained on A and transferred to B. The results (success rates) indicate that the multi-primitive model is substantially more sample efficient than the baseline (transfer A2C). Copy and Combine: more primitives are added to the model over time in a plug-and-play fashion (2 primitives are trained on A; the model is extended with a copy of itself; the resulting four-primitive model is trained on B.) This is more sample efficient than other strong baselines, such as (Frans et al., 2017; Bacon et al., 2017). Zero-Shot Generalization: A set of primitives is trained on C, and zero-shot generalization to A and B is evaluated. The primitives learn a form of spatial decomposition which allows them to be active in both target tasks, A and B. The checkpointed model is ran on 100 different episodes and the normalized frequency of activation of the different primitives is plotted.](images/3c3ebb2e69b5629fbbda5ec1d47bbc9a0a6d5884b6a0ca2abf45adb80cf64d3a.jpg)


from a set of available goals at the start of each episode. We pretrain a policy (see model details in Appendix G.1) with a motion reward in an environment which does not have any walls (similar to Haarnoja et al. (2018)), and then transfer the policy to the second task where the ant has to navigate to a random goal chosen from one of the 3 (or 10) available goal options. For our model, we make four copies of the pretrained policies and then finetune the model using the pretrained policies as primitives. We compare to both MLsH (Frans et al., 2017) and option-critic (Bacon et al., 2017). All these baselines have been pretrained in the same manner. As evident from Fig. 7, our method outperforms the other approaches. The fact that the initial policies successfully adapt to the transfer environment underlines the flexibility of our approach.


Zero Shot Generalization: The purpose of this experiment is to show that the model consisting of multiple primitives is somewhat able to decompose the task C into its subtasks, A and B. The better this decomposition the better should the model transfer to the individual subtasks. In order to test this, we trained a set of 4 primitives on task C, and then evaluate them (without retraining) on tasks A and B. We note that the ensemble is able to solve the transfer tasks, A and B, successfully  $72\%$  of the time, while a monolithic policy's success rate is  $38\%$ . This further shows that the primitives learn meaningful decompositions.


Continual Learning: 4 Rooms Scenario. We consider a continual learning scenario where we train 2 primitives for 2 goal positions, then transfer (and finetune) on 4 goal positions then transfer


![Figure 6: Continual Learning Scenario: The plot on the left shows that the primitives remain activated. The solid green line shows the boundary between the tasks. The plot on the right shows the number of samples taken by our model and the transfer baseline model across different tasks. We observe that the proposed model takes fewer steps than the baseline (an A2C policy trained in a similar way) and the gap in terms of the number of samples keeps increasing as tasks become harder. The checkpointed model is ran on 100 different episodes (after fixed number of steps/updates) and the normalized frequency of activation of the different primitives is plotted.](images/db25f40d42ebaaec04dbbf9477f0251c3b408a321a35eb0239cd2bb54f90a620.jpg)


![](images/c479b01fc8206143f24d07a42c5acac799e556f0dbdfa70206eac492c9bdfc4c.jpg)


![Figure 7: Left: Multitask setup, where we show that we are able to train 8 primitives when training on a mixture of 4 tasks. Here, the  $x$ -axis denotes the number of frames (timesteps). Right: Success rates on the different Ant Maze tasks. Success rate is measured as the number of times the ant is able to reach the goal (based on 500 sampled trajectories).](images/faaeefb42b588d523c1c9f072cdcfb0932d3678bdbc783f10799a25c1bf1a75e.jpg)


![](images/444ebbdc8fe477fbf26be57ff6c1869ce7ccc91db2602ae27511b2b64f7a3df0.jpg)


<table><tr><td>Method</td><td>3 goals</td><td>10 goals</td></tr><tr><td>Flat Policy (PPO)</td><td>11 ± 5 %</td><td>4 ± 2 %</td></tr><tr><td>Option critic</td><td>18 ± 10 %</td><td>7 ± 3 %</td></tr><tr><td>MLSH</td><td>32 ± 3 %</td><td>5 ± 3 %</td></tr><tr><td>Explicit high level policy</td><td>21 ± 5 %</td><td>11 ± 2 %</td></tr><tr><td>Proposed method</td><td>68 ± 3%</td><td>40 ± 3%</td></tr></table>


(and finetune) on 8 goals positions. The results are shown in fig. 6. The proposed method achieves better sample efficiency as compared to training a single monolithic policy.


# 6 SUMMARY AND DISCUSSION


We present a framework for learning an ensemble of primitive policies which can collectively solve tasks without learning an explicit master policy. Rather than relying on a centralized, learned meta-controller, the selection of active primitives is implemented through an information-theoretic mechanism. The learned primitives can be flexibly recombined to solve more complex tasks. Our experiments show that, on a partially observed "Minigrid" task and a continuous control "ant maze" walking task, our method can enable better transfer than flat policies and hierarchical RL baselines, including the Meta-learning Shared Hierarchies model and the Option-Critic framework. On Minigrid, we show how primitives trained with our method can transfer much more successfully to new tasks and on the ant maze, we show that primitives initialized from a pretrained walking control can learn to walk to different goals in a stochastic, multi-modal environment with nearly double the success rate of a more conventional hierarchical RL approach, which uses the same pretraining but a centralized high-level policy.
