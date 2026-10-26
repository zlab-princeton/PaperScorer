# DETAIL-PRESERVING TRANSFORMER FOR LIGHT FIELD IMAGE SUPER-RESOLUTION

Anonymous Submission

Anonymous Institute

# ABSTRACT

Recently, numerous algorithms have been developed to tackle the problem of light field super-resolution (LFSR), i.e., super-resolving low-resolution light fields to gain high-resolution views. Despite delivering encouraging results, these approaches are all convolution-based, and are naturally weak in global relation modeling of sub-aperture images necessarily to characterize the inherent structure of light fields. In this paper, we put forth a novel formulation built upon Transformers, by treating LFSR as a sequence-to-sequence reconstruction task. In particular, our model regards sub-aperture images of each vertical or horizontal angular view as a sequence, and establishes long-range geometric dependencies within each sequence via a spatial-angular locally-enhanced self-attention layer, which maintains the locality of each subaperture image as well. Additionally, to better recover image details, we propose a detail-preserving Transformer (termed as DPT), by leveraging gradient maps of light field to guide the sequence learning. DPT consists of two branches, with each associated with a Transformer for learning from an original or gradient image sequence. The two branches are finally fused to obtain comprehensive feature representations for reconstruction. Evaluations are conducted on a number of light field datasets, including real-world scenes and synthetic data. The proposed method achieves superior performance comparing with other state-of-the-art schemes. Our code is publicly available at: .

# INTRODUCTION

Light field (LF) imaging systems offer powerful capabilities to capture the 3D information of a scene, and thus enable a variety of applications going from photo-realistic image-based rendering to vision applications such as depth sensing, refocusing, or saliency detection. However, current light field cameras naturally face a trade-off between the angular and spatial resolution, that is, a camera capturing views with a high angular sampling typically at the expense of a limited spatial resolution, and vice versa. This limits the practical applications of LF, and also motivates many efforts to study super-resolution along the angular dimension (i.e., to synthesize new views) or the spatial dimension (i.e., to increase the spatial resolution). Our work focuses on the latter one.

In comparison with the traditional 2D photograph that only records the spatial intensity of light rays, a light field additionally collects the radiance of light rays along with different directions, offering a multi-view description of the scene. A naive solution of light field super-resolution (LFSR) is to super-resolve each view independently using single image super-resolution (SISR) techniques. However, despite the recent progress of SISR, the solution is sub-optimal mainly because it neglects the intrinsic relations (i.e., angular redundancy) of different light field views, possibly resulting in angularly inconsistent reconstructions. To address this, many studies exploit complementary information captured by different sub-aperture images for high-quality reconstruction. The seminal learning-based method, i.e., (Yoon et al. 2015), directly stacks 4-tuples of sub-aperture images together as an input of a SISR model. Subsequent efforts develop more advanced techniques, e.g., to explore the geometric property of a light field in multiple network branches (Zhang, Lin, and Sheng 2019), to align the features of the center view and its surrounding views with deformable convolutions (Wang et al. 2021d), to encourage interactions between spatial and angular features for more informative feature extraction (Wang et al. 2020), to combinatorially learn correlations between an arbitrary pair of views for super-resolution (Jin et al. 2020), or to gain an efficient low-rank light field representation for restoration (Farrugia and Guillemot 2019).

Despite the encouraging results of these approaches, they are all based on convolutional network architectures, thus inherently lack strong capabilities to model global relations among different views of light field images. In light of the ill-posed nature of the super-resolution problem, we believe that an ideal solution should take into account as much informative knowledge in the LR input as possible.

Motivated by the above analysis, we propose, to the best of our knowledge, the first Transformer-based model to address LFSR from a holistic perspective. In stark contrast to existing approaches, our model treats each light field as a collection of sub-aperture image (SAI) sequences (captured along horizontal or vertical directions), and exploits selfattention to reveal the intrinsic geometric structure of each sequence. Despite the advantages of Transformers in long-range sequence modeling, it is non-trivial to apply vanilla Transformers (e.g., ViT (Dosovitskiy et al. 2020)) for super-

resolution tasks, mainly because 1) they represent each input image with small-size (e.g.,  $16 \times 16$  or  $32 \times 32$ ) patches (i.e., tokens), which may damage fundamental structures (e.g., edges, corners or lines) in images, and 2) the vanilla fully connected self-attention design focus on establishing long-range dependencies among tokens, however, ignoring the locality in the spatial dimension. To address these problems, we introduce spatial-angular locally-enhanced self-attention (SA-LSA), which strengths locality using convolutions first, then promotes non-local spatial-angular dependencies of each SAI sequence.

Based on SA-LSA, we design a novel Transformer-based LFSR model, i.e., DPT, which simultaneously captures local structures within each SAI and global structures of all SAIs in the light field. Concretely, DPT consists of a content Transformer and a gradient Transformer to learn spatial-angular dependencies within each light field and the corresponding gradient images, respectively. It is also equipped with a cross-attention fusion Transformer to aggregate feature representations of the two branches, from which a high-resolution light field is reconstructed.

In a nutshell, our contributions are three-fold: 1) We reformulate the problem of LFSR from a sequence-to-sequence learning perspective, which is differentiated to prior works in a sense that it fully explores non-local contextual information among all sub-aperture images, better characterizing geometric structures of light fields; 2) We design a spatial-angular locally-enhanced self-attention layer, which, in comparison with its vanilla fully-connected counterparts, offers our Transformer a strong ability to maintain crucial local context within light fields; 3) We finally introduce DPT as a novel Transformer-based architecture, which not only mines non-local contexts from multiple views, but also preserving image details for each single view. Our DPT demonstrates the promising performance on multiple benchmarks, while maintaining similar network parameters and computational cost as existing convolution-based networks.

# RELATED WORK

Light Field Super-Resolution. Early deep learning-based methods usually respectively learn the spatial and angular information with two independent subnetworks: one subnetwork captures the spatial information and another one learns the angular information. For example, (Yoon et al. 2017) adopted the SRCNN (Dong et al. 2014) to separately process each SAI for spatial super-resolution and interpolated novel views for angular super-resolution with the angular super-resolution network. (Yuan, Cao, and Su 2018) utilized a single image super-resolution network to enlarge the spatial resolution of each SAI, and applied the proposed EPI enhancement network to restore the geometric consistency of different SAIs. Recently, many researchers seek to simultaneously capture the spatial and angular information with a unified framework. (Wang et al. 2018) built a horizontal and a vertical recurrent network to respectively super-resolve 3D LF data. (Zhang, Lin, and Sheng 2019) stacked the SAIs from different angular directions as inputs and sent them to a multi-branch network to capture the spatial and angular information. (Wang et al. 2021d) used the

deformable convolution (Dai et al. 2017) to align and aggregate the center-view and surrounding-view features to conduct LFSR. (Wang et al. 2020) developed a spatial-angular interaction network to learn the spatial-angular information from the macro-pixel image constructed with different SAIs. However, a major drawback of these approaches is that they fail to consider the long-range dependency among multiple SAIs in learning rich spatial-angular representations. To address this issue, we propose a Transformer based LFSR model, in which three proposed Transformers are leveraged to establish the non-local relationship of different SAIs for more effective representation learning.

Transformer for Image Super-Resolution. Attention-based models have demonstrated great successes in diverse vision tasks (Wang et al. 2021a; Zhou et al. 2020; Mou et al. 2021; Zhou et al. 2021a; Wang et al. 2021c,b; Zhou et al. 2021b) due to their powerful representative abilities. Some attention-based methods have been recently proposed to address the super-resolution tasks. These works can be roughly grouped into two categories: Transformer for single image super-resolution and Transformer for multiple image super-resolution. The former one mainly uses the Transformer to mine the intra-frame long-range dependency for high quality image reconstruction, i.e., IPT (Chen et al. 2021), SwinIR (Liang et al. 2021), and ESRT (Lu et al. 2021). The latter one adopts the Transformer to explore the inter-frame context information for accurate image reconstruction. i.e., TTSR (Yang et al. 2020) for reference-based image super-resolution and VSR (Cao et al. 2021) for video super-resolution. Motivated by these approaches, we devise the first Transformer based architecture for LFSR.

# OUR APPROACH

Problem Formulation. We treat the task of LFSR as a high-dimensional reconstruction problem, in which each LF is represented as a 2D angular collection of sub-aperture images (SAIs). Formally, considering an input low-resolution LF as  $L^{\mathrm{LR}}\in \mathbb{R}^{U\times V\times H\times W}$  with angular resolution of  $U\times V$  and spatial resolution of  $H\times W$ , LFSR aims at reconstructing a super-resolved LF  $L^{\mathrm{HR}}\in \mathbb{R}^{U\times V\times \alpha H\times \alpha W}$  with  $\alpha$  the upsampling factor. Following (Yeung et al. 2018; Zhang, Lin, and Sheng 2019; Wang et al. 2020, 2021d), we consider the case that SAIs distribute in a square array, i.e.,  $U = V = A$ , where  $A$  indicates the angular resolution along the horizontal or vertical direction.

Network Overview. The overall architecture of DPT is shown in Fig. 1. Given  $L^{\mathrm{LR}}$  as the network input, we compute a gradient map for each 2D SAI, and organize them together as a gradient field  $G^{\mathrm{LR}} \in \mathbb{R}^{A \times A \times H \times W}$ .  $L^{\mathrm{LR}}$  and  $G^{\mathrm{LR}}$  are separately fed into two small CNNs for convolutional feature extraction, following by two unimodal Transformers (i.e., a content Transformer and a gradient Transformer) to learn richer feature representations. To better learn relevant visual patterns among different SAIs, we treat  $L^{\mathrm{LR}}$  (or  $G^{\mathrm{LR}}$ ) as a collection of  $A$  horizontal and  $A$  vertical angular sequences, and each sequence includes  $A$  consecutive SAIs collected along one direction. Our content (or gradient) Transformer processes these sequences in  $L^{\mathrm{LR}}$  (or  $G^{\mathrm{LR}}$ )

![](images/278c2e461805b75a2523fedba3bba22b309bd24afc78bd7f3196a9b3ca981687.jpg)

one by one to avoid expensive computations as well as redundant interactions among irrelevant SAIs. Next, DPT aggregates the output features of the two Transformers via a cross-attention fusion Transformer, yielding a more comprehensive representation that is able to well preserve image details, which in final leads to better reconstruction.

In the following, we first provide a detailed description of DPT. Then, we elaborate on the proposed spatial-angular locally-enhanced self-attention layer, which is an essential component of our Transformer.

# DETAIL-PRESERVING TRANSFORMER (DPT)

Convolutional Feature Extraction. Given the light field input  $L^{\mathrm{LR}}$  as well as its gradient field  $\pmb{G}^{\mathrm{LR}}$ , two CNNs  $\mathcal{E}^{\mathrm{Cont}}$  and  $\mathcal{E}^{\mathrm{Grad}}$  are leveraged to separately extract

$$

$$
\boldsymbol {F} ^ {\text {C o n t}} = \mathcal {E} ^ {\text {C o n t}} \left(\boldsymbol {L} ^ {\text {L R}}\right) \quad \in \mathbb {R} ^ {A \times A \times C \times H \times W}, \tag {1}
$$

$$

$$

$$
\boldsymbol {F} ^ {\text {G r a d}} = \mathcal {E} ^ {\text {G r a d}} (\boldsymbol {G} ^ {\text {L R}}) \quad \in \mathbb {R} ^ {A \times A \times C \times H \times W},
$$

$$

per-SAI embeddings  $\pmb{F}^{\mathrm{Cont}}$  and  $\pmb{F}^{\mathrm{Grad}}$  at spatial resolution  $H \times W$  and with embedding channel  $C$ .

Content and Gradient Transformer. The convolutional feature embeddings (i.e.,  $F^{\mathrm{Cont}}$  and  $F^{\mathrm{Grad}}$ ) capture local context within each SAI independently but lack global context across different SAIs. We use Transformers (Dosovitskiy et al. 2020) to enrich the embeddings with sequence-level context. We start by learning unimodal contextualized representations  $T^{\mathrm{Cont}}$  and  $T^{\mathrm{Grad}}$  using a content Transformer  $T^{\mathrm{Cont}}$  and a gradient Transformer  $T^{\mathrm{Grad}}$ :

$$

$$
\boldsymbol {T} ^ {\text {C o n t}} = \mathcal {T} ^ {\text {C o n t}} (\boldsymbol {F} ^ {\text {C o n t}}) \quad \in \mathbb {R} ^ {A \times A \times C \times H \times W},
$$

$$

$$

$$
\boldsymbol {T} ^ {\text {G r a d}} = \mathcal {T} ^ {\text {G r a d}} \left(\boldsymbol {F} ^ {\text {G r a d}}\right) \in \mathbb {R} ^ {A \times A \times C \times H \times W}. \tag {2}
$$

$$

Note that the two transformers share the same network structure. For simplicity, we only describe  $\mathcal{T}^{\mathrm{Cont}}$  in the following.

The  $\mathcal{T}^{\mathrm{Cont}}$  is comprised of  $K$  spatial-angular attention blocks. Each block includes two consecutive SA-LSA layers (detailed in the next section), which exploit global relations within each horizontal or vertical sequence of the input LF image, respectively.

In particular, in the  $k$ -th block, we treat its input as a set of horizontal (or row-wise) sequences, i.e.,  $\mathcal{R} = \{\pmb{R}_i \in \mathbb{R}^{A \times C \times H \times W}\}_{i=1}^A$ , where  $\pmb{R}_i$  indicates a sequence of convolutional features, corresponding to the  $i$ -th row in the input. The first horizontal SA-LSA layer  $\mathcal{H}_k^1$  aims to explore the dependencies within each sequence independently. Specifically, for each horizontal sequence  $\pmb{R}_i$ , we obtain a non-local representation  $\hat{\pmb{R}}_i$  as follows:

$$

$$
\hat {\boldsymbol {R}} _ {i} = \mathcal {H} _ {k} ^ {1} (\boldsymbol {R} _ {i}) \quad \in \mathbb {R} ^ {A \times C \times H \times W}. \tag {3}
$$

$$

After processing all horizontal sequences in  $\mathcal{R}$ , we obtain a horizontal-enhanced content representation for the light field:

$$

$$
\boldsymbol {R} _ {k} ^ {\text {C o n t}} = \left[ \hat {\boldsymbol {R}} _ {1}, \hat {\boldsymbol {R}} _ {2}, \dots , \hat {\boldsymbol {R}} _ {A} \right] \quad \in \mathbb {R} ^ {A \times A \times C \times H \times W}, \tag {4}
$$

$$

where  $[ ]$  denotes the concatenation operation.

Next, the second vertical SA-LSA layer  $\mathcal{H}_k^2$  accepts  $R_{k}^{\mathrm{Cont}}$  as input, and explores the long-range relations of vertical (or column-wise) sequences, i.e.,  $\mathcal{C} = \{\pmb{T}_i\in \mathbb{R}^{A\times C\times H\times W}\}_{i = 1}^A$  where  $\pmb{T}_i$  indicates sequences of convolutional features, corresponding to the  $i$ -th column in  $R_{k}^{\mathrm{Cont}}$ . Similarly, the vertical sequences in  $\mathcal{C}$  are transformed via  $\mathcal{H}_k^2$  to produce a vertical-enhanced content representation  $T_{k}^{\mathrm{Cont}}$ :

$$

$$
\boldsymbol {T} _ {k} ^ {\text {C o n t}} = \left[ \hat {\boldsymbol {T}} _ {1}, \hat {\boldsymbol {T}} _ {2}, \dots , \hat {\boldsymbol {T}} _ {A} \right] \quad \in \mathbb {R} ^ {A \times A \times C \times H \times W}, \tag {5}
$$

$$

where  $\hat{T_i} = \mathcal{H}_k^2 (T_i)$  is the non-local representation of  $T_{i}$ . Our content Transformer uses  $K$  spatial-angular attention blocks to aggregate informative contextual knowledge among SAIs, eventually yielding a more comprehensive content representation  $T_K^{\mathrm{Cont}}$ .

Our gradient Transformer  $\mathcal{T}^{\mathrm{Grad}}$  performs in a similar way as the content Transformer. It accepts  $F^{\mathrm{Grad}}$  as its input, and utilizes  $K$  spatial-angular attention blocks to deliver a gradient representation  $T_K^{\mathrm{Grad}}$ .

Cross-Attention Fusion Transformer. While the content and gradient Transformers process each modality separately, we design a fusion Transformer to aggregate together their representations. To obtain more comprehensive unimodal representations, we obtain the inputs of fusion Transformer

by combining all intermediate features in the content or gradient Transformers:

$$

$$
\boldsymbol {H} ^ {\text {C o n t}} = \left[ \boldsymbol {F} ^ {\text {C o n t}}, \boldsymbol {T} _ {1} ^ {\text {C o n t}}, \dots , \boldsymbol {T} _ {K} ^ {\text {C o n t}} \right] \in \mathbb {R} ^ {A \times A \times (K + 1) C \times H \times W}, \tag {6}
$$

$$

$$

$$
\boldsymbol {H} ^ {\text {G r a d}} = \left[ \boldsymbol {F} ^ {\text {G r a d}}, \boldsymbol {T} _ {1} ^ {\text {G r a d}}, \dots , \boldsymbol {T} _ {K} ^ {\text {G r a d}} \right] \in \mathbb {R} ^ {A \times A \times (K + 1) C \times H \times W},
$$

$$

Different from  $\mathcal{T}^{\mathrm{Cont}}$  and  $\mathcal{T}^{\mathrm{Grad}}$  which capture non-local dependencies of tokens within the same sequence, our fusion Transformer aims to explore the relations between tokens of two sequences. In particular, for the  $i$ -th horizontal (or vertical) sequences in  $\pmb{H}^{\mathrm{Cont}}$  and  $\pmb{H}^{\mathrm{Grad}}$ , i.e.,  $\pmb{U}_i \in \mathbb{R}^{A \times (K + 1)C \times H \times W}$  and  $\pmb{V}_i \in \mathbb{R}^{A \times (K + 1)C \times H \times W}$ , the  $\mathcal{T}^{\mathrm{Fuse}}$  achieves a detail-preserved representation  $\pmb{Z}_i$  as:

$$

$$
\boldsymbol {Z} _ {i} = \mathcal {T} ^ {\text {F u s e}} \left(\boldsymbol {U} _ {i}, \boldsymbol {V} _ {i}\right) \in \mathbb {R} ^ {A \times (K + 1) C \times H \times W}. \tag {7}
$$

$$

The  $\mathcal{T}^{\mathrm{Fuse}}$  performs cross-attention between its inputs, with query generated from  $U_{i}$ , key and value from  $V_{i}$  to gather high-frequency information as a compensation of the content representation. We concatenate the outputs  $\{Z_i\}_i$  together to obtain the fusion output  $Z\in \mathbb{R}^{A\times A\times (K + 1)C\times H\times W}$ . Note that our fusion Transformer only includes one spatial-angular attention block, and we see minor performance improvement when adding additional blocks.

SAI Reconstruction. Finally, we leverages a reconstruction module  $\mathcal{D}^{\mathrm{Reco}}$  over  $Z$  to obtain a high-resolved LF  $L^{\mathrm{HR}}$ :

$$

$$
\boldsymbol {L} ^ {\mathrm {H R}} = \mathcal {D} ^ {\operatorname {R e c o}} (\boldsymbol {Z}) \quad \in \mathbb {R} ^ {A \times A \times \alpha H \times \alpha W}. \tag {8}
$$

$$

Here,  $\mathcal{D}^{\mathrm{Reco}}$  is separately applied to each SAI.

Remark. Our DPT employs Transformers to enrich convolutional features  $F^{\mathrm{Cont}}$  and  $F^{\mathrm{Grad}}$  (Eq. 2) into richer representations  $H^{\mathrm{Cont}}$  and  $H^{\mathrm{Grad}}$ , respectively, which are further aggregated together via a fusion Transformer. In this manner, our network is able to collect informative non-local contexts within each SAI and across different SAIs, allowing for higher-quality reconstruction.

In addition, our Transformers process each sequence (or sequence fusion) independently rather than directly process all sequences together. This enables our model to meet hardware resource constraints, and more importantly, avoid redundant interactions among irrelevant sub-aperture images.

In Transformer architectures (e.g., ViT (Dosovitskiy et al. 2020)), fully-connected self-attention is extensively employed to explore non-local interactions among input tokens. However, the vanilla self-attention layer neglects local spatial context within each input token, which is crucial for image reconstruction. To remedy this limitation, we introduce a spatial-angular locally-enhanced self-attention layer to offer our Transformer a strong capability in modeling locality.

# SPATIAL-ANGULAR LOCALLY-ENHANCED SELF-ATTENTION

Inspired by VSR (Cao et al. 2021), our SA-LSA layer includes three sequential operations: spatial-angular convolutional tokenization, spatial-angular self-attention, as well as spatial-angular convolutional de-tokenization. Its structure is illustrated in Fig. 2 (a).

Spatial-Angular Convolutional Tokenization. ViT-like architectures (Dosovitskiy et al. 2020) leverage a linear projection layer to achieve input tokens at a very early stage.

![](images/770511a5a4c48e53a88135b120d5d62d15d73a952ef6d557d32e14958e19b939.jpg)

![](images/e5405ed307118447f491c5830b25d46053e9d77bafe50ccdc2265c6ba12c71bb.jpg)

In contrast, our proposed convolutional tokenization mechanism obtains the tokens in the self-attention layer, yielding a multi-stage hierarchy like CNNs. This allows our Transformer to capture local contexts from low-level features to high-level semantic representations.

In particular, denote  $\pmb{F} \in \mathbb{R}^{A \times C \times H \times W}$  as the spatial-angular representation of an angular sequence, where  $A$  is the length of the sequence,  $C$  denotes feature dimension, and  $H \times W$  is the spatial dimension. The convolutional tokenization module generates query  $Q$ , key  $K$  and value  $V$  at each self-attention layer as follows:

$$

$$
\boldsymbol {Q} = \mathcal {U} \left(f _ {Q} (\boldsymbol {F})\right), \quad \boldsymbol {K} = \mathcal {U} \left(f _ {K} (\boldsymbol {F})\right), \quad \boldsymbol {V} = \mathcal {U} \left(f _ {V} (\boldsymbol {F})\right). \tag {9}
$$

$$

It produces the inputs of a self-attention layer via two steps. First, the input  $\mathbf{F}$  is fed into three independent convolutional layers (i.e.,  $f_{Q}, f_{K}$  and  $f_{V}$ ). We implement each layer with kernel size  $1 \times 1$ .

Next, a function  $\mathcal{U}$  is designed to obtain spatial-angular tokens. In particular, it extracts a collection of overlapping patches with size  $H_{p} \times W_{p}$ . Thus, we are able to obtain a sequence  $\mathbf{X} \in \mathbb{R}^{n \times d}$  of  $n$  tokens, and each token with a feature dimension  $d = C \times H_{p} \times W_{p}$ .

Spatial-Angular Self-Attention. The fully-connected selfattention layer is applied on  $X$  to explore the non-local spatial-angular relations among the tokens as follows:

$$

$$
\boldsymbol {X} ^ {\prime} = \text {A t t e n t i o n} (\boldsymbol {Q}, \boldsymbol {K}, \boldsymbol {V}) = \operatorname {s o f t m a x} \left(\boldsymbol {Q} \boldsymbol {K} ^ {\top}\right) \boldsymbol {V}, \tag {10}
$$

$$

where Attention denotes a standard self-attention layer as in (Vaswani et al. 2017; Dosovitskiy et al. 2020).

Spatial-Angular Convolutional De-Tokenization. To enable the application of convolutional tokenization in each self-attention layer, we further de-tokenize the attended feature sequence  $\mathbf{X}^{\prime}$  to fold the patches into a large feature map  $\hat{\mathbf{F}}$  with the same dimension as  $\mathbf{F}$ :

$$

$$
\hat {\boldsymbol {F}} = f _ {P} \left(\mathcal {P} \left(\boldsymbol {X} ^ {\prime}\right)\right) + \boldsymbol {F} \quad \in \mathbb {R} ^ {A \times C \times H \times W}, \tag {11}
$$

$$

where  $\mathcal{P}$  denotes the de-tokenization operation, and  $f_{P}$  is a convolutional layer as Eq. 9. A residual layer is also used to avoid the loss of important features. Here,  $\hat{\pmb{F}}$  well encodes the local (in the spatial dimension) and global (in both spatial and angular dimensions) context of the input angular sequence, which could be expected to help produce better reconstruction results.

Note that in the fusion Transformer, we improve SA-LSA into a cross-attention SA-LSA layer to support the computation of cross-attention between the two modalities. Fig. 2 shows the detailed structure.

# DETAILED NETWORK ARCHITECTURE

Convolutional Feature Extraction. The convolutional modules  $\mathcal{E}^{\mathrm{Cont}}$  and  $\mathcal{E}^{\mathrm{Grad}}$  (Eq. 2) share a similar network structure. Following (Wang et al. 2021d), each of them consists of two residual blocks and two residual atrous spatial pyramid pooling blocks organized in an intertwine manner, following by an angular alignment module without using deformable convolution to align the convolutional features of different views.

Transformer Networks. In the content and gradient Transformers, we set the number of attention blocks  $K$  to 2 by default. For  $T^{\mathrm{Fuse}}$ , Eq. 9 is slightly modified to enable the exploration of cross-attention by generating query from the content representation, while key and value from the gradient representation.

Reconstruction Module. The reconstruction module  $\mathcal{D}^{\mathrm{Reco}}$  (Eq. 8) is implemented as a cascaded of five information multi-distillation blocks, following with a upsampling layer, which consists of a pixel shuffle operation combined with two  $1\times 1$  convolution layers, to generate the super-resolved SAIs (Wang et al. 2021d).

# EXPERIMENT

# DATASETS AND EVALUATION METRICS

We conduct extensive experiments on five popular LFSR benchmarks, i.e., EPFL (Rerabek and Ebrahimi 2016), HCInew (Honauer et al. 2016), HCIold (Wanner, Meister, and Goldluecke 2013), INRIA (Le Pendu, Jiang, and Guillemot 2018), and STFgantry (Vaish and Adams 2008). All of the light field images from the above benchmarks have a  $9 \times 9$  angular resolution (i.e.,  $U = V = A = 9$ ). PSNR and SSIM are chosen as the evaluation metrics. For a testing dataset with  $T$  scenes, we first obtain the metric values of  $U \times V$  sub-aperture images, and average the summation results of  $T \times U \times V$  to obtain the final metric score.

# IMPLEMENTATION DETAILS

Following (Wang et al. 2021d), we convert the light field images from the RGB space to the YCbCr space. Our model only super-resolves Y channel images, and uses the bicubic interpolation to super-resolve Cb and Cr channels images, respectively. The gradient maps are extracted from Y channel of each SAI along the spatial dimension with the function provided by (Ma et al. 2020). We perform  $\times 2$  and  $\times 4$  SR with  $5\times 5$  angular resolution on all benchmarks. In the training stage, the  $64\times 64$  patches are cropped from each

sub-aperture image, and we apply the bicubic interpolation to generate  $\times 2$  and  $\times 4$  patches. We use random horizontal rotation, vertical rotation and  $90^{\circ}$  rotation to augment the training data. Spatial and angular resolution are processed simultaneously for preserving the LF structure. The  $\ell_1$  loss is used to optimize our network. We use the Adam optimizer to train our network, with a batch size of 8. The initial learning rate is set to  $2\times 10^{-4}$  and it will be halved every 15 epochs. We train the network for 75 epochs in total. All experiments are carried out on a single Tesla V100 GPU card.

# COMPARISONS WITH STATE-OF-THE-ART

To evaluate the effectiveness of DPT, we compare it with the several state-of-the-art LFSR methods (i.e., LFBM5D (Alain and Smolic 2018), GB (Rossi and Frossard 2018), resLF (Zhang, Lin, and Sheng 2019), LFSSR (Yeung et al. 2018), LF-InterNet (Wang et al. 2020), and LF-DFNet (Wang et al. 2021d)) and single image super-resolution methods ( i.e., VDSR (Kim, Lee, and Lee 2016), EDSR (Lim et al. 2017), and RCAN (Zhang et al. 2018)) on five LFSR benchmarks. Following (Wang et al. 2021d), we treat Bicubic upsampling as the baseline.

Quantitative Results. The quantitative performance comparisons are reported in Table 1. As seen, DPT achieves a promising performance in comparison with other methods. Single image super-resolution methods (i.e., VDSR (Kim, Lee, and Lee 2016), EDSR (Lim et al. 2017) and RCAN (Zhang et al. 2018)) super-resolve each view separately. DPT outperforms all of them, which can be attributed to its outstanding capability to capture the complementary information of different views.

Furthermore, compared with other CNN-based methods, e.g., resLF (Zhang, Lin, and Sheng 2019), LFSSR (Yeung et al. 2018), LF-InterNet (Wang et al. 2020), our DPT outperforms all of them for  $\times 4$  SR. The reason lies in that these methods are still weak in modeling global relations of different views. In contrast, our DPT can establish the global relations among different SAIs with the Transformer for efficient spatial-angular representations.

Specifically, compared with the current leading approach LF-DFNet (Wang et al. 2021d), DPT obtains superior results on EPFL (Rerabek and Ebrahimi 2016), HCIold (Wanner, Meister, and Goldluecke 2013) and INRIA (Le Pendu, Jiang, and Guillemot 2018) for  $\times 4$  SR. The reason is that, LF-DFNet only models the short-range dependencies between each side-view SAI and the center-view SAI, while our DPT explores long-range relations among all SAIs.

Qualitative Results. Fig. 3 depicts some representative visual results of different approaches for  $\times 4$  SR. As seen, LF-InterNet (Wang et al. 2020) and LF-DFNet (Wang et al. 2021d) produce distorted results for the structures of the telescope stand in the Bicycle scene and the flowerbed edge in the Sculpture scene. In contrast, our DPT yields much better results, with all above regions being well preserved.

Computational Analysis. Table 2 provides a detailed analysis of the LFSR models in terms of parameters, FLOPs, and reconstruct accuracy on EPFL. Following (Wang et al. 2021d), we set the size of the input LF image as  $5 \times 5 \times 32 \times 32$  for the computation of the FLOPs. As can be seen, DPT

shows the best trade-off between reconstruction accuracy and model efficiency. For  $\times 4$  SR, DPT has fewer parameters while better accuracy, in comparison with LF-DFNet (Wang et al. 2021d). This result further confirms the effectiveness of DPT, not only in better performance but also its efficiency.

# ABLATION STUDY

To gain more insights into our model, we conduct a set of ablative experiments on the EPFL dataset for  $\times 4$  SR. The results are reported in Table 3. We show in the 1st row the performance of baseline model (i.e., LF-DFNet (Wang et al. 2021d)), and the 7th row the results of our full model.

Content Transformer. We first investigate the effect of the content Transformer by constructing a network which is implemented with the convolution feature extraction module  $\mathcal{E}^{\mathrm{Cont}}$ , content Transformer  $\mathcal{T}^{\mathrm{Cont}}$  and the image reconstruction module  $\mathcal{D}^{\mathrm{Reco}}$ . Its results are given in the 2nd row. We can see that the content Transformer itself can lead to a similar performance as the baseline model, however, our content Transformer has fewer parameters. This confirms that

global relations among different views brought by the content Transformer are beneficial to LFSR.

Gradient Transformer. Furthermore, we combine the gradient Transformer into the content Transformer, with the results being shown in the 3rd row. As we can see, by introducing the gradient Transformer, the performance of the content Transformer (the 2nd row) improves by 0.09dB in terms of PSNR and 0.0011 improvement in terms of SSIM, respectively. Moreover, the models with dual transformers (the 4th, the 6th and the 7th rows) outperform the model with only a content Transformer (the 2nd row), which further confirms the effectiveness of gradient Transformer. Finally, we replace the content Transformer and gradient Transformer of DPT with residual blocks while maintaining the network parameters almost unchanged. Its results are given in the 5th row. The dual branches in DPT are equivalent to two convolutional neural networks for feature extraction. As reported in Table 3, DPT has a 0.21 dB PSNR drop, demonstrating the effectiveness of the proposed Transformers.

Impact of Fusion Mechanism. To explore the effect of fusion mechanism for content and gradient feature aggregation, we construct two model variants, which generates the detailed-preserved representation  $Z_{i}$  in Eq. 7 by the element-wise summation (the 3rd row) and the Transformer for a single view non-local dependencies exploration (the 4th row), respectively. As can be observed, our crossattention fusion Transformer (the 7th row) brings a promising performance improvement over the results of 3rd row and the 4th row in PSNR and SSIM, which is attributed to its effectiveness for the complementary non-local information exploration of the content and gradient features.

Efficacy of SA-LSA. We study the effect of the SA-LSA layers by replacing them with the vanilla fully-connected self-attention layers in DPT (the 6th row). As seen, the model with SA-LSA layers (the 7th row) obtains a better

![](images/e4a95d16bce92c666242138a1f4088fe69d0f1fd9d96ee278a3349e1282f61a0.jpg)

performance with fewer parameters, proving the effectiveness of local spatial context for SAI reconstruction.

Number of Attention Blocks  $K$ . At last, we study the performance of DPT with respect to the number of spatial-angular attention blocks in the content Transformer (or the graidient Transformer). Table 4 reports the comparison results. As seen, DPT achieves the best results at  $K = 2$ . Thus, we choose  $K = 2$  as the default number of spatial-angular attention blocks in DPT.

# CONCLUSION

This work proposes a Detail Preserving Transformer (DPT), as the first application of Transformer for LFSR. Instead of leveraging a vanilla fully-connected self-attention layer, we develop a spatial-angular locally-enhanced self-attention layer (SA-LSA) to promote non-local spatial-angular dependencies of each sub-aperture image sequence. Based on SA-LSA, we leverage a content Transformer and a gradient Transformer to learn spatial-angular content and gradient representations, respectively. The comprehensive spatial-angular representations are further processed by a cross-attention fusion Transformer to aggregate the output of the two Transformers, from which a high-resolution light field is reconstructed. We compare the proposed network with other state-of-the-art methods over five commonly-used benchmarks, and the experimental results demonstrate that it achieves favorable performance against other competitors.