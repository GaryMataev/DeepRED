# DeepRED: Deep Image Prior Powered by RED
Link: https://arxiv.org/abs/1903.10176

You can reproduce the results in the article using this code

## Abstract:
Inverse problems in imaging are extensively studied, with
a variety of strategies, tools, and theory that have been accumulated
over the years. Recently, this field has been immensely
influenced by the emergence of deep-learning techniques.
One such contribution, which is the focus of this
paper, is the Deep Image Prior (DIP) work by Ulyanov,
Vedaldi, and Lempitsky (2018). DIP offers a new approach
towards the regularization of inverse problems, obtained by
forcing the recovered image to be synthesized from a given
deep architecture. While DIP has been shown to be quite
an effective unsupervised approach, its results still fall short
when compared to state-of-the-art alternatives.
In this work, we aim to boost DIP by adding an explicit
prior, which enriches the overall regularization effect in order
to lead to better-recovered images. More specifically,
we propose to bring-in the concept of Regularization by Denoising
(RED), which leverages existing denoisers for regularizing
inverse problems. Our work shows how the two
(DIP and RED) can be merged into a highly effective unsupervised
recovery process while avoiding the need to differentiate
the chosen denoiser, and leading to very effective
results, demonstrated for several tested problems.

#### Python 3
#### Pytorch >= 0.4

Follow the comments and instructions in the jupyter notebook 
