# Image-Similarity-using-Deep-Ranking


Overview

The goal of this project is to get introduced to the computer vision task of image similarity. Like
most tasks in this field, it’s been aided by the ability of deep networks to extract image features.
The task of image similarity is retrieve a set of n images closest to the query image. One
application of this task could involve visual search engine where we provide a query image and
want to find an image closest that image in the database.

My task, for this project, was to implement a slightly different version of the pipeline introduced in
“Learning Fine-grained Image Similarity with Deep Ranking”. I strongly encourage you to
read this pager - https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf.

Here’s another good resource - https://medium.com/@akarshzingade/image-similarity-usingdeep-ranking-c1bd83855978.

Project Description

Designed a slightly different version of the deep ranking
model as discussed in the paper. My network
architecture looks exactly the same, but the details of
the triplet sampling layer are a bit simpler. The
architecture consists of 3 identical networks (Q,P,N).
Each of these networks take a single image denoted by
pi, pi+, pi- respectively.

pi: Input to the Q (Query) network. This image is
randomly sampled across any class.

pi+: Input to the P (Positive) network. This image is
randomly sampled from the SAME class as the query
image.

pi-: Input to the N (Negative) network. This image is
randomly sample from any class EXCEPT the class of
pi.

The output of each network, denoted by f(pi), f(pi+), f(pi-) is the feature embedding of an image.
This gets fed to the ranking layer.

Ranking Layer
The ranking layer just computes the triplet loss. It teaches the network to produce similar feature
embeddings for images from the same class (and different embeddings for images from
different classes). g is a gap parameter used for regularization purposes.
