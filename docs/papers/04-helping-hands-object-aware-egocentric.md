---
title: "Helping Hands: An Object-Aware Ego-Centric Video Recognition Model"
authors: "Chuhan Zhang, Ankush Gupta, Andrew Zisserman"
venue: "ICCV 2023"
year: 2023
shelf: "2 — Action Detection / Segmentation"
arxiv: "https://arxiv.org/abs/2308.07918"
relevance: "Object awareness in egocentric recognition — actions inseparable from tools/parts being manipulated"
chosen: false
reevaluation: "Good insight (hands+objects are the primary carriers of action semantics in ego video) but it's a training-time auxiliary objective technique. We can't add object-aware decoders to a prompt-based VLM pipeline. The takeaway is conceptual: our VLM prompt should explicitly ask about hand positions and object states, not just 'what action is happening'. Low direct actionability, but the insight informs prompt design."
---

# Helping Hands: An Object-Aware Ego-Centric Video Recognition Model

## Abstract
We introduce an object-aware decoder for improving the performance of spatio-temporal representations on ego-centric videos. The key idea is to enhance object-awareness during training by tasking the model to predict hand positions, object positions, and the semantic label of the objects using paired captions when available. At inference time the model only requires RGB frames as inputs, and is able to track and ground objects (although it has not been trained explicitly for this). We demonstrate the performance of the object-aware representations learnt by our model by evaluating it for strong transfer through zero-shot testing on a number of downstream video-text retrieval and classification benchmarks and by using the representations learned as input for long-term video understanding tasks (such as Episodic Memory in Ego4D), with performance improving over the state of the art in all cases, even compared to networks trained with far larger batch sizes.

## Introduction
Ego-centric (first-person) video understanding has become increasingly important with the rise of wearable cameras and AR/VR applications. A key observation is that in ego-centric videos, hands and the objects they interact with are central to understanding the ongoing activity. However, most existing spatio-temporal video models process raw RGB frames without explicit awareness of these critical elements.

This paper introduces an object-aware decoder that enhances spatio-temporal representations by learning to attend to hands and objects during training. The architecture is composed of three parts: a video backbone, a text backbone, and an object-aware decoder. The decoder is a cross-attention transformer that takes the visual feature map as keys and values, which are attended by a set of learnable queries. Among these permutation-invariant query vectors, the hand and object queries are trained to be object-aware and predict the localization and class of hands and objects.

The key contributions are: (1) an object-aware decoder that improves spatio-temporal representations for ego-centric videos by predicting hand positions, object positions, and semantic labels during training; (2) at inference time, the model only requires RGB frames — no hand/object annotations are needed — yet it retains the ability to track and ground objects even without explicit training for this; (3) strong zero-shot transfer results on multiple downstream benchmarks including video-text retrieval on EgoMCQ, action classification on Epic-Kitchens and EGTEA, and long-term video understanding tasks on Ego4D Episodic Memory.

## Conclusion
The paper demonstrates that injecting object-awareness into spatio-temporal representations through auxiliary training objectives (hand/object localization and semantic labeling) leads to consistent improvements across a wide range of ego-centric video understanding tasks. The inclusion of hand bounding box predictions during training consistently improves zero-shot transfer performance across all evaluated benchmarks, yielding approximately 1% higher accuracy on Epic-Kitchens and EGTEA. The model achieves comparable results on multiple-choice questions on EgoMCQ, and state-of-the-art results on the multi-instance retrieval task on EpicKitchens and action classification on EGTEA. Performance also improves over the state of the art on long-term video understanding tasks such as Ego4D Episodic Memory, even compared to networks trained with far larger batch sizes. This suggests that object-awareness is a powerful inductive bias for ego-centric video models — the hands and objects they interact with are the primary carriers of action semantics in this viewpoint.
