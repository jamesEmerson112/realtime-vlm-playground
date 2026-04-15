---
title: "Ego-Only: Egocentric Action Detection without Exocentric Transferring"
authors: "Huiyu Wang, Mitesh Kumar Singh, Lorenzo Torresani"
venue: "ICCV 2023"
year: 2023
shelf: "2 — Action Detection / Segmentation"
arxiv: "https://arxiv.org/abs/2301.01380"
relevance: "Strong egocentric action detection without third-person transfer — how to detect actions over time in ego video"
chosen: false
reevaluation: "Academically strong (overturns the 'you need exocentric pretraining' dogma) but the MAE + temporal segmentation + ActionFormer pipeline is a custom model stack. Not applicable to our prompt-based VLM approach. The conceptual takeaway is encouraging: egocentric-only data is sufficient, you don't need third-person video. But no extractable techniques for us."
---

# Ego-Only: Egocentric Action Detection without Exocentric Transferring

## Abstract
We present Ego-Only, the first approach that enables state-of-the-art action detection on egocentric (first-person) videos without any form of exocentric (third-person) transferring. Despite the content and appearance gap separating the two domains, large-scale exocentric transferring has been the default choice for egocentric action detection. This is because prior works found that egocentric models are difficult to train from scratch and that transferring from exocentric representations leads to improved accuracy. However, in this paper, we revisit this common belief. Motivated by the large gap separating the two domains, we propose a strategy that enables effective training of egocentric models without exocentric transferring. Our Ego-Only approach is simple. It trains the video representation with a masked autoencoder finetuned for temporal segmentation. The learned features are then fed to an off-the-shelf temporal action localization method to detect actions. We find that this renders exocentric transferring unnecessary by showing remarkably strong results achieved by this simple Ego-Only approach on three established egocentric video datasets: Ego4D, EPIC-Kitchens-100, and Charades-Ego. On both action detection and action recognition, Ego-Only outperforms previous best exocentric transferring methods that use orders of magnitude more labels. Ego-Only sets new state-of-the-art results on these datasets and benchmarks without exocentric data.

## Introduction
Egocentric (first-person) action detection is the task of temporally localizing and classifying actions in long, untrimmed egocentric videos. Unlike exocentric (third-person) videos where the full body of the actor is visible, egocentric videos are characterized by a unique visual perspective: only the hands and the objects they interact with are visible, backgrounds change frequently due to head motion, and there is significant camera motion and blur.

The prevailing paradigm for egocentric action detection relies on transferring representations pretrained on large-scale exocentric video datasets (e.g., Kinetics). This exocentric-to-egocentric transfer has been the default approach because prior works found that training egocentric models from scratch leads to suboptimal results. However, this paper challenges this assumption by asking: is exocentric transferring truly necessary?

The authors observe that the content and appearance gap between egocentric and exocentric videos is substantial — different viewpoints, different visual features, different motion patterns. This domain gap motivates the exploration of an ego-only strategy. The Ego-Only approach consists of three training stages: (1) a masked autoencoder (MAE) stage that bootstraps the backbone representation through self-supervised learning on egocentric data only, (2) a simple fine-tuning stage that performs temporal semantic segmentation of egocentric actions, and (3) a final detection stage using an off-the-shelf temporal action detector such as ActionFormer, without any modification.

The key insight is that self-supervised pretraining via masked autoencoders on egocentric video data alone provides a strong enough initialization to train effective action detectors, eliminating the need for any exocentric data.

## Conclusion
Ego-Only demonstrates that state-of-the-art egocentric action detection is achievable without any exocentric data or transferring. The approach outperforms all previous results based on exocentric transferring, setting new state-of-the-art results obtained for the first time without additional data. Specifically, Ego-Only advances the state-of-the-art on Ego4D Moments Queries detection (+6.5% average mAP), EPIC-Kitchens-100 Action Detection (+5.5% on verbs and +6.2% on nouns), Charades-Ego action recognition (+3.1% mAP), and EPIC-Kitchens-100 action recognition (+1.1% top-1 accuracy on verbs). These results overturn the conventional wisdom that exocentric pretraining is essential for egocentric understanding, showing that the domain gap between the two viewpoints may actually hurt rather than help. The simplicity of the approach — self-supervised MAE pretraining followed by standard fine-tuning and off-the-shelf detection — makes it broadly applicable and easy to adopt.
