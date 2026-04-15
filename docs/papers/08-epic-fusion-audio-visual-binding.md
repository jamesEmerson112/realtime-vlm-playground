---
title: "EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition"
authors: "Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen"
venue: "ICCV 2019"
year: 2019
shelf: "3 — Audio-Visual Fusion"
arxiv: "https://arxiv.org/abs/1908.08498"
relevance: "Directly addresses audio-video temporal misalignment in egocentric video — temporal binding window for asynchronous fusion"
chosen: true
actionable_insights: "Temporal binding window — keep sliding buffer of last N audio transcripts, not just most recent. Audio offset from visual event by 2-5s"
---

# EPIC-Fusion: Audio-Visual Temporal Binding for Egocentric Action Recognition

## Abstract

We focus on multi-modal fusion for egocentric action recognition, and propose a novel architecture for multi-modal temporal-binding, i.e. the combination of modalities within a range of temporal offsets, trained end-to-end, with three modalities: RGB, Flow and Audio. We introduce the Temporal Binding Network (TBN) that fuses modalities with mid-level fusion alongside sparse temporal sampling of fused representations. In contrast with previous works, modalities are fused before temporal aggregation, with shared modality and fusion weights over time. The proposed architecture is trained end-to-end, outperforming individual modalities as well as late-fusion of modalities. We demonstrate the importance of audio in egocentric vision, on per-class basis, for identifying actions as well as interacting objects. Our method achieves state of the art results on both the seen and unseen test sets of the largest egocentric dataset: EPIC-Kitchens, on all metrics using the public leaderboard.

## Introduction

The egocentric domain offers rich sounds resulting from the interactions between hands and objects, as well as the close proximity of the wearable microphone to the undergoing action. Audio can capture actions that are out of the wearable camera's field of view, but audible (e.g., "eat" can be heard but not seen). Despite the availability of audio in egocentric datasets, most prior work in egocentric action recognition has focused exclusively on visual modalities (RGB and optical flow), overlooking the audio signal.

In this paper, the authors propose the Temporal Binding Network (TBN), a novel architecture for multi-modal temporal-binding. The key insight is that different modalities need not be strictly temporally aligned -- an audio event might precede or follow the visual manifestation of the same action. The temporal binding mechanism allows the network to learn the optimal temporal offset between modalities, fusing them within a range of temporal offsets rather than requiring strict synchrony.

The paper makes three main contributions:

1. An end-to-end trainable mid-level fusion Temporal Binding Network (TBN) is proposed, which fuses modalities before temporal aggregation with shared modality and fusion weights over time.
2. The first audio-visual fusion attempt in egocentric action recognition, demonstrating that audio carries significant complementary information to visual modalities in the egocentric setting.
3. State-of-the-art results on both the seen and unseen test sets of the EPIC-Kitchens dataset, the largest egocentric dataset at the time, on all metrics using the public leaderboard.

The architecture uses sparse temporal sampling, where segments are sampled from the video and for each segment, snippets from the three modalities (RGB, optical flow, audio) are extracted. These snippets need not be temporally aligned -- they can be offset within a temporal binding window. Each modality is processed by its own backbone network (e.g., BN-Inception for RGB and flow, AudioVGG for audio), and the resulting features are fused at mid-level before temporal aggregation via consensus pooling.

## Conclusion

The paper presents EPIC-Fusion, the first audio-visual fusion approach for egocentric action recognition. The results demonstrate three key findings: (i) the efficacy of audio for egocentric action recognition -- audio provides complementary information to RGB and flow, particularly for actions where the sound is more discriminative than the visual appearance; (ii) the advantage of mid-level fusion within a Temporal Binding Window (TBW) over late fusion -- binding modalities before temporal aggregation with learned temporal offsets outperforms simply combining predictions from independently trained modality-specific networks; and (iii) the robustness of the model to background or irrelevant sounds.

The per-class analysis reveals that audio is particularly useful for recognizing certain verbs (e.g., "wash", "mix") and nouns (e.g., "tap", "water") where the sound signature is highly informative. The temporal binding mechanism allows the model to handle the natural asynchrony between visual and auditory signals in egocentric recordings, where the sound of an action may slightly precede or follow the visual observation.
