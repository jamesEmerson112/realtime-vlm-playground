---
title: "SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos"
authors: "Changan Chen, Kumar Ashutosh, Rohit Girdhar, David Harwath, Kristen Grauman"
venue: "CVPR 2024"
year: 2024
shelf: "3 — Audio-Visual Fusion"
arxiv: "https://arxiv.org/abs/2404.05206"
relevance: "Audio carries action semantics — learns from audio+language+vision together from narrated egocentric videos"
chosen: true
actionable_insights: "Consensus mechanism — when audio+video agree boost confidence, when they disagree downgrade. Procedure text as bridge/anchor modality"
---

# SoundingActions: Learning How Actions Sound from Narrated Egocentric Videos

## Abstract

We propose a novel self-supervised embedding to learn how actions sound from narrated in-the-wild egocentric videos. Whereas existing methods rely on curated data with known audio-visual correspondence, our multimodal contrastive-consensus coding (MC3) embedding reinforces the associations between audio, language, and vision when all modality pairs agree, while diminishing those associations when any one pair does not. We show our approach can successfully discover how the long tail of human actions sound from egocentric video, outperforming an array of recent multimodal embedding techniques on two datasets (Ego4D and EPIC-Sounds) and multiple cross-modal tasks.

## Introduction

Human activity often produces sounds. Closing a door, chopping vegetables, typing on a keyboard, talking with a friend -- our interactions with the objects and people around us generate audio that reveals our physical behaviors. Understanding the link between sounds and actions is valuable for a number of applications, such as multimodal activity recognition, cross-modal retrieval, content generation, or forecasting the physical effects of a person's actions.

The key challenge addressed in this paper is learning audio-visual-language correspondences from in-the-wild egocentric video, where the data is weakly supervised and noisy. Unlike curated datasets where audio-visual pairs are carefully annotated, narrated egocentric videos contain natural narrations by the camera wearer describing their actions. These narrations provide a language signal that can bridge the audio and visual modalities, but the correspondence between modalities is imprecise -- the narration may not exactly describe the concurrent sound, and background noise may obscure the action sound.

The paper introduces the Multimodal Contrastive-Consensus Coding (MC3) embedding, a self-supervised learning framework that operates on three modalities simultaneously: audio, video (visual), and language (narrations). The core idea is to seek video samples where there is semantic agreement between all three modalities -- the audio, visual, and language -- while distancing those that do not agree. Language serves as an anchor that assures correspondences in the audio and visual streams stem from alignment on the sounding action, rather than incidental co-occurrences.

The MC3 method consists of two key stages:

1. **Alignment Stage**: A contrastive learning approach is used, with contrastive loss computed for each possible pair of modalities (Audio-Video, Audio-Language, Video-Language). This learns initial correspondences between all modality pairs.

2. **Consensus Stage**: The method reinforces associations where all three modality pairs agree (consensus) and diminishes those where any one pair disagrees. This filtering mechanism handles the noise inherent in in-the-wild data, where not every audio clip corresponds to the visible action or the narration.

The approach is particularly well-suited to egocentric video because the camera wearer's close proximity to the action means the microphone captures action-relevant sounds, and the narrations provide weak supervision about which actions are occurring.

## Conclusion

The paper presents SoundingActions, a self-supervised approach to learning how the long tail of human actions sound from narrated egocentric videos. The key contribution lies in the MC3 embedding's ability to learn about sounding actions from weakly supervised data, enabling learning from readily available egocentric video data rather than requiring carefully curated datasets with precise audio-visual correspondences.

The approach outperforms an array of recent multimodal embedding techniques on two large-scale datasets -- Ego4D and EPIC-Sounds -- across multiple cross-modal tasks including audio-to-video retrieval, video-to-audio retrieval, and zero-shot action recognition. The consensus mechanism proves critical: by requiring agreement across all three modality pairs (audio-video, audio-language, video-language), the model effectively filters out noisy correspondences and focuses on samples where the sound genuinely reflects the visible action described in the narration.

The work demonstrates that language (narrations) serves as a powerful bridge modality that disambiguates audio-visual associations, particularly for the long tail of actions where individual modality pairs may be unreliable. This has implications for scaling audio-visual learning to large uncurated video collections, where human annotations are impractical but natural narrations are available.
