---
title: "Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos"
authors: "Yuhan Shen, Ehsan Elhamifar"
venue: "CVPR 2024"
year: 2024
shelf: "2 — Action Detection / Segmentation"
arxiv: "https://openaccess.thecvf.com/content/CVPR2024/html/Shen_Progress-Aware_Online_Action_Segmentation_for_Egocentric_Procedural_Task_Videos_CVPR_2024_paper.html"
relevance: "Closest to our streaming case — online/streaming action segmentation for procedural tasks, discusses offline vs online mismatch"
chosen: true
actionable_insights: "Action progress prediction (% complete instead of binary), task graph constraints from procedure JSON, over-segmentation fixes"
---

# Progress-Aware Online Action Segmentation for Egocentric Procedural Task Videos

## Abstract
While previous studies have mostly focused on offline action segmentation where entire videos are available for both training and inference, the transition to online action segmentation is crucial for practical applications like AR/VR task assistants. Notably, applying an offline-trained model directly to online inference results in a significant performance drop due to the inconsistency between training and inference. We propose an online action segmentation framework by first modifying existing architectures to make them causal. Second, we develop a novel action progress prediction module to dynamically estimate the progress of ongoing actions and use them to refine the predictions of causal action segmentation. Third, we propose to learn task graphs from training videos and leverage them to obtain smooth and procedure-consistent segmentations. The combination of progress and task graph with causal action segmentation effectively addresses prediction uncertainty and over-segmentation in online action segmentation and achieves significant improvement on three egocentric datasets.

## Introduction
Action segmentation aims to assign an action label to each frame of a long, untrimmed video. It is a fundamental task for understanding procedural activities, with applications in AR/VR assistants, instructional video analysis, and robotic task planning. Most existing work addresses offline action segmentation, where the entire video is available at both training and inference time. However, many real-world applications demand online (streaming) segmentation, where predictions must be made frame-by-frame as the video arrives, without access to future frames.

A naive approach to online action segmentation is to train a model offline and apply it in a causal (online) manner at inference time. However, this leads to a significant performance drop due to the inconsistency between training and inference: the model was trained with access to future context but must now predict without it. This offline-to-online gap is a central challenge that this work addresses.

The paper proposes ProTAS (Progress-aware Online Temporal Action Segmentation), a framework with three key components: (1) Causal architectures — existing temporal action segmentation models are modified to use only past and current frames, making them suitable for online inference while being trained in a causal manner to eliminate the train-test mismatch; (2) Action Progress Prediction (APP) — a novel module that dynamically estimates how far along each action has progressed (e.g., 30% complete, 80% complete), providing a richer signal than binary action labels alone. By knowing an action's progress, the model can better predict when transitions will occur, reducing over-segmentation; (3) Task graphs — learned from training videos, these graphs encode the typical ordering and dependencies between actions in a procedure. At inference time, the task graph constrains predictions to follow plausible action sequences, yielding smoother and more procedure-consistent segmentations.

The framework is evaluated on three egocentric procedural video datasets and achieves significant improvements over prior online methods, substantially closing the gap with offline approaches.

## Conclusion
ProTAS demonstrates that progress-aware reasoning and procedural knowledge (via task graphs) are effective tools for online action segmentation in egocentric procedural videos. The action progress prediction module provides a continuous, fine-grained signal that reduces the over-segmentation problem common in causal models, while the task graph enforces procedure-level consistency. Together, these components achieve significant improvements on three egocentric datasets, substantially narrowing the gap between online and offline action segmentation performance. The work highlights the importance of designing models specifically for the online setting rather than naively adapting offline models, and the value of incorporating procedural structure as an inductive bias for streaming video understanding.
