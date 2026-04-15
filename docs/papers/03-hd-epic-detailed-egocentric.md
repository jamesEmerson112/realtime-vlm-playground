---
title: "HD-EPIC: A Highly-Detailed Egocentric Video Dataset"
authors: "Toby Perrett, Ahmad Darkhalil, Saptarshi Sinha, Omar Emara, Sam Pollard, Kranti Parida, Kaiting Liu, Prajwal Gatti, Siddhant Bansal, Kevin Flanagan, Jacob Chalk, Zhifan Zhu, Rhodri Guerrier, Fahd Abdelazim, Bin Zhu, Davide Moltisanti, Michael Wray, Hazel Doughty, Dima Damen"
venue: "CVPR 2025"
year: 2025
shelf: "1 — Benchmark Papers"
arxiv: "https://arxiv.org/abs/2502.04144"
relevance: "Fine-grained steps, audio annotations, 3D grounding — relevant for better temporal understanding"
chosen: false
reevaluation: "The Gemini Pro VQA result (38.5% on fine-grained egocentric questions) directly calibrates our expectations — current VLMs struggle with this exact type of understanding. The 51K audio event annotations and 263 annotations/minute density show what 'detailed' actually means. Kitchen-specific (not mechanics), but the annotation methodology and VLM failure modes are informative. Moderate relevance."
---

# HD-EPIC: A Highly-Detailed Egocentric Video Dataset

## Abstract

We present a validation dataset of newly-collected kitchen-based egocentric videos, manually annotated with highly detailed and interconnected ground-truth labels covering: recipe steps, fine-grained actions, ingredients with nutritional values, moving objects, and audio annotations. Importantly, all annotations are grounded in 3D through digital twinning of the scene, fixtures, object locations, and primed with gaze. Footage is collected from unscripted recordings in diverse home environments, making HD-EPIC the first dataset collected in-the-wild but with detailed annotations matching those in controlled lab environments. We show the potential of our highly-detailed annotations through a challenging VQA benchmark of 26K questions assessing the capability to recognise recipes, ingredients, nutrition, fine-grained actions, 3D perception, object motion, and gaze direction. The powerful long-context Gemini Pro only achieves 38.5% on this benchmark, showcasing its difficulty and highlighting shortcomings in current VLMs. We additionally assess action recognition, sound recognition, and long-term video-object segmentation on HD-EPIC. HD-EPIC is 41 hours of video in 9 kitchens with digital twins of 413 kitchen fixtures, capturing 69 recipes, 59K fine-grained actions, 51K audio events, 20K object movements and 37K object masks lifted to 3D.

## Introduction

Egocentric video datasets have grown substantially in scale over recent years, from EPIC-KITCHENS (55 hours in kitchens) to Ego4D (3,670 hours across diverse scenarios). However, this scaling has typically come at the cost of annotation granularity: larger datasets tend to rely on coarser labels or automated annotations. There remains a fundamental tension between dataset scale and annotation detail. Controlled lab environments can produce highly detailed annotations but lack ecological validity, while in-the-wild recordings capture naturalistic behavior but have been limited to coarse or single-modality labels.

HD-EPIC resolves this tension by collecting unscripted kitchen recordings in diverse home environments -- capturing real cooking behavior as it naturally occurs -- while providing annotation detail that matches or exceeds controlled lab datasets. The key enabler is the use of digital twinning: each kitchen is scanned to produce a 3D reconstruction, and all 413 kitchen fixtures (cabinets, appliances, surfaces) are identified within the digital twin. This grounds every annotation in 3D space, connecting actions to the physical environment in which they occur.

The dataset provides an unprecedented density of annotations -- an average of 263 annotations per minute of video footage -- spanning multiple interconnected modalities: recipe-level structure (69 recipes with hierarchical step decompositions), fine-grained temporal actions (59K action segments with verb-noun labels), ingredient tracking with nutritional values, audio event annotations (51K events covering speech, object sounds, and ambient noise), object motion trajectories (20K movements), and dense video-object segmentation masks (37K masks) lifted to 3D coordinates. Crucially, all annotations share a common spatial reference frame through the digital twin and are temporally aligned, enabling research on the interplay between different modalities and levels of granularity.

## Conclusion

We presented HD-EPIC, a highly-detailed egocentric video dataset of 41 hours of unscripted kitchen recordings across 9 diverse home kitchens. By combining in-the-wild collection with digital twinning, HD-EPIC achieves a level of annotation detail previously possible only in controlled lab settings -- with an average of 263 annotations per minute spanning recipe steps, fine-grained actions, ingredients with nutritional values, audio events, object movements, and segmentation masks, all grounded in 3D.

Our VQA benchmark of 26K questions reveals significant shortcomings in current vision-language models, with the powerful long-context Gemini Pro achieving only 38.5%. This underscores the gap between current VLM capabilities and the level of understanding required for fine-grained egocentric video comprehension. The benchmarks on action recognition, sound recognition, and video-object segmentation further demonstrate that HD-EPIC provides challenging and complementary evaluation settings.

We believe HD-EPIC will serve as a valuable resource for research on fine-grained activity understanding, multimodal learning, and 3D-grounded video reasoning, and we will continue to expand the dataset with additional kitchens and annotation types. The interconnected nature of HD-EPIC's annotations -- linking recipes to actions to objects to audio to 3D space -- creates opportunities for holistic understanding that go beyond what any single annotation modality can provide.
