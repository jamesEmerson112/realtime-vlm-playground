---
title: "Action Scene Graphs for Long-Form Understanding of Egocentric Videos"
authors: "Ivan Rodin, Antonino Furnari, Kyle Min, Subarna Tripathi, Giovanni Maria Farinella"
venue: "CVPR 2024"
year: 2024
shelf: "2 — Action Detection / Segmentation"
arxiv: "https://arxiv.org/abs/2312.03391"
relevance: "Graph representation of actions/objects/relationships over time — captures sequence structure for long procedures"
chosen: true
actionable_insights: "PRE/PNR/POST framing — detect point-of-no-return to predict step completion earlier, reduce latency"
---

# Action Scene Graphs for Long-Form Understanding of Egocentric Videos

## Abstract
We present Egocentric Action Scene Graphs (EASGs), a new representation for long-form understanding of egocentric videos. EASGs extend standard manually-annotated representations of egocentric videos, such as verb-noun action labels, by providing a temporally evolving graph-based description of the actions performed by the camera wearer, including interacted objects, their relationships, and how actions unfold in time. Through a novel annotation procedure, we extend the Ego4D dataset by adding manually labeled Egocentric Action Scene Graphs offering a rich set of annotations designed for long-form egocentric video understanding. We hence define the EASG generation task and provide a baseline approach, establishing preliminary benchmarks. Experiments on two downstream tasks, egocentric action anticipation and egocentric activity summarization, highlight the effectiveness of EASGs for long-form egocentric video understanding.

## Introduction
Understanding long-form egocentric videos is a key challenge in computer vision, with applications ranging from activity recognition and anticipation to assistive technologies and robotics. Current approaches to egocentric video understanding typically rely on simple verb-noun action labels (e.g., "cut tomato", "open drawer") to represent activities. While these labels capture the core action-object pair, they lose critical information about the broader context: which other objects are present, how objects relate to each other spatially, and how the scene evolves over the course of a procedure.

This paper proposes Egocentric Action Scene Graphs (EASGs) as a richer representation that addresses these limitations. An EASG is formalized as a time-varying directed graph G(t) = (V(t), E(t)), where V(t) is the set of nodes at time t and E(t) is the set of edges between such nodes. Each temporal realization of the graph corresponds to an egocentric action spanning over a set of three frames defined as the precondition (PRE), the point of no return (PNR), and the postcondition (POST) frames. This three-frame structure captures the state before an action, the moment of commitment, and the resulting state — providing a much richer description than a single verb-noun label.

The annotation procedure involves several stages: obtaining an initial EASG by leveraging existing annotations from Ego4D through an initialization and refinement procedure, followed by graph refinement via inputs from 3 annotators, with a validation stage that aggregates data from the three annotators to ensure the quality of final annotations. Through this process, the authors extend the Ego4D dataset with manually labeled EASGs.

The key contributions are: (1) the EASG representation itself — a temporally evolving graph that captures objects, their relationships, and action dynamics over time; (2) a novel annotation procedure and the resulting dataset extending Ego4D; (3) the EASG generation task with a baseline approach and preliminary benchmarks; (4) experiments on egocentric action anticipation and activity summarization demonstrating that EASGs are more effective than standard verb-noun representations for long-form understanding.

## Conclusion
Egocentric Action Scene Graphs provide a structured, temporally evolving representation that goes beyond simple verb-noun labels to capture the full richness of egocentric activities — including objects, spatial relationships, and how actions transform the scene over time. The experiments demonstrate that short EASG sequences tend to outperform long verb-noun sequences, highlighting the higher representation power of EASGs compared to standard verb-noun representations. EASG representations achieve the best results for long sequences on both the action anticipation and activity summarization downstream tasks. The work establishes the EASG generation task, provides baseline approaches and benchmarks, and releases annotations extending the Ego4D dataset. Future directions include extending the dataset to videos from other institutions and developing more powerful EASG generation models that can automatically produce these rich graph-based descriptions from raw video input.
