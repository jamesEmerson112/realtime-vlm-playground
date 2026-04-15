---
title: "SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation"
authors: "Arpit Bahety, Arnav Balaji, Ben Abbatematteo, Roberto Martin-Martin"
venue: "RSS 2025"
year: 2025
shelf: "4 — Robot Learning Bridge"
arxiv: "https://arxiv.org/abs/2506.15847"
relevance: "Dream version — robot learns from single human video, segments demo, translates to robot morphology safely"
chosen: false
reevaluation: "The 'dream version' of the full pipeline. Interesting that SafeMimic also uses VLMs to parse video into semantic segments — conceptually similar to what we're doing. The backtracking mechanism (when forward progress fails, try alternatives) is a clever robustness pattern. The safety Q-function ensemble concept could inspire our confidence thresholding. But overall too far from our current work to be directly actionable."
---

# SafeMimic: Towards Safe and Autonomous Human-to-Robot Imitation for Mobile Manipulation

## Abstract

For robots to become efficient helpers in the home, they must learn to perform new mobile manipulation tasks simply by watching humans perform them. Learning from a single video demonstration from a human is challenging as the robot needs to first extract from the demo what needs to be done and how, translate the strategy from a third to a first-person perspective, and then adapt it to be successful with its own morphology. Furthermore, to mitigate the dependency on costly human monitoring, this learning process should be performed in a safe and autonomous manner. We present SafeMimic, a framework to learn new mobile manipulation skills safely and autonomously from a single third-person human video. Given an initial human video demonstration of a multi-step mobile manipulation task, SafeMimic first parses the video into segments, inferring both the semantic changes caused and the motions the human executed to achieve them and translating them to an egocentric reference. It then adapts the behavior to the robot's own morphology by sampling candidate actions around the human ones, and verifying them for safety before execution in a receding horizon fashion using an ensemble of safety Q-functions trained in simulation. When safe forward progression is not possible, SafeMimic backtracks to previous states and attempts a different sequence of actions, adapting both the trajectory and the grasping modes when required for its morphology. Our experiments show that our method allows robots to safely and efficiently learn multi-step mobile manipulation behaviors from a single human demonstration, from different users, and in different environments, with improvements over state-of-the-art baselines across seven tasks.

## Introduction

For robots to become truly useful assistants in domestic environments, they need to acquire new manipulation skills rapidly and safely — ideally by simply watching a human perform the task. This vision of "learning by watching" is compelling because it eliminates the need for specialized teleoperation interfaces, kinesthetic teaching, or extensive reward engineering. However, translating a single human video demonstration into safe robot execution involves several deeply challenging steps.

First, the robot must parse the demonstration video into meaningful segments, understanding both what semantic changes occurred (e.g., an object was moved from one location to another) and how the human achieved them (the trajectory and manipulation strategy used). Second, it must translate these observations from the third-person camera perspective of the demonstration to its own first-person, egocentric reference frame. Third, and critically, it must adapt the demonstrated behavior to its own morphology — a mobile manipulator with different kinematics, reach, and grasping capabilities than a human hand.

Beyond these perception and planning challenges, safety is a paramount concern. When a robot attempts to reproduce a demonstrated behavior autonomously, it may encounter states where the human's strategy is infeasible or unsafe given the robot's morphology and the physical constraints of the environment. Existing human-to-robot imitation methods typically require human supervision during execution to ensure safety, creating a bottleneck that limits scalability.

SafeMimic addresses all of these challenges in a unified framework. The system uses vision-language models to parse the demonstration video into semantic segments, extracting both the goals (what changed) and the motions (how it changed) for each step. These are translated into the robot's egocentric frame. For execution, SafeMimic employs a receding-horizon safety verification mechanism: candidate actions are sampled around the human-demonstrated trajectory and evaluated by an ensemble of safety Q-functions trained in simulation before execution on the real robot. This ensures that the robot only executes actions that are predicted to be safe.

A key innovation is the backtracking mechanism: when no safe forward progression is possible from the current state, SafeMimic autonomously backtracks to a previous safe state and explores alternative action sequences, potentially adapting the grasping mode or trajectory to find a safe path forward. This enables fully autonomous execution without human monitoring.

## Conclusion

SafeMimic demonstrates that robots can safely and autonomously learn multi-step mobile manipulation behaviors from a single third-person human video demonstration. The framework successfully addresses the three core challenges of human-to-robot imitation: video parsing and semantic understanding, perspective and morphology translation, and safe autonomous execution.

The experiments across seven diverse mobile manipulation tasks show that SafeMimic achieves consistent improvements over state-of-the-art baselines, with the key advantages being: (1) safety — the ensemble of safety Q-functions prevents unsafe executions while still allowing task completion; (2) autonomy — the backtracking mechanism eliminates the need for human monitoring during execution; and (3) generalization — the system works across different human demonstrators and environments.

The combination of vision-language model-based video parsing, receding-horizon safety verification, and autonomous backtracking represents a significant step toward the vision of robots that can learn new skills simply by watching humans — safely and without human intervention. The project page with additional details is available at https://robin-lab.cs.utexas.edu/SafeMimic/.
