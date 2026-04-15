---
title: "RoboTube: Learning Household Manipulation from Human Videos with Simulated Twin Environments"
authors: "Haoyu Xiong, Haoyuan Fu, Jieyi Zhang, Chen Bao, Qiang Zhang, Yongxi Huang, Wenqiang Xu, Animesh Garg, Cewu Lu"
venue: "CoRL 2022"
year: 2022
shelf: "4 — Robot Learning Bridge"
arxiv: "https://proceedings.mlr.press/v205/xiong23a.html"
relevance: "Benchmark + simulation bridge — 5,000 human video demonstrations with digital twin environments for reproducible robot learning"
chosen: false
reevaluation: "Furthest from our immediate work. A dataset+simulation paper for robot learning. The digital twin concept is interesting long-term (imagine a digital twin of the circuit breaker station) but not actionable now. Lowest priority of the 12."
---

# RoboTube: Learning Household Manipulation from Human Videos with Simulated Twin Environments

## Abstract

We aim to build a useful, reproducible, democratized benchmark for learning household robotic manipulation from human videos. To realize this goal, a diverse, high-quality human video dataset curated specifically for robots is desired. A simulated twin environment that resembles the appearance and the dynamics of the physical world would help roboticists and AI researchers validate their algorithms convincingly and efficiently before testing on a real robot. Hence, we present RoboTube, a human video dataset, and its digital twins for learning various robotic manipulation tasks. The RoboTube video dataset contains 5,000 video demonstrations recorded with multi-view RGB-D cameras of human-performing everyday household tasks including manipulation of rigid objects, articulated objects, granular objects, deformable objects, and bimanual manipulation. RT-sim, as the simulated twin environments, consists of 3D scanned, photo-realistic objects, minimizing the visual domain gap between the physical world and the simulated environment. After extensively benchmarking existing methods in the field of robot learning from videos, our empirical results suggest that knowledge and models learned from the RoboTube video dataset can be deployed, benchmarked, and reproduced in RT-sim and be transferred to a real robot. We hope RoboTube can lower the barrier to robotics research for beginners while facilitating reproducible research in the community.

## Introduction

Learning robotic manipulation from human video demonstrations is a promising direction for enabling robots to acquire diverse household skills, but progress in this area has been hampered by two key limitations: the lack of large-scale, high-quality human video datasets designed specifically for robotic learning, and the absence of faithful simulation environments where algorithms can be validated before real-world deployment.

Existing robotic manipulation datasets tend to be either collected via teleoperation (limiting scale and naturalness) or sourced from internet videos (lacking the multi-view, depth, and calibration information needed for precise manipulation learning). Meanwhile, simulation environments often suffer from a significant visual and physical domain gap with the real world, making sim-to-real transfer unreliable.

RoboTube addresses both challenges simultaneously. The dataset provides 5,000 video demonstrations across diverse manipulation categories — rigid objects, articulated objects, granular objects, deformable objects, and bimanual tasks — all recorded with calibrated multi-view RGB-D cameras capturing human hands performing everyday household tasks. This provides the diversity, scale, and sensing modalities needed for robust robot learning from human video.

The companion RT-sim simulation environment bridges the gap between human demonstrations and robot execution. By using 3D-scanned, photo-realistic digital twins of the physical objects and environments, RT-sim minimizes the visual domain gap that typically plagues sim-to-real transfer. This enables researchers to develop, benchmark, and iterate on algorithms in simulation with high confidence that results will transfer to real robots.

## Conclusion

RoboTube presents a comprehensive benchmark ecosystem for learning household robotic manipulation from human videos, combining a large-scale human video dataset with high-fidelity simulated twin environments. The extensive benchmarking of existing methods demonstrates that knowledge learned from the RoboTube video dataset can be successfully deployed and reproduced in RT-sim, and ultimately transferred to real robot execution.

The key contributions are threefold: (1) a 5,000-demonstration human video dataset spanning five manipulation categories with multi-view RGB-D recording; (2) RT-sim, a photo-realistic simulated twin environment built from 3D-scanned objects that minimizes the visual domain gap; and (3) comprehensive benchmarks of state-of-the-art methods for robot learning from videos, establishing reproducible baselines for the community.

The authors hope that RoboTube will lower the barrier to entry for robotics research — providing beginners with accessible data and simulation tools — while enabling experienced researchers to conduct reproducible experiments. The project website with dataset access is available at https://www.robotube.org/.
