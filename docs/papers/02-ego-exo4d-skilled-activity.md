---
title: "Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives"
authors: "Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, Eugene Byrne, Zach Chavis, Joya Chen, Feng Cheng, Fu-Jen Chu, Sean Crane, Avijit Dasgupta, Jing Dong, Maria Escobar, Cristhian Forigua, Abrham Gebreselasie, Sanjay Haresh, Jing Huang, Md Mohaiminul Islam, Suyog Jain, Rawal Khirodkar, Devansh Kukreja, Kevin J Liang, Jia-Wei Liu, Sagnik Majumder, Yongsen Mao, Miguel Martin, Effrosyni Mavroudi, Tushar Nagarajan, Francesco Ragusa, Santhosh Kumar Ramakrishnan, Luigi Seminara, Arjun Somayazulu, Yale Song, Shan Su, Zihui Xue, Edward Zhang, Jinxu Zhang, Angela Castillo, Changan Chen, Xinzhu Fu, Ryosuke Furuta, Cristina Gonzalez, Prince Gupta, Jiabo Hu, Yifei Huang, Yiming Huang, Weslie Khoo, Anush Kumar, Robert Kuo, Sach Lakhavani, Miao Liu, Mi Luo, Zhengyi Luo, Brighid Meredith, Austin Miller, Oluwatumininu Oguntola, Xiaqing Pan, Penny Peng, Shraman Pramanick, Merey Ramazanova, et al."
venue: "CVPR 2024"
year: 2024
shelf: "1 — Benchmark Papers"
arxiv: "https://arxiv.org/abs/2311.18259"
relevance: "Closest to our 'student performing skilled task' setup — has keysteps, procedural dependencies, proficiency ratings"
chosen: false
reevaluation: "UNDERRATED — on re-read, more relevant than initially scored. The 'expert commentary' annotation (52 coaches critiquing technique) is DIRECTLY analogous to our instructor audio. Proficiency ratings = our error detection. Keystep temporal segmentation = our step tracking. Procedural dependency graphs = our procedure JSON. Consider promoting to read-closely tier. The expert commentary concept could inform how we interpret instructor corrections."
---

# Ego-Exo4D: Understanding Skilled Human Activity from First- and Third-Person Perspectives

## Abstract

We present Ego-Exo4D, a diverse, large-scale multimodal multiview video dataset and benchmark challenge. Ego-Exo4D centers around simultaneously-captured egocentric and exocentric video of skilled human activities (e.g., sports, music, dance, bike repair). 740 participants from 13 cities worldwide performed these activities in 123 different natural scene contexts, yielding long-form captures from 1 to 42 minutes each and 1,286 hours of video combined. The multimodal nature of the dataset is unprecedented: the video is accompanied by multichannel audio, eye gaze, 3D point clouds, camera poses, IMU, and multiple paired language descriptions -- including a novel "expert commentary" done by coaches and teachers and tailored to the skilled-activity domain. To push the frontier of first-person video understanding of skilled human activity, we also present a suite of benchmark tasks and their annotations, including fine-grained activity understanding, proficiency estimation, cross-view translation, and 3D hand/body pose.

## Introduction

People learn to perform physical activities -- cooking, sports, music, crafts, repair -- through a combination of practice and expert guidance. Understanding such skilled human activity from video is a fundamental challenge for computer vision, with implications for augmented reality, robotics, and education. We contend that both the egocentric and exocentric viewpoints are critical for capturing human skill. The two viewpoints are synergistic: the first-person (ego) perspective captures the details of close-by hand-object interactions and the camera wearer's attention, whereas the third-person (exo) perspective captures the full body pose and surrounding environment context.

Prior egocentric video datasets have focused on daily-life activities (e.g., Ego4D) or specific kitchen scenarios (e.g., EPIC-KITCHENS), but none have jointly captured ego and exo views of skilled activities at scale. Meanwhile, existing multiview datasets from controlled lab settings lack the diversity and naturalism needed to study real-world skill. Ego-Exo4D fills this gap by providing the largest public dataset of time-synchronized first- and third-person video, captured in naturalistic environments across a wide range of physical activities.

The dataset is the result of a two-year effort by a consortium of 15 research institutions. Each recording session captures one participant performing a skilled activity while wearing a head-mounted camera (ego view), simultaneously recorded by multiple stationary and handheld cameras (exo views). The activities span eight domains: cooking, bike/motorcycle repair, music, dance, basketball, soccer, bouldering/rock climbing, and health (physical therapy exercises). To annotate the skilled nature of these activities, we introduce several novel annotation types: (1) keystep temporal segments with natural language descriptions, (2) proficiency ratings by domain experts, (3) expert commentaries -- narrations by coaches and teachers who watch the recordings and critique the participant's technique, and (4) fine-grained procedural dependency graphs linking keysteps.

52 experts were recruited to critique the recorded videos, call out strengths and weaknesses, and explain how the specific behavior of the participant (e.g., hand/body pose, use of objects) affects the performance. These commentaries focus on how the activity is executed rather than what it entails, capturing subtle differences in skilled execution.

## Conclusion

We presented Ego-Exo4D, a diverse, large-scale multimodal multiview video dataset and benchmark challenge for understanding skilled human activity from both ego and exo perspectives. With 1,286 hours of time-synchronized video from 740 participants across 13 cities, 8 activity domains, and 123 scene contexts, the dataset is unprecedented in its scale and multimodal richness -- combining video with multichannel audio, eye gaze, 3D point clouds, camera poses, IMU, and multiple forms of language annotation including expert commentary.

Our benchmark tasks -- fine-grained activity understanding (keystep recognition and temporal segmentation), proficiency estimation, cross-view translation (ego-to-exo and exo-to-ego generation), and 3D hand/body pose estimation -- are designed to push the frontier of understanding how people perform skilled physical activities. Initial results demonstrate that these tasks are challenging and that the joint ego-exo setting provides unique opportunities for learning representations that transfer across viewpoints.

Ego-Exo4D presents significant implications for advancing AI in areas like augmented reality and robotics, where it could fuel advancements in learning from human demonstration. We believe the dataset will catalyze research on skill understanding, procedural reasoning, and cross-view learning, and we look forward to the community's contributions through the ongoing benchmark challenges.
