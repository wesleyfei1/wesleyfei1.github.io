---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

I'm **Weihan Fei**, a sophomore undergraduate student at <a href="https://www.ustc.edu.cn/">the University of Science and Technology of China (USTC)</a>, School of the Gifted Young, majoring in Artificial Intelligence (GPA: 3.87/4.3, Rank: 14/103).

My research interests lie in **interpretable AI systems**, with a focus on understanding and shaping the mechanisms behind LLM reasoning, agent memory, and human-centered recommendation. I am particularly interested in methods with clear mathematical or geometric structure that explain empirical phenomena, support verification, and make AI behavior more reliable and safe.

I started systematic AI study in July 2025 and have completed full assignment tracks for **Berkeley CS61B**, **Stanford CS229**, **CS230**, and **CS224n**, with implementations documented in GitHub repositories. I also studied **MIT 6.S184 (Generative AI Foundations)** and have been reading broadly across LLM reasoning, recommendation, and agent memory.

# Research

**SHELF: From Similarity Retrieval to Path-Aware Auditable Memory for LLM Agents**  
Research Intern, USTC, advised by <a href="https://xiangwang1223.github.io/">Prof. Xiang Wang</a>  
**Under Review at NeurIPS 2026**

Through a broad survey of LLM-agent memory literature, I identified a coupled bottleneck in existing systems: structure is often used only after similarity retrieval, while reflection gives verbal failure feedback without specifying which access decision should be repaired. I contributed to the theoretical formulation of SHELF, a path-aware auditable memory framework that writes facts into explicit structural addresses and retrieves evidence through query-conditioned paths over entity, facet, time, relation, provenance, and coverage variables. SHELF was evaluated on LoCoMo with Qwen3-8B, Qwen3-32B, and DeepSeek-V3.2, achieving the best Overall F1, BLEU, and LLM-judge scores among compared memory baselines.

**Adaptive-Thinking for Generative Recommendation**  
Research Intern, <a href="https://anzhang314.github.io/">Alpha-Lab</a>, USTC, advised by <a href="https://anzhang314.github.io/">Prof. An Zhang</a>

I am investigating the behavioral gap between "think" and "not-think" inference in generative recommendation, motivated by the observation that explicit reasoning can hurt performance on simpler reasoning tasks. I am studying whether these two inference modes rely on different recommendation signals, such as collaborative filtering patterns, popularity bias, user-item affinity, and semantic item descriptions, and exploring criteria for deciding when reasoning should be invoked.

# Projects

**CoSwipe: AI Companion for Short-Video Feed Interaction**  
Third Prize, Track 1, Douyin AI Innovator Plan 2026 Hackathon, USTC Station

CoSwipe is an AI companion experience embedded in the short-video feed. Instead of being a generic emotional companion, it acts more like a browsing friend: it can notice what users just watched, playfully point out behavior contrasts, name latent needs behind surface interests, invite lightweight participation, and make subsequent content feel more personally connected. [[Demo](https://johnny-xuan.github.io/CoSwipe/)]

# News
- [May 2026] Our project **CoSwipe** won Third Prize in Track 1 at the Douyin AI Innovator Plan 2026 Hackathon, USTC Station.
- [May 2026] **SHELF** is under review at **NeurIPS 2026**.
- [Mar 2026] I started research on **SHELF: Path-Aware Auditable Memory for LLM Agents**, supervised by <a href="https://xiangwang1223.github.io/">Prof. Xiang Wang</a>.
- [Nov 2025] I joined <a href="https://anzhang314.github.io/">Alpha-Lab</a> at USTC, working on **Adaptive-Thinking for Generative Recommendation** under the supervision of <a href="https://anzhang314.github.io/">Prof. An Zhang</a>.
- [Oct 2025] I won the **First Prize (Provincial Level, Top 20)** at the 17th Chinese Mathematics Competitions, Non-Math Major Track.
- [Sept 2025] I was awarded the **Silver Prize** of the Outstanding Undergraduates Scholarship


<br><br><br>

<!-- Visitor tracker -->
<!-- {% include base_path %} -->
