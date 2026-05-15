---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

<p>
  <a href="{{ base_path }}/files/CV.pdf" class="btn btn--primary" download>
    Download CV (PDF)
  </a>
</p>

I am interested in interpretable AI systems, with a focus on understanding and shaping the mechanisms behind LLM reasoning, agent memory, and human-centered recommendation. I am particularly interested in methods with clear mathematical or geometric structure that explain empirical phenomena, support verification, and make AI behavior more reliable and safe.

# Education

- **B.S. in Artificial Intelligence**, University of Science and Technology of China (USTC), School of the Gifted Young &nbsp; *Expected 2028*
- **GPA:** 3.87/4.3 &nbsp; **Rank:** 14/103
- **Selected Core Coursework:** Mathematical Analysis (97), Discrete Mathematics (100), Probability and Mathematical Statistics (92), Linear Algebra B(1) (90), Data Structures A (90)
- **Self-Directed AI/CS Study:** Since July 2025, completed full assignment tracks for Berkeley CS61B, Stanford CS229, CS230, and CS224n, with implementations documented in GitHub repositories; also studied MIT 6.S184.


# Research Experience

<div style="display: flex; align-items: flex-start; margin-bottom: 30px;">
  <div style="flex: 1; max-width: 120px;">
    <img src="../images/about/alphalab.png" style="width: 100%; height: auto;">
  </div>
  <div style="flex: 6; padding-left: 20px;">
    <p style="margin: 0;"><strong>Research Intern, Alpha-Lab, USTC</strong><br>
    Nov 2025 -- Present<br>
    Advisor: <a href="https://anzhang314.github.io/">Prof. An Zhang</a><br>
    <strong>Adaptive-Thinking for Generative Recommendation</strong><br>
    Investigating the behavioral gap between "think" and "not-think" inference in generative recommendation, motivated by the observation that explicit reasoning can hurt performance on simpler reasoning tasks. Studying whether the two inference modes rely on different recommendation signals, such as collaborative filtering patterns, popularity bias, user-item affinity, and semantic item descriptions.
    </p>
  </div>
</div>

<div style="display: flex; align-items: flex-start; margin-bottom: 30px;">
  <div style="flex: 1; max-width: 120px;">
    <img src="../images/about/ustc.jpg" style="width: 100%; height: auto;">
  </div>
  <div style="flex: 6; padding-left: 20px;">
    <p style="margin: 0;"><strong>Research Intern, USTC</strong><br>
    Mar 2026 -- Present<br>
    Advisor: <a href="https://xiangwang1223.github.io/">Prof. Xiang Wang</a><br>
    <strong>SHELF: From Similarity Retrieval to Path-Aware Auditable Memory for LLM Agents</strong> [Under Review at NeurIPS 2026]<br>
    Contributed to the theoretical formulation of SHELF, a path-aware auditable memory framework that writes facts into explicit structural addresses and retrieves evidence through query-conditioned paths. Designed the Auditor-Gater mechanism for variable-level path diagnosis; evaluated SHELF on LoCoMo with Qwen3-8B, Qwen3-32B, and DeepSeek-V3.2, achieving the best Overall F1, BLEU, and LLM-judge scores among compared memory baselines.
    </p>
  </div>
</div>

# Honors and Awards
- **Undergraduate Research Opportunities Program (UROP)**, Research on Generative Recommendation Systems based on Large Language Models, advised by [Prof. An Zhang](https://anzhang314.github.io/) &nbsp; *Dec 2025 -- Present*
- **First Prize (Provincial Level, Top 20)**, The 17th Chinese Mathematics Competitions, Non-Math Major Track &nbsp; *Oct 2025*
- **Silver Prize**, Outstanding Undergraduates Scholarship &nbsp; *Sept 2025*
- **Bronze Prize**, Outstanding Student Scholarship &nbsp; *Dec 2024*

# Skills
- **Research:** Paper reading and synthesis, experimental design, empirical analysis, ablation studies, and end-to-end implementation for ML research.
- **Programming:** Python, C, Java, Shell/Bash; familiar with Linux command-line workflows.
- **LLM Frameworks:** PyTorch, Hugging Face Transformers, vLLM, verl.
- **Training & Inference:** SFT with LoRA, RL training with verl, preference-optimization basics, inference-time prompting and reasoning.
- **Tools:** Linux, Git, Conda, tmux, nvitop, Weights & Biases.
- **Engineering:** Rapid prototyping, debugging, and LLM-assisted research implementation.
