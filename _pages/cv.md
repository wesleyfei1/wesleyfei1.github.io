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

I am interested in generative recommendation, long-context memory for LLM agents, and efficient reasoning and retrieval mechanisms. Across these topics, I usually work in a finding-driven way: starting from concrete behavioral observations, then designing mechanisms that are concise, empirically grounded, and easy to reason about.

# Education

- **B.S. in Artificial Intelligence**, University of Science and Technology of China (USTC), School of the Gifted Young &nbsp; *Expected 2028*
- **GPA:** 3.87/4.3 &nbsp; **Rank:** 14/103
- **Selected Core Coursework:** Discrete Mathematics (100), Probability and Mathematical Statistics (92), Linear Algebra B(1) (90), Data Structures A (90)
- **Selected Self-Directed Coursework:** Stanford CS229, CS230, CS224n, MIT 6.S184, Berkeley CS61B


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
    Studied a consistent behavioral gap between "think" and "not-think" inference in generative recommendation, especially their differences in predictive entropy, popularity bias, and downstream recommendation quality. Based on these findings, developed an adaptive-thinking framework that selectively invokes reasoning only when uncertainty is high, aiming to balance effectiveness and inference cost. Work in preparation for submission to NeurIPS 2026.
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
    <strong>QQMem: Hierarchical Query-to-Query Retrieval for Long-Context Agent Memory</strong><br>
    Developing a memory retrieval framework for LLM agents motivated by the observation that direct episode retrieval is often semantically brittle in long-context settings. QQMem replaces episode-level matching with query-space alignment, using structured intermediate queries as semantic anchors to support more stable retrieval and grounded generation. Work in preparation for submission to NeurIPS 2026.
    </p>
  </div>
</div>

# Honors and Awards
- **Undergraduate Research Opportunities Program (UROP)**, Research on Generative Recommendation Systems based on Large Language Models, advised by [Prof. An Zhang](https://anzhang314.github.io/) &nbsp; *Dec 2025 -- Present*
- **First Prize (Provincial Level)**, The 17th Chinese Mathematics Competitions (Non-Math Major, top 20) &nbsp; *Oct 2025*
- **Silver Prize**, Outstanding Undergraduates Scholarship &nbsp; *Sept 2025*
- **Bronze Prize**, Outstanding Student Scholarship &nbsp; *Dec 2024*

# Skills
- **Research:** Literature review, experimental design, empirical analysis, and end-to-end implementation for machine learning research.
- **Programming:** Python, C, Java, Shell/Bash.
- **ML Frameworks:** PyTorch, Hugging Face, vLLM, verl.
- **Model Training:** Supervised fine-tuning (SFT), preference optimization (RLHF, DPO, RLvR), parameter-efficient tuning (LoRA), inference-time prompting and reasoning.
- **Tools:** Linux, Git, Conda, tmux, nvitop, Weights & Biases.
- **Engineering:** Rapid prototyping with LLM-assisted coding workflows and modern web tooling.
