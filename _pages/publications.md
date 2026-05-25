---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

# Publications and Preprints

My research focuses on interpretable AI systems: understanding and shaping the mechanisms behind LLM reasoning, agent memory, and human-centered recommendation. I am particularly interested in methods with clear mathematical or geometric structure that explain empirical phenomena, support verification, and make AI behavior more reliable and safe.

- **SHELF: From Similarity Retrieval to Path-Aware Auditable Memory for LLM Agents**  
  Weihan Fei, advised by Prof. Xiang Wang  
  *Under Review at NeurIPS 2026*  
  <span style="font-family: 'Times New Roman';"><i>-- Path-aware auditable memory for LLM agents, with explicit retrieval paths, variable-level diagnosis, and targeted path revision.</i></span>

# Work in Progress

- **Adaptive-Thinking for Generative Recommendation**  
  Weihan Fei, advised by Prof. An Zhang  
  *Ongoing research*  
  <span style="font-family: 'Times New Roman';"><i>-- Investigating when explicit reasoning helps or hurts generative recommendation, and whether think/not-think inference relies on different recommendation signals.</i></span>

# Selected Projects

- **CoSwipe: AI Companion for Short-Video Feed Interaction**  
  Third Prize, Track 1, Douyin AI Innovator Plan 2026 Hackathon, USTC Station  
  <span style="font-family: 'Times New Roman';"><i>-- An AI companion experience embedded in the short-video feed, designed to notice browsing behavior, respond like a friend, and make downstream content feel more personally connected.</i></span> [[Demo](https://johnny-xuan.github.io/CoSwipe/)]


{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
