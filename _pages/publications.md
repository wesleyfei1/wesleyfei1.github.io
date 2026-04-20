---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

# Work in Progress

Here are my current research projects. I focus on finding-driven research: starting from observable model behavior, then developing targeted and concise mechanisms to improve performance and reliability. Papers will be updated upon acceptance.

- **Adaptive-Thinking for Generative Recommendation**  
  Weihan Fei, advised by Prof. An Zhang  
  *In preparation for NeurIPS 2026*  
  <span style="font-family: 'Times New Roman';"><i>-- Balancing effectiveness and inference cost via selective reasoning in generative recommendation.</i></span>

- **QQMem: Hierarchical Query-to-Query Retrieval for Long-Context Agent Memory**  
  Weihan Fei, advised by Prof. Xiang Wang  
  *In preparation for NeurIPS 2026*  
  <span style="font-family: 'Times New Roman';"><i>-- Query-space alignment for more stable memory retrieval and grounded generation in long-context LLM agents.</i></span>


{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}
