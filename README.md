<div align="center">

# TIF-GRPO: Regulating Anatomy-Aware Rewards via Trajectory-Integral Feedback for Volumetric Computed Tomography Analysis

<p>
  Tianwei Lin<sup>1,2,*</sup>,
  Zhongwei Qiu<sup>2,3,1,*</sup>,
  Jie Cao<sup>1</sup>,
  Jiang Liu<sup>1</sup>,
  Wenjie Yan<sup>1</sup>,
  Bo Zhang<sup>4</sup>,
  Yu Zhong<sup>1</sup>,
  Wenqiao Zhang<sup>1,&#8224;</sup>,
  Yingda Xia<sup>2</sup>,
  Ling Zhang<sup>2,&#8224;</sup>
</p>

<p>
  <sup>1</sup>Zhejiang University &nbsp;|&nbsp;
  <sup>2</sup>DAMO Academy, Alibaba Group &nbsp;|&nbsp;
  <sup>3</sup>Hupan Lab &nbsp;|&nbsp;
  <sup>4</sup>University of Electronic Science and Technology of China
</p>

<p><sup>*</sup>Equal contribution &nbsp;&nbsp; <sup>&#8224;</sup>Corresponding authors</p>

<!-- [![Project Page](https://img.shields.io/badge/Project-README-1f6feb?style=flat-square)](#overview)
[![Paper Status](https://img.shields.io/badge/Paper-Research%20Manuscript-bf8700?style=flat-square)](#)
[![Code Status](https://img.shields.io/badge/Code-Coming%20Soon-6f42c1?style=flat-square)](#repository-status) -->

<p>
TIF-GRPO is a clinically grounded reinforcement learning framework for 3D CT MRG task.
It replaces surface-level proxy rewards with structured abnormality feedback and trajectory-integral regulation,
improving factuality, abnormality detection, and robustness in volumetric radiology report generation.
</p>

</div>

## Overview

Medical vision-language models (VLMs) for CT analysis are typically trained with supervised fine-tuning (SFT) based on next-token prediction over reference reports. This objective mainly encourages fitting dominant reporting patterns and surface-level language distributions, rather than explicitly grounding findings in image evidence. As a result, models can generate fluent and stylistically consistent reports while still hallucinating findings, missing subtle abnormalities, or producing anatomically inconsistent descriptions.

Subsequent reinforcement learning (RL) optimization often relies on lexical overlap rewards such as BLEU, ROUGE, or related text-similarity proxies, which further amplifies the mismatch between linguistic similarity and clinical correctness. Consequently, a report may achieve high text-level scores while still containing clinically dangerous errors, including incorrect locations, wrong attributes, or omission of critical abnormalities.


This project is built around two ideas:

| Component | Role |
| --- | --- |
| **CABS** | A **Clinical Abnormality Benchmarking Substrate** that decomposes reports into structured abnormality units: organ, disease entity, attributes, anatomical location, certainty, and evidence. |
| **TIF-GRPO** | A **Trajectory-Integral Feedback** version of GRPO that regulates reward assignment along a pseudo-temporal abnormality trajectory, explicitly penalizing persistent omissions and suppressing hallucinated findings. |

Together, they turn report optimization from surface-text matching into **fine-grained clinical credit assignment**.

## Motivation

<div align="center">
  <img src="images/motivation.png" alt="Motivation of TIF-GRPO: evaluation hallucinations and mechanistic divergence" width="100%">
</div>

Standard report-level metrics may assign high scores to outputs that are still clinically wrong.  
Our paper identifies two key failure modes:

1. **Evaluation Hallucination**: existing metrics overestimate clinical quality because they emphasize language similarity rather than verifiable medical facts.
2. **Mechanistic Divergence**: when these proxy metrics are used as RL rewards, optimization can actively drift away from real diagnostic correctness.

## Method

<div align="center">
  <img src="images/method.png" alt="Overview of the TIF-GRPO framework" width="100%">
</div>

TIF-GRPO introduces a clinically meaningful optimization loop:

1. A predicted report and the ground-truth report are both parsed by **CABS** into structured abnormality units.
2. These units define an **anatomy-aware reward space**, rather than a single undifferentiated text score.
3. A **trajectory-integral controller** accumulates false-negative signals across the abnormality trajectory while penalizing false positives as unnecessary control effort.
4. The resulting reward is injected into GRPO to produce more stable and more clinically faithful policy updates.

In short, TIF-GRPO encourages the model to be correct for the right reason:  
**detect the right abnormality, at the right anatomical location, with the right clinical attributes.**
