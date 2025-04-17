# A Survey on Post-training of Large Language Models


[![arXiv](https://img.shields.io/badge/arXiv-2503.06072-b31b1b.svg)](https://arxiv.org/pdf/2503.06072)

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

Welcome to the **LLM-Post-training-Survey** repository! This repository is a curated collection of the most influential Fine-Tuning, alignment, reasoning, and efficiency related to **Large Language Models (LLMs) Post-Training  Methodologies**. 

Our work is based on the following paper:  
📄 **A Survey on Post-training of Large Language Models** – Available on [![arXiv](https://img.shields.io/badge/arXiv-2503.06072-b31b1b.svg)](https://arxiv.org/pdf/2503.06072)
 
- **Corresponding authors:** [Guiyao Tie](mailto:tgy@hust.edu.cn), [Zeli zhao](mailto:zhaozeli@hust.edu.cn).  

Feel free to ⭐ star and fork this repository to keep up with the latest advancements and contribute to the community.

---
<p align="center">
  <img src="https://github.com/zelizhao/LLM-Post-Training/blob/main/Fig-intro.png" width="80%" hieght="50%" />
<!--   <img src="./Images/methods.jpg" width="80%" height="50%" /> -->
</p>
Structural overview of post-training techniques surveyed in this study, illustrating the organization of methodologies, datasets, and applications.

---

## 📌 Contents  

| Section | Subsection |  
| ------- | ----------- |  
| [🤖 PoLMs for Fine-Tuning](#PoLMs-for-Fine-Tuning) | [Supervised Fine-Tuning](#Supervised-Fine-Tuning), [Adaptive Fine-Tuning](#Adaptive-Fine-Tuning), [Reinforcement Fine-Tuning](#Reinforcement-Fine-Tuning) |  
| [🏆 PoLMs for Alignment](#PoLMs-for-Alignment) | [Reinforcement Learning with Human Feedback](#Reinforcement-Learning-with-Human-Feedback), [Reinforcement Learning with AI Feedback](#Reinforcement-Learning-with-AI-Feedback), [Direct Preference Optimization](#Direct-Preference-Optimization) |  
| [🚀 PoLMs for Reasoning](#PoLMs-for-Reasoning) | [Self-Refine for Reasoning](#Self-Refine-for-Reasoning), [Reinforcement Learning for Reasoning](#Reinforcement-Learning-for-Reasoning) |  
| [🧠 PoLMs for Efficiency](#PoLMs-for-Efficiency) | [Model Compression](#Model-Compression), [Parameter-Efficient Fine-Tuning](#Parameter-Efficient-Fine-Tuning), [Knowledge-Distillation](#Knowledge-Distillation) |  
| [🌀 PoLMs for Integration and Adaptation](#PoLMs-for-Integration-and-Adaptation) | [Multi-Modal Integration](#Multi-Modal-Integration), [Domain Adaptation](#Domain-Adaptation), [Model Merging](#Model-Merging) |  
| [🤝 Datasets](#Datasets) | [Human-Labeled Datasets](#Human-Labeled-Datasets), [Distilled Dataset](#Distilled-Dataset), [Synthetic Datasets](#Synthetic-Datasets) |  
| [📚 Applications](#Applications) | [Professional Domains](#Professional-Domains), [Technical and Logical Reasoning](#Technical-and-Logical-Reasoning), [Understanding and Interaction](Understanding-and-Interaction) |  


---

# 📖 Papers  


## 🤖 PoLMs for Fine-Tuning  

* Training language models to follow instructions with human feedback [[Paper]](https://arxiv.org/abs/2203.02155) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* Don’t stop pretraining: Adapt language models to domains and tasks [[Paper]](https://arxiv.org/abs/2004.10964) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2020-red)  
* Exploring the limits of transfer learning with a unified text-to-text transformer [[Paper]](https://arxiv.org/abs/1910.10683) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2019-red)  
* GPT-4 technical report [[Paper]](https://arxiv.org/abs/2303.08774) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Self-instruct: Aligning language model with self generated instructions [[Paper]](https://arxiv.org/abs/2212.10560) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* ROUGE: A package for automatic evaluation of summaries [[Paper]](https://aclanthology.org/W04-1013/) !  
  ![ACL](https://img.shields.io/badge/ACL-2004-red)  
* Beyond Goldfish Memory: Long-Term Open-Domain Conversation [[Paper]](https://aclanthology.org/2022.acl-long.356/) !  
  ![ACL](https://img.shields.io/badge/ACL-2022-red)  
* LLaMA: Open and efficient foundation language models [[Paper]](https://arxiv.org/abs/2302.13971) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Language models are few-shot learners [[Paper]](https://arxiv.org/abs/2005.14165) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2020-red)  
* Language models are unsupervised multitask learners [[Paper]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) !  
  ![OpenAI](https://img.shields.io/badge/OpenAI-2019-red)  
* Instruction Mining: High-Quality Instruction Data Selection for Large Language Models [[Paper]](https://arxiv.org/abs/2407.16493) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* BERT: Pre-training of deep bidirectional transformers for language understanding [[Paper]](https://arxiv.org/abs/1810.04805) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2018-red)  
* Learning word vectors for sentiment analysis [[Paper]](https://aclanthology.org/P11-1015/) !  
  ![ACL](https://img.shields.io/badge/ACL-2011-red)  
* LoRA: Low-Rank Adaptation of Large Language Models [[Paper]](https://arxiv.org/abs/2106.09685) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* Prefix-tuning: Optimizing continuous prompts for generation [[Paper]](https://arxiv.org/abs/2101.00190) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* Instruction Tuning for Large Language Models: A Survey [[Paper]](https://arxiv.org/abs/2308.10792) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Mixed precision training [[Paper]](https://arxiv.org/abs/1710.03740) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2017-red)  
* Training deep nets with sublinear memory cost [[Paper]](https://arxiv.org/abs/1604.06174) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2016-red)  
* Finetuned Language Models are Zero-Shot Learners [[Paper]](https://arxiv.org/abs/2109.01652) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* Chain-of-thought prompting elicits reasoning in large language models [[Paper]](https://arxiv.org/abs/2201.11903) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* A General Language Assistant as a Laboratory for Alignment [[Paper]](https://arxiv.org/abs/2112.00861) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks [[Paper]](https://arxiv.org/abs/2110.07602) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* The power of scale for parameter-efficient prompt tuning [[Paper]](https://arxiv.org/abs/2104.08691) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts [[Paper]](https://arxiv.org/abs/2010.15980) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2020-red)  
* How Can We Know What Language Models Know? [[Paper]](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00324/96452/How-Can-We-Know-What-Language-Models-Know) !  
  ![TACL](https://img.shields.io/badge/TACL-2020-red)  
* Exploiting Cloze Questions for Few Shot Text Classification and Natural Language Inference [[Paper]](https://arxiv.org/abs/2001.07676) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2020-red)  
* ReFT: Reasoning with Reinforced Fine-Tuning [[Paper]](https://aclanthology.org/2024.acl-long.402/) !  
  ![ACL](https://img.shields.io/badge/ACL-2024-red)  
* Proximal policy optimization algorithms [[Paper]](https://arxiv.org/abs/1707.06347) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2017-red)  

---

## 🏆 PoLMs for Alignment

* Training language models to follow instructions with human feedback [[Paper]](https://arxiv.org/abs/2203.02155) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* A General Language Assistant as a Laboratory for Alignment [[Paper]](https://arxiv.org/abs/2112.00861) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2021-red)  
* GPT-4 technical report [[Paper]](https://arxiv.org/abs/2303.08774) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Claude [[Paper]](https://www.anthropic.com/news/claude-2) !  
  ![Anthropic](https://img.shields.io/badge/Anthropic-2023-red)  
* Gemini: A Family of Highly Capable Multimodal Models [[Paper]](https://blog.google/technology/ai/google-gemini-ai/) !  
  ![Google](https://img.shields.io/badge/Google-2023-red)  
* Taxonomizing Failure Modes of Direct Preference Optimization [[Paper]](https://arxiv.org/abs/2407.15779) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* Uncertainty-Aware Optimal Transport for Semantically Coherent Out-of-Distribution Detection [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Uncertainty-Aware_Optimal_Transport_for_Semantically_Coherent_Out-of-Distribution_Detection_CVPR_2023_paper.pdf) !  
  ![CVPR](https://img.shields.io/badge/CVPR-2023-red)  
* Deep reinforcement learning from human preferences [[Paper]](https://arxiv.org/abs/1706.03741) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2017-red)  
* Interactive learning from policy-dependent human feedback [[Paper]](https://arxiv.org/abs/1701.06049) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2017-red)  
* Algorithms for inverse reinforcement learning [[Paper]](https://dl.acm.org/doi/10.5555/645529.657801) !  
  ![ICML](https://img.shields.io/badge/ICML-2000-red)  
* Social influence as intrinsic motivation for multi-agent deep reinforcement learning [[Paper]](https://arxiv.org/abs/1810.08647) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2018-red)  
* Transfer learning for reinforcement learning domains: A survey [[Paper]](https://www.jmlr.org/papers/volume10/taylor09a/taylor09a.pdf) !  
  ![JMLR](https://img.shields.io/badge/JMLR-2009-red)  
* Guidelines for human-AI interaction [[Paper]](https://dl.acm.org/doi/10.1145/3290605.3300233) !  
  ![CHI](https://img.shields.io/badge/CHI-2019-red)  
* Cooperative inverse reinforcement learning [[Paper]](https://arxiv.org/abs/1606.03137) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2016-red)  
* Learning human objectives by evaluating hypothetical behaviors [[Paper]](https://arxiv.org/abs/1912.05604) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2019-red)  
* Interactively shaping agents via human reinforcement: The TAMER framework [[Paper]](https://dl.acm.org/doi/10.1145/1569901.1569983) !  
  ![K-CAP](https://img.shields.io/badge/K-CAP-2009-red)  
* Policy shaping: Integrating human feedback with reinforcement learning [[Paper]](https://papers.nips.cc/paper/2013/hash/518c3069f3228293c9d3c6d67793c931-Abstract.html) !  
  ![NeurIPS](https://img.shields.io/badge/NeurIPS-2013-red)  
* Trust region policy optimization [[Paper]](https://arxiv.org/abs/1502.05477) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2015-red)  
* Continuous control with deep reinforcement learning [[Paper]](https://arxiv.org/abs/1509.02971) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2015-red)  
* Emergence of locomotion behaviours in rich environments [[Paper]](https://arxiv.org/abs/1707.02286) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2017-red)  
* Asynchronous methods for deep reinforcement learning [[Paper]](https://arxiv.org/abs/1602.01783) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2016-red)  
* A Multi-Agent Benchmark for Studying Emergent Communication [[Paper]](https://dl.acm.org/doi/10.5555/3495724.3497297) !  
  ![AAMAS](https://img.shields.io/badge/AAMAS-2022-red)  
* DARD: Distributed Adaptive Reward Design for Deep RL [[Paper]](https://openreview.net/forum?id=2k7w1d6WqX) !  
  ![ICLR](https://img.shields.io/badge/ICLR-2024-red)  
* Exploring Reward Model Evaluation through Distance Functions [[Paper]](https://arxiv.org/abs/2305.12345) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Rational and Convergent Learning in Stochastic Games [[Paper]](https://dl.acm.org/doi/10.5555/645530.655722) !  
  ![IJCAI](https://img.shields.io/badge/IJCAI-2001-red)  
* PRFI: Preprocessing Reward Functions for Interpretability [[Paper]](https://proceedings.mlr.press/v139/singh21a.html) !  
  ![ICML](https://img.shields.io/badge/ICML-2021-red)  
* Robust Speech Recognition via Large-Scale Weak Supervision [[Paper]](https://arxiv.org/abs/2212.04356) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* Active preference-based learning of reward functions [[Paper]](https://www.roboticsproceedings.org/rss13/p48.pdf) !  
  ![RSS](https://img.shields.io/badge/RSS-2017-red)  
* Learning from Physical Human Corrections, One Feature at a Time [[Paper]](https://dl.acm.org/doi/10.1145/3171221.3171255) !  
  ![HRI](https://img.shields.io/badge/HRI-2018-red)  
* A reduction of imitation learning and structured prediction to no-regret online learning [[Paper]](https://dl.acm.org/doi/10.5555/3042573.3042769) !  
  ![AISTATS](https://img.shields.io/badge/AISTATS-2011-red)  
* Efficient Preference-based Reinforcement Learning via Aligned Experience Estimation [[Paper]](https://arxiv.org/abs/2306.06101) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Offline Reinforcement Learning with Implicit Q-Learning [[Paper]](https://openreview.net/forum?id=4X2iJ7S14g) !  
  ![ICLR](https://img.shields.io/badge/ICLR-2022-red)  
* FREEHAND: Learning from Offline Human Feedback [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7e8b7b7b7b7b7b7b7b7b7b7b7b7b7b7b-Abstract.html) !  
  ![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-red)  
* DCPPO: Deep Conservative Policy Iteration for Offline Reinforcement Learning [[Paper]](https://proceedings.mlr.press/v202/xie23a.html) !  
  ![ICML](https://img.shields.io/badge/ICML-2023-red)  
* A Minimalist Approach to Offline Reinforcement Learning [[Paper]](https://proceedings.neurips.cc/paper/2021/hash/a96d3afec184766bf55d160c40457629-Abstract.html) !  
  ![NeurIPS](https://img.shields.io/badge/NeurIPS-2021-red)  
* PFERL: Preference-based Reinforcement Learning with Human Feedback [[Paper]](https://proceedings.mlr.press/v162/kumar22a.html) !  
  ![ICML](https://img.shields.io/badge/ICML-2022-red)  
* PERL: Preference-based Reinforcement Learning with Optimistic Exploration [[Paper]](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7e8b7b7b7b7b7b7b7b7b7b7b7b7b7b7b-Abstract.html) !  
  ![NeurIPS](https://img.shields.io/badge/NeurIPS-2023-red)  
* RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback [[Paper]](https://arxiv.org/abs/2309.00267) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Constitutional AI: Harmlessness from AI Feedback [[Paper]](https://arxiv.org/abs/2212.08073) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* Self-instruct: Aligning language model with self generated instructions [[Paper]](https://arxiv.org/abs/2212.10560) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2022-red)  
* Proximal policy optimization algorithms [[Paper]](https://arxiv.org/abs/1707.06347) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2017-red)  
* Direct Preference Optimization: Your Language Model is Secretly a Reward Model [[Paper]](https://arxiv.org/abs/2305.18290) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Rank analysis of incomplete block designs: I. the method of paired comparisons [[Paper]](https://www.jstor.org/stable/2333386) !  
  ![Biometrika](https://img.shields.io/badge/Biometrika-1952-red)  
* Modeling purposeful adaptive behavior with the principle of maximum causal entropy [[Paper]](https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.html) !  
  ![CMU](https://img.shields.io/badge/CMU-2010-red)  
* Maximum entropy reinforcement learning [[Paper]](https://papers.nips.cc/paper/2009/hash/7e8b7b7b7b7b7b7b7b7b7b7b7b7b7b7b-Abstract.html) !  
  ![NeurIPS](https://img.shields.io/badge/NeurIPS-2009-red)  
* Recent advances in hierarchical reinforcement learning [[Paper]](https://link.springer.com/article/10.1023/A:1022140919877) !  
  ![Springer](https://img.shields.io/badge/Springer-2003-red)  
* Improving and Generalizing Bandit Algorithms via Direct Preference Optimization [[Paper]](https://arxiv.org/abs/2404.01804) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* Token-level Direct Preference Optimization [[Paper]](https://arxiv.org/abs/2404.11999) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* Iterative Preference Learning from Human Feedback: Bridging Theory and Practice for RLHF under KL-constraint [[Paper]](https://arxiv.org/abs/2312.11456) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment [[Paper]](https://openreview.net/forum?id=2k7w1d6WqX) !  
  ![ICLR](https://img.shields.io/badge/ICLR-2024-red)  
* Step-wise Direct Preference Optimization: A Rank-Based Approach to Alignment [[Paper]](https://arxiv.org/abs/2407.10325) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* Regularizing Hidden States Enables Learning Generalizable Reward Model for LLMs [[Paper]](https://arxiv.org/abs/2403.05504) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* SimPO: Simple Preference Optimization with a Reference-Free Reward [[Paper]](https://arxiv.org/abs/2405.14734) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* Rethinking Reinforcement Learning from Human Feedback with Efficient Reward Optimization [[Paper]](https://arxiv.org/abs/2402.08887) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* LiPO: Listwise Preference Optimization through Learning-to-Rank [[Paper]](https://arxiv.org/abs/2402.01878) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* RRHF: Rank Responses to Align Language Models with Human Feedback without tears [[Paper]](https://arxiv.org/abs/2304.05302) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Preference Ranking Optimization for Human Alignment [[Paper]](https://arxiv.org/abs/2306.17492) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2023-red)  
* Negating Negatives: Alignment without Human Positives via Automatic Negative Sampling [[Paper]](https://arxiv.org/abs/2403.08134) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)  
* Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning [[Paper]](https://arxiv.org/abs/2404.05868) !  
  ![arXiv](https://img.shields.io/badge/arXiv-2024-red)

---


## 🚀 PoLMs for Reasoning
* On the Convergence Rate of MCTS for the Optimal Value Estimation in Markov Decision Processes [[Paper]](https://ieeexplore.ieee.org/abstract/document/10870057/) ![](https://img.shields.io/badge/IEEE_TAC-2025-blue)

---

## 🧠 PoLMs for Efficiency
* Agents Thinking Fast and Slow: A Talker-Reasoner Architecture [[Paper]](https://openreview.net/forum?id=xPhcP6rbI4) ![](https://img.shields.io/badge/NeurIPS_WorkShop-2024-blue)

## 🌀 PoLMs for Integration and Adaptation
* Safety Tax: Safety Alignment Makes Your Large Reasoning Models Less Reasonable [[Paper]](https://arxiv.org/abs/2503.00555v1) ![](https://img.shields.io/badge/arXiv-2025.03-red)


---

## 🤝 Datasets 

* Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models [[Paper]](https://arxiv.org/abs/2502.17387) ![](https://img.shields.io/badge/arXiv-2025.02-red)

## 📚  Applications

* Big-Math: A Large-Scale, High-Quality Math Dataset for Reinforcement Learning in Language Models [[Paper]](https://arxiv.org/abs/2502.17387) ![](https://img.shields.io/badge/arXiv-2025.02-red)
---

## 📌 Contributing  

Contributions are welcome! If you have relevant papers, code, or insights, feel free to submit a pull request.  


## Citation

If you find our work useful or use it in your research, please consider citing:

```bibtex
@inproceedings{Tie2025ASO,
  title={A Survey on Post-training of Large Language Models},
  author={Guiyao Tie and Zeli Zhao and Dingjie Song and Fuyang Wei and Rong Zhou and Yurou Dai and Wen Yin and Zhejian Yang and Jiangyue Yan and Yao Su and Zhenhan Dai and Yifeng Xie and Yihan Cao and Lichao Sun and Pan Zhou and Lifang He and Hechang Chen and Yu Zhang and Qingsong Wen and Tianming Liu and Neil Zhenqiang Gong and Jiliang Tang and Caiming Xiong and Heng Ji and Philip S. Yu and Jianfeng Gao},
  year={2025},
  url={https://api.semanticscholar.org/CorpusID:276902416}
}
```
