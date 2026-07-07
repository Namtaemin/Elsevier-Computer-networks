# DDS-over-TSN Framework for Time-Critical Applications in Industrial Metaverses

> **Cross-Layer QoS Mapping with RAG-driven CQM, RoBERTa-based SPE, and PPO-driven ARG for Intelligent Resource Management in Industrial Networks**

---

## Overview

This repository contains the official implementation of the simulation framework proposed in the paper:

> **"Cross-Layer QoS Mapping for DDS-over-TSN in Time-Critical Industrial Metaverse Applications"**  
> *Taemin Nam et al., Korea University of Technology and Education (KOREATECH)*

The framework integrates **Data Distribution Service (DDS)** middleware over **Time-Sensitive Networking (TSN)** to enable deterministic, low-latency communication in industrial metaverse environments. Three core modules are implemented:

- **CQM (Cross-layer QoS Mapper)** — RAG (Retrieval-Augmented Generation) driven QoS policy inference
- **SPE (Semantic Policy Encoder)** — RoBERTa-based hidden state feature extraction for QoS semantics
- **ARG (Adaptive Resource Governor)** — PPO (Proximal Policy Optimization) based deep reinforcement learning for dynamic resource scheduling

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               Industrial Metaverse Application           │
├─────────────────────────────────────────────────────────┤
│  DDS Layer  │  QoS Profiles  │  CQM (RAG-driven)        │
├─────────────────────────────────────────────────────────┤
│  SPE (RoBERTa)  │  Semantic Feature Extraction           │
├─────────────────────────────────────────────────────────┤
│  ARG (PPO-DRL)  │  Adaptive Resource Scheduling          │
├─────────────────────────────────────────────────────────┤
│  TSN Layer  │  IEEE 802.1Qbv/Qav  │  Time-Aware Shaper  │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
.
├── Experiment_CASE1/       # Baseline: Priority Queuing (PQ)
├── Experiment_CASE2/       # DNN-based scheduling
├── Experiment_CASE3/       # DRL (PPO) scheduling
├── Experiment_CASE4/       # DRL + CQM integration
├── Experiment_CASE5/       # DRL + SPE (RoBERTa) integration
├── Experiment_CASE6/       # Full framework (CQM + SPE + ARG)
├── Experiment_CASE7/       # Ablation: varying traffic load
├── Experiment_CASE8/       # Ablation: varying network topology
├── Experiment_CASE9/       # Ablation: varying QoS constraint levels
├── Combination_BEx/        # Combined baseline experiment results
├── Each_Loss_BEx/          # Per-component loss curve analysis
└── Loss_RL_BEx/            # RL training loss benchmarks
```

---

## Key Modules

### CQM — Cross-layer QoS Mapper (RAG-driven)
Retrieves relevant QoS policy knowledge from a vector database and maps application-level QoS requirements (latency, reliability, bandwidth) to TSN scheduling parameters (gate control lists, traffic classes).

### SPE — Semantic Policy Encoder (RoBERTa-based)
Extracts hidden-state semantic embeddings from QoS policy descriptions using a fine-tuned RoBERTa model. The encoded representations are used to condition the RL agent's policy network.

### ARG — Adaptive Resource Governor (PPO-driven)
A deep reinforcement learning agent trained with PPO to dynamically allocate TSN resources. The agent observes network state and QoS requirements, then outputs scheduling decisions to minimize end-to-end latency while satisfying deadline constraints.

---

## Experimental Setup

Simulations were conducted using **OMNeT++ / INET Framework**, modeling a TSN-based industrial network with:

- Multiple traffic classes (TT, AVB Class A/B, BE)
- Dynamic publisher/subscriber topologies
- Varying QoS constraint profiles (strict, moderate, best-effort)

Each `Experiment_CASE` folder contains:
- Simulation configuration files (`.ini`, `.ned`)
- Training logs and reward curves
- Result CSV files for latency, jitter, and packet loss metrics

---

## Requirements

```
Python >= 3.8
PyTorch >= 1.13
Transformers (HuggingFace) >= 4.x
OMNeT++ >= 6.0 (for network simulation)
INET Framework >= 4.4
```

Install Python dependencies:

```bash
pip install torch transformers sentence-transformers stable-baselines3
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nam2024dds_tsn,
  title     = {Cross-Layer QoS Mapping for DDS-over-TSN in Time-Critical Industrial Metaverse Applications},
  author    = {Taemin Nam and others},
  journal   = {IEEE Transactions on Consumer Electronics},
  year      = {2024}
}
```

---

## License

This project is released for academic research purposes.  
© 2024 KOREATECH Future Convergence Engineering Lab. All rights reserved.
