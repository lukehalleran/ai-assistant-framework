# Daemon RAG Agent

A near production-grade conversational AI system with hierarchical memory, semantic search, and Wikipedia-scale knowledge retrieval — **built from first principles over 8 months (part-time)**.

<div align="center">

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker Ready](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

*Built from first principles over 8 months (part-time) • 54,000+ lines of code • 195 files*

[Features](#features) • [Quick Start](#quick-start) • [Architecture](#architecture) • [Usage](#usage) • [Deployment](#deployment)

</div>

---

## Overview

Daemon is a **full-stack RAG (Retrieval-Augmented Generation) system** implementing a complete conversational AI pipeline with a cognitive-inspired memory architecture. It was built entirely from first principles—before discovering LangChain or LlamaIndex—as a way to understand every component in a modern AI assistant.

**What makes Daemon different:**

- **5-tier hierarchical memory** modeled on human cognition (episodic, semantic, procedural, summary, meta)
- **Multi-stage relevance filtering** using FAISS → cosine similarity → cross-encoder reranking
- **Wikipedia-scale knowledge** with 6.5M+ articles semantically indexed
- **Crisis-aware tone detection** that adapts response depth to emotional context
- **Chain-of-thought reasoning** via “thinking blocks” for transparent decision-making

> For a compressed, architecture-focused overview, see [PROJECT_SKELETON.md](./PROJECT_SKELETON.md).

---

## Screenshots 

<img width="1919" height="1071" alt="2025-07-28_20-48" src="https://github.com/user-attachments/assets/eac216c2-b9be-4e5b-a799-cc0a41d80266" />

This is assistant responding to me on a fresh start up, demonstrating tone and memory persistence.

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Hierarchical Memory** | 5-tier cognitive architecture with temporal decay and access reinforcement |
| **Hybrid Retrieval** | Combines recent context + semantic search for balanced recall |
| **Multi-Stage Gating** | FAISS → cosine → cross-encoder pipeline (~200ms total) |
| **STM Analyzer** | Short-term memory compression reduces redundant context |
| **Thinking Blocks** | Chain-of-thought reasoning with transparent decision logs |
| **Tone Detection** | Crisis-aware response adaptation (HIGH/MEDIUM/CONCERN/CONVERSATIONAL) |
| **Multi-Provider LLM** | OpenAI, Anthropic, DeepSeek, Google, and local models |
|
