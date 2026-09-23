# 🗣️ Mozhi
### *A phonetic correction layer for speech-to-text — because "we should meet at 3 pm" shouldn't turn into "we should meat at freepy."*

> Final Year Project — B.E. Computer Science and Business Systems, KPR Institute of Engineering and Technology

---

## 📖 What Is Mozhi?

Speech-to-text systems are impressively good at everyday language and impressively bad at the things that actually matter in professional contexts — **acronyms, proper nouns, and alphanumeric strings**. Say "GPU," "Coimbatore," or "IPv6" out loud to most STT engines and watch them guess something creative instead.

**Mozhi** is a correction layer that sits after any speech-to-text engine and fixes exactly these failure points, using a **pluggable JSON/YAML dictionary** so it can be adapted to different domains — medical, legal, technical — without retraining the underlying model.

*"Mozhi"* (மொழி) means **"language"** in Tamil — fitting, since the whole project is about making machines understand language a little more precisely.

---

## 🎯 The Problem, Precisely

Generic STT models are trained on generic speech. They're not trained on your company's product names, your professor's technical vocabulary, or your team's internal acronyms. That gap is where transcripts quietly go wrong — and it's usually the *important* words that get mangled, not the filler ones.

Mozhi targets that gap directly instead of trying to rebuild an entire STT pipeline from scratch.

---

## ✨ Features

- 🔤 **Acronym & proper noun correction** — post-processes STT output to fix the words generic models get wrong most
- 🔢 **Alphanumeric string handling** — model numbers, codes, and IDs come out the way they were actually said
- 🗂️ **Pluggable config dictionary (JSON/YAML)** — swap in a domain-specific correction set without touching the core logic
- 🧪 **Benchmarking suite** — measurable accuracy improvements, not just vibes
- 🐳 **Dockerized** — consistent environment for demos, testing, and (eventually) deployment
- 🖥️ **Streamlit demo app** — see the correction layer in action without digging through code

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| **Core language** | Python |
| **Demo interface** | Streamlit |
| **Containerization** | Docker |
| **Testing** | Custom test suite (`tests/`) |
| **Benchmarking** | Dedicated benchmark requirements & scripts |
| **Config** | JSON / YAML domain dictionaries |

---

## 📂 Project Structure

```
mozhi/
├── app/                        # Core correction logic
├── data/                       # Reference & domain dictionaries
├── demo/                       # Demo assets
├── reports/                    # Benchmark & evaluation reports
├── scripts/                    # Utility & processing scripts
├── tests/                      # Test suite
├── .streamlit/                 # Streamlit config
├── streamlit_app.py            # Demo application entry point
├── Dockerfile
├── requirements.txt
├── requirements-demo.txt
└── requirements-benchmark.txt
```

---


## 🎓 Why This Project Exists

This started as a Phase I academic proposal and grew into something with a real test suite, benchmarking, and a working demo — the kind of scope that's meant to hold up under both a viva panel and a GitHub reader. The framing throughout has been deliberate: solve a narrow, well-defined problem (acronyms, proper nouns, alphanumeric strings) really well, rather than promise a general-purpose STT overhaul that a semester timeline could never deliver.

---
## 🤝 Team Project
 
Mozhi is a collaborative Final Year Project, built and iterated on with teammates — not a solo effort.
 
### Teammates
 
- [Prabakaran S R](https://github.com/Prabakaransr19)
- [Sarvatarshan Sankar](https://github.com/sarva-20)
- [Rakesh Pranav](https://github.com/)
- [Danush Aditya](https://github.com/DanushAdtiya)

### Built With
 
- [Claude Code](https://claude.com/product/claude-code) — Opus 4.5 & Sonnet, for planning and building

---


## 📄 License

Academic project — built for coursework and research purposes.

---

<p align="center">
  <i>Because the words that matter most are usually the ones speech-to-text gets wrong.</i>
</p>