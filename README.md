# 🔐 ProofLayer

**Verify human authorship with keystroke biometrics.**

ProofLayer detects whether text was typed by a real human or copy-pasted from AI/chatbots — in under 2 seconds, with no CAPTCHA, no privacy invasion, and zero friction.

---

## 🎯 The Problem

- **Professors** can't tell if essays are ChatGPT
- **Recruiters** get 500 AI-generated resumes per posting
- **Dating apps** are flooded with fake profiles
- **Content platforms** lose trust as AI content explodes

Current solutions (Turnitin, GPTZero) analyze **text**. We analyze **behavior** — how you type, not what you write.

---

## 🚀 How It Works

| Human Typing | AI Paste |
|---|---|
| Variable rhythm (pauses, backspaces) | Perfect consistency |
| 40-80 WPM with errors | 120+ WPM, zero errors |
| Natural dwell & flight times | Machine-precision timing |

Our classifier scores 0.0 (AI) to 1.0 (Human) based on:
- Keystroke dynamics (dwell time, flight time)
- Error correction rate
- Typing rhythm consistency
- Pause patterns

---

## 🛠️ Tech Stack

- **Backend:** Python + FastAPI + SQLite
- **ML:** scikit-learn (Isolation Forest) + heuristic rules
- **Crypto:** SHA-256 proofs + blockchain anchoring (Polygon testnet)
- **Frontend:** Vanilla HTML/JS (dark mode, responsive)
- **SDK:** JavaScript, 5-line embed

---

## 📦 Installation

```bash
# Clone repo
git clone https://github.com/YOURNAME/prooflayer.git
cd prooflayer

# Install dependencies
pip install -r requirements.txt

# Run server
python main.py
