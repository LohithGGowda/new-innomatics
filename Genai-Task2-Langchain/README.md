# 🤖 AI Resume Screening System

> Built with **LangChain** + **LangSmith** | GenAI Internship Assignment

---

## 📌 Overview

An AI-powered resume screening pipeline that evaluates candidates against a job description using LLMs. The system provides **scoring**, **skill matching**, and **explainability** — with full **LangSmith tracing** for debugging and monitoring.

### Pipeline Architecture

```
Resume → Skill Extraction → Matching → Scoring → Explanation → LangSmith Traces
```

---

## 🗂️ Project Structure

```
ai_resume_screening/
├── main.py                        # Entry point — run this
├── resume_screening.ipynb         # Jupyter Notebook version
├── requirements.txt               # Dependencies
├── .env.example                   # Copy this → .env
│
├── prompts/
│   ├── __init__.py
│   └── resume_prompts.py          # All PromptTemplates (4 stages)
│
├── chains/
│   ├── __init__.py
│   └── screening_chains.py        # LCEL chains + pipeline orchestrator
│
├── data/
│   ├── __init__.py
│   └── sample_data.py             # 3 resumes + job description
│
├── results/                       # Auto-created — stores JSON output
└── screenshots/                   # Add LangSmith screenshots here
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ai-resume-screening.git
cd ai-resume-screening
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
```bash
cp .env.example .env
```

Edit `.env` with your keys:
```env
OPENAI_API_KEY=sk-your-openai-key
LANGCHAIN_API_KEY=ls__your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ai-resume-screening
```

**Get your keys:**
- OpenAI: https://platform.openai.com/api-keys
- LangSmith: https://smith.langchain.com/settings

### 4. Run the pipeline
```bash
python main.py
```

Or open the Jupyter notebook:
```bash
jupyter notebook resume_screening.ipynb
```

---

## 🔄 Pipeline Details

### Stage 1: Skill Extraction
- Extracts: skills, experience years, tools, programming languages, ML frameworks, cloud platforms, certifications
- Anti-hallucination rule: **only extract what is explicitly in the resume**

### Stage 2: Matching
- Compares extracted profile vs. job description requirements
- Identifies: matched requirements, missing requirements, key strengths, key gaps

### Stage 3: Scoring (0–100)
```
80–100  Excellent  → Strongly Recommend
60–79   Good       → Recommend
40–59   Average    → Consider
20–39   Below Avg  → Not Recommended
0–19    Poor       → Not Recommended
```

Score breakdown:
- Technical Skills: `/40`
- Experience: `/25`
- Education: `/15`
- Tools & Frameworks: `/20`

### Stage 4: Explanation
- Generates a human-readable report with:
  - Summary
  - Strengths (evidence-based)
  - Gaps (specific)
  - Hiring recommendation with reasoning

---

## 📊 LangSmith Tracing

All runs are automatically traced when `LANGCHAIN_TRACING_V2=true`.

### Traces Created
| Run | Candidate | Tag |
|-----|-----------|-----|
| 1   | Aisha Patel (Strong) | `strong`, `run-1` |
| 2   | Rahul Sharma (Average) | `average`, `run-2` |
| 3   | Vikram Singh (Weak) | `weak`, `run-3` |
| Debug | Buzz Wordsmith | `debug`, `edge-case` |

### View traces at:
👉 https://smith.langchain.com → Projects → `ai-resume-screening`

Each trace shows:
- All 4 pipeline stages as nested spans
- Input/output at every step
- Token usage and latency
- Debug: hallucination guard test

---

## 🧪 Sample Output

```
Candidate  : Aisha Patel
Score      : 92/100  [██████████████████░░]
Category   : Excellent
Decision   : Strongly Recommend

Candidate  : Rahul Sharma
Score      : 62/100  [████████████░░░░░░░░]
Category   : Good
Decision   : Recommend

Candidate  : Vikram Singh
Score      : 12/100  [██░░░░░░░░░░░░░░░░░░]
Category   : Poor
Decision   : Not Recommended
```

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM Framework | LangChain (LCEL) |
| LLM Provider | OpenAI (GPT-4o-mini) |
| Tracing/Monitoring | LangSmith |
| Prompt Design | PromptTemplate + Few-shot |
| Output Format | JSON (structured) |
| Notebook | Jupyter |

---

## ✅ Features

- [x] Modular pipeline (`prompts/`, `chains/`, `main.py`)
- [x] 3 candidate profiles (Strong, Average, Weak)
- [x] LangSmith tracing with `@traceable` decorator
- [x] Minimum 3 traced runs
- [x] Debug trace (hallucination guard test)
- [x] Structured JSON output at each stage
- [x] Few-shot prompting in explanation stage
- [x] Anti-hallucination rules in all prompts
- [x] Score breakdown (4 dimensions)
- [x] Automated results saved to JSON

---

## 📝 Evaluation Criteria Mapping

| Criterion | Implementation |
|-----------|---------------|
| Pipeline Design (20%) | 4-stage LCEL pipeline in `chains/screening_chains.py` |
| LangChain Implementation (20%) | PromptTemplate + LCEL pipes + `.invoke()` |
| Scoring & Logic (15%) | Rubric-based scoring with breakdown in `scoring_prompt` |
| Explainability (15%) | Few-shot explanation report in `explanation_prompt` |
| LangSmith Tracing (15%) | `@traceable` + `LANGCHAIN_TRACING_V2=true` |
| Code Quality (10%) | Modular, commented, typed |
| Bonus Features (5%) | Few-shot, JSON output, LangSmith tags |

---

## 👤 Author

**[Your Name]**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)
