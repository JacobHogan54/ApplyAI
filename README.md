# ApplyAI

ApplyAI is a local AI-powered job matching tool built with Python, Streamlit, Ollama, and Qwen.

It compares a job description against an uploaded resume, separates hard qualifications from broader competencies and preferences, and produces a structured match analysis without requiring a paid AI API.

## Features

- Upload a resume or reuse the last uploaded resume
- Local PDF text extraction
- Local candidate-profile generation
- Job-description parsing
- Qualification matching with local Qwen models through Ollama
- Deterministic scoring
- Direct / Partial / Missing qualification labels
- Evidence-based match explanations
- Candidate eligibility settings such as work authorization and sponsorship
- Job preferences such as remote / hybrid / onsite, relocation, travel, and job type
- Separate Qualification Match, Eligibility, and Preferences sections
- Local persistence for private candidate data
- Regression tests for matching behavior

## Privacy

ApplyAI is designed to keep personal data local.

Files such as uploaded resumes, generated candidate profiles, and local candidate settings should not be committed to Git.

The project ignores private local data such as:

- `.applyai/`
- uploaded PDF resumes
- `candidate_profile.json`
- local settings
- `.env`
- virtual environments

Before publishing changes, always run:

```bash
git status
```

and verify that no personal files are being tracked.

## Requirements

- Python 3.11+ recommended
- Ollama installed locally
- Qwen model installed through Ollama

Install the model with:

```bash
ollama pull qwen3:1.7b
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd ApplyAI
```

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the Ollama model is available:

```bash
ollama pull qwen3:1.7b
```

Run the app:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How It Works

ApplyAI uses a hybrid local AI architecture:

1. A resume is uploaded and parsed locally.
2. Resume information is converted into a compact candidate profile.
3. The job description is parsed into:
   - Core requirements
   - Required competencies
   - Preferred qualifications
   - Responsibilities
4. Qwen compares the candidate profile against job requirements.
5. Deterministic Python validation checks the model output and prevents unsupported matches.
6. Python calculates the qualification score.
7. Candidate-supplied eligibility and preference data are evaluated separately.

This prevents information such as work authorization from being incorrectly inferred from resume text.

## Scoring

Qualification scoring is based on resume-supported evidence.

Match states include:

- `direct` — strong evidence directly supports the requirement
- `partial` — related or transferable evidence exists, but the requirement is not fully supported
- `missing` — no supported evidence is present

Eligibility and job preferences are kept separate from the qualification score.

## Candidate Profile

ApplyAI stores private candidate information locally.

A sample public profile is included:

```text
candidate_profile.example.json
```

The real candidate profile should remain excluded from Git.

## Testing

Run the test suite with:

```bash
pytest
```

## Tech Stack

- Python
- Streamlit
- Ollama
- Qwen
- Pydantic
- pypdf
- pytest

## Project Status

ApplyAI is currently an early local-first version focused on:

- Resume-to-job matching
- Evidence grounding
- Deterministic scoring
- Candidate eligibility
- Job preference matching
- Local privacy

Potential future improvements include:

- Job history
- Saved analyses
- Side-by-side job comparison
- Tailored resume suggestions
- Exportable reports
- Expanded regression testing

## License

No license has been selected yet.
