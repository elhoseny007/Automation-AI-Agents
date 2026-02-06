# 🤖 AI-Powered Resume ATS Analyzer (n8n Agent)

An autonomous AI agent built with **n8n** that streamlines the recruitment lifecycle. This system automates the ingestion, parsing, and scoring of resumes against specific Job Descriptions (JDs) using LLMs, delivering instant feedback to hiring teams.

## 🌟 Overview
Manual CV screening is a bottleneck in modern hiring. This project demonstrates an **End-to-End Automation Pipeline** that replaces hours of manual work with a real-time AI evaluation engine.

### ⚡ The Result:
* **80% Reduction** in initial screening time.
* **Objective Scoring** based on data-driven LLM analysis.
* **Instant Notifications** for HR via Slack and Email.

---

## 🏗️ Workflow Architecture
The agent follows a sophisticated logic path:
1.  **Trigger:** Monitors a **Google Drive** folder for new `.pdf` or `.docx` uploads.
2.  **Extraction:** Converts binary files to raw text for processing.
3.  **Intelligence:** Sends the resume and JD to **OpenAI (GPT models)** via API to identify skill gaps, years of experience, and cultural fit.
4.  **Logic:** Uses **n8n Function Nodes** (JavaScript) to parse the AI response and calculate a final match score.
5.  **Output:** * If score > 80%: Sends a "Priority Candidate" alert to **Slack**.
    * All results: Sends a detailed summary report via **Gmail**.

---

## 🛠️ Tech Stack
* **Workflow Engine:** [n8n.io](https://n8n.io/) (Self-hosted/Cloud)
* **AI Layer:** OpenAI API (GPT-4 / GPT-3.5-turbo)
* **Storage:** Google Drive API
* **Communication:** Slack Webhooks & SMTP (Gmail)
* **Processing:** JavaScript / JSON

---

## 🚀 Getting Started

### Prerequisites
* An n8n instance (Desktop, Docker, or Cloud).
* OpenAI API Key.
* Google Cloud Console Credentials (for Drive access).

### Installation
1.  **Clone this repo:**
    ```bash
    git clone [https://github.com/elhoseny007/Automation-AI-Agents.git](https://github.com/elhoseny007/Automation-AI-Agents.git)
    ```
2.  **Import Workflow:**
    * Open n8n.
    * Click on **Workflows** > **Import from File**.
    * Select `ats_analyzer_workflow.json` from this repository.
3.  **Setup Credentials:**
    * Configure your OpenAI, Google Drive, and Slack credentials in the n8n credential manager.
4.  **Activate:** Toggle the "Active" switch to start the automation.

---

## 📊 Logic & Decision Making
This project implements custom **Conditional Logic**:
* **High Match:** Triggers an immediate Slack notification to the recruiter.
* **Medium Match:** Logs the data into a spreadsheet for later review.
* **Low Match:** Sends an automated, polite rejection or "keep in file" email to the candidate.

---

## 👨‍💻 Author
**Elhoseny Hassan** *Junior AI Automation & Agent Engineer* [LinkedIn](https://linkedin.com/in/elhoseny-hassan-65561b191) | [GitHub](https://github.com/elhoseny007)
