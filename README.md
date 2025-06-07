# szia-ai-tools

**Szia AI Tools** is a lightweight Python toolkit designed for building basic Generative AI applications with minimal setup. It includes utility components for document parsing, content chunking, prompt engineering, and interaction with LLM APIs such as OpenAI. For notebook and app tutorials, visit the [examples](./examples/) folder.

This toolkit is ideal for:
- Hobby projects
- Educational demos
- Small to mid-scale GenAI apps where simplicity and clarity matter

If you're looking for **enterprise-grade, scalable solutions**, including advanced RAG graphs, multi-LLM orchestration, and graph-based reasoning, please visit the [LynxScribe platform](https://www.lynxanalytics.com/generative-ai-platform) by Lynx Analytics.

---

## 🛠 Installing the Environment

To install the environment, the easiest option is to use [uv](https://github.com/astral-sh/uv). You can install `uv` by following the instructions in the linked README:
- **Linux/macOS**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **macOS (alternative)**: `brew install uv`
- **Windows**:
  - **PowerShell**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - However, the most recommended approach is to [set up a WSL (Windows Subsystem for Linux) environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment), as working directly in Windows can be inconvenient for this workflow.

> ⚠️ Make sure Python 3.12 (or newer) installed before proceeding (use `python --version` to check your current version).
> If it’s not installed, [download it here](https://www.python.org/downloads).

Once `uv` and Python are available, follow these steps to set up your environment and run the demos:

```bash
# Step 0: Clone the library
git clone https://github.com/mszel/szia-ai-tools.git # or git clone git@github.com:mszel/szia-ai-tools.git
cd szia-ai-tools

# Step 1: Create a new uv environment
# You can specify any compatible Python version here (e.g., 3.12 or 3.13)
uv venv --python=$(which python3.12) .venv

# Step 2: Activate the virtual environment
source .venv/bin/activate

# Step 3: Install the dependencies (including development extras)
uv pip install -e ".[dev,ingest]"
```

Next, create a `.env` file by copying the provided template (`.env.example`) and filling in the required secrets, such as:
- OPENAI_API_KEY=...

You can generate your OpenAI API key at: https://platform.openai.com/api-keys
(Note: You may need to sign up or log in first.)

### 🧪 Using Locally-Hosted Models

If you'd like to experiment with locally hosted models in your development environment, you can use [Ollama](https://ollama.com):

- **Install Ollama**: https://ollama.com/download
- **Run the Ollama service**:
  - On **Windows** or **macOS**: launch the Ollama app
  - On **Linux** or any system: run `ollama serve` in a terminal
- In a **new terminal**, start a model (e.g., Qwen3 8B):
  `ollama run qwen3:8b`
  You can use any model from the [Ollama model library](https://ollama.com/search) — including modern options like Qwen3, not just Gemma or LLaMA.

Configuration examples for locally hosted models are included throughout the project.

---

## 🔐 License

This project is licensed under the **Apache License 2.0**.
See the full license in the [LICENSE](./LICENSE) file.

---

## ⭐️ Support

If you find this project helpful, please give it a star.

---
