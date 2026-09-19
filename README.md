---
title: Final Assignment
emoji: ???????
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
license: mit
---
<br />
<div align="center">
  <a href="https://huggingface.co/learn/agents-course/unit4/introduction">
    
  </a>

  <h3 align="center">Agents Course Final Project</h3>

  <p align="center" style="width:80%">
    Final hands-on assignment for the Hugging Face Agents course. In this project I built a multi-agent solution, evaluated it against questions from the General AI Assistants (GAIA) benchmark (level one only), and got creative with some agent and tool improvements.
</div>

## About The Project

Achieving 30 points for the certification was relatively easy with the template provided and a powerful enough LLM. However, evaluation revealed the unique types of implementation challenges with AI agents. Some of which include...

* Cost
* Reliability
* Response Times

Beyond what looks like a smolagents guided tour, you can find the following in this repo...

* Research agent armed with Google search via Serper and both Audio and Video Understanding via Gemini
* Chess agent leveraging a board_to_fen fork and a Stockfish API.
* Langfuse setup boilerplate, a working example.
* Pydantic settings for type safety, centralized, and encapsulated config.
* Basic parallel agent task execution, compatible with smolagents and the Gradio UI.

## Getting Started

### Prerequisites

* python
* pip
* git

### Installation

1. Get an API key for OpenRouter, Gemini, Serper, and Langfuse.
2. Clone the repo and install packages.
3. Enter your API keys in a .env file.
4. Run the app with: python app.py
