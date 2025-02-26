# LLM Sports


[LLM]("https://x.com/napoles3D/status/1893860366540591311?t=c_k2NGDvACI8AN9JyHZPug&s=19") Sports is a turn-based simulation game where two teams—each controlled by a selected language model (LLM)—compete in a soccer-inspired match on a grid-based field. The field features dynamic obstacles and designated goal areas, and teams are represented by customizable emojis. 

<img src="https://raw.githubusercontent.com/napoles-uach/LLMSports/refs/heads/main/demo.png" alt="Demo Image" width="600" />
Moves are determined by the LLMs based on the current state of the field, with the aim of scoring a goal, avoiding obstacles and the opponent.
<p align="center">
  <img src="https://raw.githubusercontent.com/napoles-uach/LLMSports/refs/heads/main/demo.gif" alt="Demo GIF" width="600" />
</p>


....
## Features

- **Turn-Based Gameplay:** Teams take alternating turns to move.
- **Customizable Teams:** Choose from a variety of emojis to represent each team.
- **Selectable LLMs:** Assign different language models (e.g., Gemma2, Mistral, Deepseek, etc.) to control each team.
- **Dynamic Field:** Play on a grid-based field with obstacles and goal areas defined by white blocks.
- **Interactive Interface:** Built with Streamlit for real-time game updates.
- **Goal Detection:** A goal is scored when the ball moves into one of the designated goal cells.

## Requirements

- Python 3.7 or higher
- [Streamlit](https://streamlit.io/)
- [ollama](https://ollama.ai/) (or your configured LLM API client)
- Other standard libraries: `random`, `time`, `json`, `re`, `typing`

## Installation

   ```bash
   git clone https://github.com/napoles-uach/LLMSports.git
   cd LLMSports
   pip install streamlit ollama
(install locally models like mistral, llama3.2, Gemma2, etc.)
   streamlit run app.py


