import streamlit as st
import random
import time
import json
import re
import os
from typing import List, Tuple, Set, Optional

# Importa el cliente de OpenAI (usado para la API de NVIDIA)
from openai import OpenAI

# Parámetros del campo
ROWS, COLS = 7, 7
NUM_OBSTACLES = 5

# Emojis fijos para otros elementos
GRASS = "🟩"
OBSTACLE = "🧱"
BALL_EMOJI = "⚽"
GOAL_EMOJI = "⬜"

# API Key para NVIDIA API 
NV_API_KEY = "...."

# Inicializa el cliente de NVIDIA
nv_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NV_API_KEY
)

def query_model(model_name: str, prompt: str) -> list:
    """
    Utiliza la API de NVIDIA para obtener una respuesta de generación de texto en modo streaming.
    Retorna una lista con un diccionario que contiene la clave "generated_text".
    """
    messages = [{"role": "user", "content": prompt}]
    completion = nv_client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
        stream=True
    )
    generated_text = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            generated_text += chunk.choices[0].delta.content
    return [{"generated_text": generated_text}]

def initialize_obstacles(rows: int, cols: int, num: int, forbidden: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Inicializa las posiciones de los obstáculos en el campo, evitando posiciones prohibidas."""
    obstacles = set()
    while len(obstacles) < num:
        r = random.randint(0, rows - 1)
        c = random.randint(0, cols - 1)
        if (r, c) in forbidden:
            continue
        obstacles.add((r, c))
    return obstacles

def format_field(rows: int, cols: int, obstacles: Set[Tuple[int, int]],
                 team_a: Tuple[int, int], team_b: Tuple[int, int],
                 ball: Tuple[int, int], ball_owner: Optional[str],
                 team_a_emoji: str, team_b_emoji: str) -> str:
    """
    Retorna un string que representa el campo en un bloque de código Markdown.
    El campo se dibuja con césped, goles, obstáculos y se sobreponen los equipos y la pelota.
    """
    # Define los goles: tres celdas centradas en el borde izquierdo y derecho
    left_goal = {(rows // 2 - 1, 0), (rows // 2, 0), (rows // 2 + 1, 0)}
    right_goal = {(rows // 2 - 1, cols - 1), (rows // 2, cols - 1), (rows // 2 + 1, cols - 1)}
    
    # Inicializa la cuadrícula con césped
    field = [[GRASS for _ in range(cols)] for _ in range(rows)]
    
    # Dibuja los goles
    for (r, c) in left_goal:
        field[r][c] = GOAL_EMOJI
    for (r, c) in right_goal:
        field[r][c] = GOAL_EMOJI

    # Coloca los obstáculos
    for (r, c) in obstacles:
        field[r][c] = OBSTACLE

    # Prepara la representación de los equipos (agregando el emoji de la pelota si lo poseen)
    a_str = team_a_emoji + (BALL_EMOJI if ball_owner == "A" else "")
    b_str = team_b_emoji + (BALL_EMOJI if ball_owner == "B" else "")
    
    # Coloca los equipos en el campo (incluso si se superponen a un gol)
    r, c = team_a
    field[r][c] = a_str
    r, c = team_b
    field[r][c] = b_str

    # Si la pelota no está en posesión, dibujarla en su celda (si la celda está libre)
    if ball_owner is None:
        r, c = ball
        if field[r][c] in [GRASS, OBSTACLE]:
            field[r][c] = BALL_EMOJI

    # Genera el string del campo con saltos de línea en un bloque de código Markdown
    field_str = "```\n"
    for i in range(rows):
        field_str += " ".join(field[i]) + "\n"
    field_str += "```"
    return field_str

def get_valid_moves(pos: Tuple[int, int], obstacles: Set[Tuple[int, int]], rows: int, cols: int) -> List[str]:
    """
    Retorna una lista de movimientos válidos para el agente, evitando colisiones con obstáculos o salirse del campo.
    """
    valid_moves = []
    for move in ["U", "D", "L", "R"]:
        new_pos = update_position(pos, move, obstacles, rows, cols)
        if new_pos != pos:
            valid_moves.append(move)
    valid_moves.append("S")
    return valid_moves

def get_team_move(team: str, team_pos: Tuple[int, int], opponent_pos: Tuple[int, int], obstacles: Set[Tuple[int,int]],
                  rows: int, cols: int, ball: Tuple[int,int], ball_owner: Optional[str],
                  team_llm: str, team_a_emoji: str, team_b_emoji: str) -> str:
    """
    Solicita un movimiento desde el modelo LLM seleccionado a través de la API de NVIDIA,
    proporcionando contexto sobre el estado actual del campo.
    """
    model_name = team_llm
    valid_moves = get_valid_moves(team_pos, obstacles, rows, cols)
    
    # Determina las posiciones de ambos equipos para la visualización completa del campo
    team_a_pos = team_pos if team == "A" else opponent_pos
    team_b_pos = team_pos if team == "B" else opponent_pos

    prompt = f"""
You are controlling the agent of Team {team} in an LLM Sports match.
Here is the current state of the field:
- Valid moves (without collisions): {valid_moves}

The field looks like this:
{format_field(rows, cols, obstacles, team_a_pos, team_b_pos, ball, ball_owner, team_a_emoji, team_b_emoji)}

Your goal is to move the ball ⚽ into the opponent's goal area (the white blocks):
- For Team A {team_a_emoji}: score by moving the ball into the right goal area.
- For Team B {team_b_emoji}: score by moving the ball into the left goal area.

Your possible moves are:
- 'U' to move up
- 'D' to move down
- 'L' to move left
- 'R' to move right
- 'S' to stay in place

If {ball_owner} == True, then you own the ball at the moment. False means you don't. 

You may only make one move per turn, and it must be one of the valid moves: {valid_moves}.
If you have the ball ⚽ you should avoid your opponent. 
If your opponent has the ball, then you should go and collide with it to take the ball.
Remember, your ultimate aim is to move the ball ⚽ into the opponent's goal area (the white blocks)
Analyze the current state and choose the best move to progress toward your goal.
Respond in pure JSON format, without any additional text. For example:
{{
    "Team": "your team's emoji",
    "Target Goal": "Right/Left",
    "valid_moves": "{valid_moves}",
    "reasoning": "Explanation of your choice.",
    "move": "U"  // or "D", "L", "R", or "S"
}}
"""
    try:
        result = query_model(model_name, prompt)
        generated_text = result[0]["generated_text"]
        with st.expander("See model thought"):
            st.write(f"🧠 Response from {model_name} for Team {team}:")
            st.write(generated_text)
    except Exception as e:
        st.write(f"Error generating text with model {model_name}: {e}")
        return random.choice(valid_moves)
    
    # Extrae el movimiento utilizando re.finditer y obtiene la última coincidencia
    matches = list(re.finditer(r'"move"\s*:\s*"(U|D|L|R|S)"', generated_text))
    if matches:
        move = matches[-1].group(1).upper()
        if move not in valid_moves:
            st.write("🚨 Invalid move received, choosing a random valid move.")
            move = random.choice(valid_moves)
    else:
        st.write("🚨🚨 No valid move found in the response, choosing a random valid move.")
        move = random.choice(valid_moves)
    
    return move

def update_position(pos: Tuple[int,int], move: str, obstacles: Set[Tuple[int,int]],
                    rows: int, cols: int) -> Tuple[int,int]:
    r, c = pos
    new_r, new_c = r, c
    if move == "U":
        new_r = r - 1
    elif move == "D":
        new_r = r + 1
    elif move == "L":
        new_c = c - 1
    elif move == "R":
        new_c = c + 1
    elif move == "S":
        new_r, new_c = r, c
    # Valida límites y obstáculos
    if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
        return pos
    if (new_r, new_c) in obstacles:
        return pos
    return (new_r, new_c)

def is_adjacent(pos: Tuple[int,int], target: Tuple[int,int]) -> bool:
    """
    Retorna True si pos y target están en celdas adyacentes (incluyendo diagonales).
    Nota: La igualdad no se considera; verifica la igualdad por separado si es necesario.
    """
    return max(abs(pos[0] - target[0]), abs(pos[1] - target[1])) == 1

def run_game(team_a_emoji: str, team_b_emoji: str, team_a_llm: str, team_b_llm: str):
    """
    Ejecuta la simulación del partido.
    Cada equipo toma turnos para moverse y se actualiza la posesión de la pelota.
    Se marca gol cuando el equipo en posesión mueve la pelota a las celdas del área de gol designada:
      - Team A marca si la pelota se mueve al área de gol derecha.
      - Team B marca si la pelota se mueve al área de gol izquierda.
    """
    # Posiciones iniciales
    team_a = (3, 1)
    team_b = (3, 5)
    ball = (3, 3)
    ball_owner: Optional[str] = None  # "A", "B", o None

    # Define áreas de gol (para evitar colocar obstáculos allí)
    left_goal = {(ROWS // 2 - 1, 0), (ROWS // 2, 0), (ROWS // 2 + 1, 0)}
    right_goal = {(ROWS // 2 - 1, COLS - 1), (ROWS // 2, COLS - 1), (ROWS // 2 + 1, COLS - 1)}

    forbidden = {team_a, team_b, ball}
    forbidden.update(left_goal)
    forbidden.update(right_goal)

    obstacles = initialize_obstacles(ROWS, COLS, NUM_OBSTACLES, forbidden)

    max_turns = 20
    turn = 0

    field_placeholder = st.empty()
    st.markdown("## Initial State")
    field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                              team_a_emoji, team_b_emoji))



    while turn < max_turns:
        turn += 1
        st.markdown(f"### Turn {turn}")

        # Solicita movimientos a cada equipo usando la API de NVIDIA.
        move_a = get_team_move("A", team_a, team_b, obstacles, ROWS, COLS, ball, ball_owner,
                               team_a_llm, team_a_emoji, team_b_emoji)
        move_b = get_team_move("B", team_b, team_a, obstacles, ROWS, COLS, ball, ball_owner,
                               team_b_llm, team_a_emoji, team_b_emoji)
        st.write(f"{team_a_emoji} Team A ({team_a_llm}) moves: **{move_a}**")
        st.write(f"{team_b_emoji} Team B ({team_b_llm}) moves: **{move_b}**")

        # Actualiza las posiciones de los equipos
        team_a = update_position(team_a, move_a, obstacles, ROWS, COLS)
        team_b = update_position(team_b, move_b, obstacles, ROWS, COLS)

        # Actualiza la posesión de la pelota
        if ball_owner is None:
            a_adj = is_adjacent(team_a, ball) or team_a == ball
            b_adj = is_adjacent(team_b, ball) or team_b == ball
            if a_adj and not b_adj:
                ball_owner = "A"
                ball = team_a
                st.write(f"{team_a_emoji} Team A takes possession of the ball.")
            elif b_adj and not a_adj:
                ball_owner = "B"
                ball = team_b
                st.write(f"{team_b_emoji} Team B takes possession of the ball.")
            elif a_adj and b_adj:
                ball_owner = random.choice(["A", "B"])
                ball = team_a if ball_owner == "A" else team_b
                st.write(f"Both teams contest the ball. It is assigned to {team_a_emoji if ball_owner=='A' else team_b_emoji}.")
        else:
            # La pelota se mueve con el equipo que la posee
            ball = team_a if ball_owner == "A" else team_b
            # Verifica si el equipo contrario está adyacente para robar la pelota
            if ball_owner == "A" and is_adjacent(team_b, ball):
                ball_owner = "B"
                ball = team_b
                st.write(f"{team_b_emoji} Team B steals the ball from {team_a_emoji} Team A.")
            elif ball_owner == "B" and is_adjacent(team_a, ball):
                ball_owner = "A"
                ball = team_a
                st.write(f"{team_a_emoji} Team A steals the ball from {team_b_emoji} Team B.")

        # Verifica si se marca un gol usando las celdas designadas para el área de gol
        if ball_owner == "A" and ball in right_goal:
            st.markdown(f"### GOAL! {team_a_emoji} Team A ({team_a_llm}) scores.")
            field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                                      team_a_emoji, team_b_emoji))
            st.balloons()
            break
        if ball_owner == "B" and ball in left_goal:
            st.markdown(f"### GOAL! {team_b_emoji} Team B ({team_b_llm}) scores.")
            field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                                      team_a_emoji, team_b_emoji))
            st.balloons()
            break

        field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                                  team_a_emoji, team_b_emoji))

        time.sleep(1)
    else:
        st.markdown("### End of match: Maximum turns reached.")

# Interfaz de Streamlit
st.title("LLM Sports 🦙⚽🥅")
st.write("A turn-based competition between two teams controlled by LLMs (using NVIDIA's hosted Inference API).")

st.markdown("""
**Legend:**
- **Team A:** Must score by moving the ball into the right goal area 👉.
- **Team B:** Must score by moving the ball into the left goal area 👈.
""")

# Lista extendida de emojis relacionados con LLM, tecnología y animales
emoji_options = [
    "🦙",  # llama
    "💎",  # gemma2
    "🌬️",  # mistral
    "🤗",  # Hugging Face
    "🐋",  # Whale
    "🌟",  # SuperNova
    "🔍",  # deepseek
    "🤖",  # arcee / general LLM
    "🚀",  # speed/performance
    "💡",  # ideas
    "🧠",  # intelligence
    "🪐",  # cosmic
    "🐱",  # Cat
    "🐶",  # Dog
    "🐼",  # Panda
    "🦊"   # Fox
]

# Lista de LLMs disponibles (model IDs en NVIDIA Hub)
llm_options = [
    "meta/llama-3.3-70b-instruct",
    "mistralai/mistral-7b-instruct-v0.3",
    "mistralai/mistral-small-24b-instruct",
    "deepseek-ai/deepseek-r1",
    "google/gemma-2-27b-it",
    "google/gemma-2-2b-it"
]




team_a_llm = st.sidebar.selectbox("Choose LLM for Team A", llm_options, index=1)
team_a_emoji = st.sidebar.selectbox("Choose emoji for Team A", emoji_options, index=0)
team_b_llm = st.sidebar.selectbox("Choose LLM for Team B", llm_options, index=3)
team_b_emoji = st.sidebar.selectbox("Choose emoji for Team B", emoji_options, index=3)

if st.button("Start Match"):
    run_game(team_a_emoji, team_b_emoji, team_a_llm, team_b_llm)
