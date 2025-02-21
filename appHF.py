import streamlit as st
import random
import time
import json
import re
import requests
import os
from typing import List, Tuple, Set, Optional

# Field parameters
ROWS, COLS = 7, 7
NUM_OBSTACLES = 5

# Fixed emojis for other elements
GRASS = "🟩"
OBSTACLE = "🧱"
BALL_EMOJI = "⚽"
GOAL_EMOJI = "⬜"

# Retrieve your Hugging Face API token (set HF_API_TOKEN in your environment or Streamlit secrets)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "YOUR_HF_API_TOKEN_HERE")

def query_model(model_name: str, prompt: str) -> dict:
    """
    Uses the Hugging Face Inference API to get a text generation response.
    """
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}: {response.text}")
    return response.json()

def initialize_obstacles(rows: int, cols: int, num: int, forbidden: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Initializes obstacle positions on the field, avoiding forbidden positions."""
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
    Returns a string representing the field as a Markdown code block.
    The field is drawn with grass, goals, obstacles, and overlays the teams and the ball.
    """
    # Define goals: three centered cells on the left and right edges
    left_goal = {(rows // 2 - 1, 0), (rows // 2, 0), (rows // 2 + 1, 0)}
    right_goal = {(rows // 2 - 1, cols - 1), (rows // 2, cols - 1), (rows // 2 + 1, cols - 1)}
    
    # Initialize the grid with grass
    field = [[GRASS for _ in range(cols)] for _ in range(rows)]
    
    # Draw the goals
    for (r, c) in left_goal:
        field[r][c] = GOAL_EMOJI
    for (r, c) in right_goal:
        field[r][c] = GOAL_EMOJI

    # Place obstacles
    for (r, c) in obstacles:
        field[r][c] = OBSTACLE

    # Prepare the representation for the teams (adding the ball emoji if possessed)
    a_str = team_a_emoji + (BALL_EMOJI if ball_owner == "A" else "")
    b_str = team_b_emoji + (BALL_EMOJI if ball_owner == "B" else "")
    
    # Place the teams on the field (even if they overlap a goal)
    r, c = team_a
    field[r][c] = a_str
    r, c = team_b
    field[r][c] = b_str

    # If the ball is not possessed, draw it on its cell (if the cell is free)
    if ball_owner is None:
        r, c = ball
        if field[r][c] in [GRASS, OBSTACLE]:
            field[r][c] = BALL_EMOJI

    # Generate the field string with line breaks in a Markdown code block
    field_str = "```\n"
    for i in range(rows):
        field_str += " ".join(field[i]) + "\n"
    field_str += "```"
    return field_str

def get_valid_moves(pos: Tuple[int, int], obstacles: Set[Tuple[int, int]], rows: int, cols: int) -> List[str]:
    """
    Returns a list of valid moves for the agent without colliding with obstacles or leaving the field.
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
    Requests a move from the selected Hugging Face model using the Inference API,
    providing context about the current field state.
    """
    model_name = team_llm
    valid_moves = get_valid_moves(team_pos, obstacles, rows, cols)
    
    # Determine positions of both teams for full field display
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

You may only make one move per turn, and it must be one of the valid moves: {valid_moves}.

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
        # Use the Hugging Face Inference API to generate text
        result = query_model(model_name, prompt)
        # The API typically returns a list of outputs; we take the first one.
        generated_text = result[0]["generated_text"]
        st.write(f"🧠 Response from {model_name} for Team {team}:")
        st.write(generated_text)
    except Exception as e:
        st.write(f"Error generating text with model {model_name}: {e}")
        return random.choice(valid_moves)
    
    # Extract JSON from the generated text using regex
    json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
    if json_match:
        try:
            response_json = json.loads(json_match.group(0))
            move = response_json.get("move", "").upper()
            if move not in valid_moves:
                st.write("Invalid move received, choosing a random valid move.")
                move = random.choice(valid_moves)
        except json.JSONDecodeError:
            st.write("Error decoding JSON, choosing a random valid move.")
            move = random.choice(valid_moves)
    else:
        st.write("No JSON found in the response, choosing a random valid move.")
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
    # Validate boundaries and obstacles
    if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
        return pos
    if (new_r, new_c) in obstacles:
        return pos
    return (new_r, new_c)

def is_adjacent(pos: Tuple[int,int], target: Tuple[int,int]) -> bool:
    """
    Returns True if pos and target are in adjacent cells (including diagonals).
    Note: Equality is not considered; check for equality separately if needed.
    """
    return max(abs(pos[0] - target[0]), abs(pos[1] - target[1])) == 1

def run_game(team_a_emoji: str, team_b_emoji: str, team_a_llm: str, team_b_llm: str):
    """
    Runs the match simulation.
    Each team takes turns moving and ball possession is updated.
    A goal is scored when the team in possession moves the ball into the designated goal area cells:
      - Team A scores if the ball is moved into the right goal area.
      - Team B scores if the ball is moved into the left goal area.
    """
    # Initial positions
    team_a = (3, 1)
    team_b = (3, 5)
    ball = (3, 3)
    ball_owner: Optional[str] = None  # "A", "B", or None

    # Define goal areas (to avoid placing obstacles there)
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

        # Request moves from each team using the selected Hugging Face model via the API.
        move_a = get_team_move("A", team_a, team_b, obstacles, ROWS, COLS, ball, ball_owner,
                               team_a_llm, team_a_emoji, team_b_emoji)
        move_b = get_team_move("B", team_b, team_a, obstacles, ROWS, COLS, ball, ball_owner,
                               team_b_llm, team_a_emoji, team_b_emoji)
        st.write(f"{team_a_emoji} Team A ({team_a_llm}) moves: **{move_a}**")
        st.write(f"{team_b_emoji} Team B ({team_b_llm}) moves: **{move_b}**")

        # Update team positions
        team_a = update_position(team_a, move_a, obstacles, ROWS, COLS)
        team_b = update_position(team_b, move_b, obstacles, ROWS, COLS)

        # Update ball possession
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
            # The ball moves with the team that possesses it
            ball = team_a if ball_owner == "A" else team_b
            # Check if the opposing team is adjacent to steal the ball
            if ball_owner == "A" and is_adjacent(team_b, ball):
                ball_owner = "B"
                ball = team_b
                st.write(f"{team_b_emoji} Team B steals the ball from {team_a_emoji} Team A.")
            elif ball_owner == "B" and is_adjacent(team_a, ball):
                ball_owner = "A"
                ball = team_a
                st.write(f"{team_a_emoji} Team A steals the ball from {team_b_emoji} Team B.")

        # Check for a goal using the designated goal area cells
        if ball_owner == "A" and ball in right_goal:
            st.markdown(f"### GOAL! {team_a_emoji} Team A ({team_a_llm}) scores.")
            field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                                      team_a_emoji, team_b_emoji))
            break
        if ball_owner == "B" and ball in left_goal:
            st.markdown(f"### GOAL! {team_b_emoji} Team B ({team_b_llm}) scores.")
            field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                                      team_a_emoji, team_b_emoji))
            break

        field_placeholder.markdown(format_field(ROWS, COLS, obstacles, team_a, team_b, ball, ball_owner,
                                                  team_a_emoji, team_b_emoji))
        time.sleep(1)
    else:
        st.markdown("### End of match: Maximum turns reached.")

# Streamlit Interface
st.title("LLM Sports: LLM Match")
st.write("A turn-based competition between two teams controlled by LLMs (using Hugging Face's hosted Inference API).")

st.markdown("""
**Legend:**
- **Team A:** Must score by moving the ball into the right goal area (the white blocks).
- **Team B:** Must score by moving the ball into the left goal area (the white blocks).
""")

# Extended list of emojis related to LLM, technology, and animals
emoji_options = [
    "💎",  # gemma2
    "🌟",  # SuperNova
    "🔍",  # deepseek
    "🌬️",  # mistral
    "🦙",  # llama
    "🤖",  # arcee / general LLM
    "🚀",  # speed/performance
    "💡",  # ideas
    "🧠",  # intelligence
    "🪐",  # cosmic
    "🤗",  # Hugging Face
    "🐋",  # Whale
    "🐱",  # Cat
    "🐶",  # Dog
    "🐼",  # Panda
    "🦊"   # Fox
]

# List of available LLMs (model IDs on Hugging Face Hub)
llm_options = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma2:latest",
    "deepseek-r1:latest",
    "mistral:latest",
    "gemma2:9b",
    "gemma2:27b",
    "llama3.2:latest",
    "deepseek-r1:32b",
    "deepseek-r1:70b",
    "deepseek-r1:14b"
]

team_a_emoji = st.selectbox("Choose emoji for Team A", emoji_options, index=0)
team_b_emoji = st.selectbox("Choose emoji for Team B", emoji_options, index=3)

team_a_llm = st.selectbox("Choose LLM for Team A", llm_options, index=1)
team_b_llm = st.selectbox("Choose LLM for Team B", llm_options, index=3)

if st.button("Start Match"):
    run_game(team_a_emoji, team_b_emoji, team_a_llm, team_b_llm)
