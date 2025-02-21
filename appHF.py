import streamlit as st
import random
import time
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Set, Optional

# Field parameters
ROWS, COLS = 7, 7
NUM_OBSTACLES = 5

# Fixed emojis for other elements
GRASS = "🟩"
OBSTACLE = "🧱"
BALL_EMOJI = "⚽"
GOAL_EMOJI = "⬜"

# Load model and tokenizer (this is a heavy model; ensure you have sufficient resources)
@st.cache_resource
def load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Set your model name (this example uses Qwen/Qwen2.5-72B-Instruct)
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

def initialize_obstacles(rows: int, cols: int, num: int, forbidden: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
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
    # Define goal areas: left and right (white blocks)
    left_goal = {(rows // 2 - 1, 0), (rows // 2, 0), (rows // 2 + 1, 0)}
    right_goal = {(rows // 2 - 1, cols - 1), (rows // 2, cols - 1), (rows // 2 + 1, cols - 1)}
    
    # Create grid of grass
    field = [[GRASS for _ in range(cols)] for _ in range(rows)]
    
    # Draw goal cells
    for (r, c) in left_goal:
        field[r][c] = GOAL_EMOJI
    for (r, c) in right_goal:
        field[r][c] = GOAL_EMOJI

    # Place obstacles
    for (r, c) in obstacles:
        field[r][c] = OBSTACLE

    # Place teams (and append ball emoji if they have possession)
    a_str = team_a_emoji + (BALL_EMOJI if ball_owner == "A" else "")
    b_str = team_b_emoji + (BALL_EMOJI if ball_owner == "B" else "")
    r, c = team_a
    field[r][c] = a_str
    r, c = team_b
    field[r][c] = b_str

    # If the ball is free, place it if the cell is available
    if ball_owner is None:
        r, c = ball
        if field[r][c] in [GRASS, OBSTACLE]:
            field[r][c] = BALL_EMOJI

    field_str = "```\n"
    for row in field:
        field_str += " ".join(row) + "\n"
    field_str += "```"
    return field_str

def get_valid_moves(pos: Tuple[int, int], obstacles: Set[Tuple[int, int]], rows: int, cols: int) -> List[str]:
    valid_moves = []
    for move in ["U", "D", "L", "R"]:
        new_pos = update_position(pos, move, obstacles, rows, cols)
        if new_pos != pos:
            valid_moves.append(move)
    valid_moves.append("S")
    return valid_moves

def get_team_move_local(team: str, team_pos: Tuple[int, int], opponent_pos: Tuple[int, int],
                        obstacles: Set[Tuple[int, int]], rows: int, cols: int, ball: Tuple[int, int],
                        ball_owner: Optional[str], team_a_emoji: str, team_b_emoji: str) -> str:
    valid_moves = get_valid_moves(team_pos, obstacles, rows, cols)
    
    prompt = f"""
You are controlling the agent of Team {team} in an LLM Sports match.
Valid moves: {valid_moves}

Current field:
{format_field(rows, cols, obstacles, team_pos if team=="A" else opponent_pos,
             team_pos if team=="B" else opponent_pos, ball, ball_owner, team_a_emoji, team_b_emoji)}

Your goal is to move the ball into the opponent's goal area (the white blocks).
You may only choose one move from {valid_moves}.
Respond in pure JSON format. For example:
{{
    "Team": "your team's emoji",
    "Target Goal": "Right/Left",
    "valid_moves": "{valid_moves}",
    "reasoning": "Explanation of your choice.",
    "move": "U"
}}
"""
    # Prepare a chat-style prompt using a system and user message.
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # Use the tokenizer's chat template function to build a prompt.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # Generate a response; adjust max_new_tokens as needed.
    generated_ids = model.generate(**model_inputs, max_new_tokens=256)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    st.write(f"🧠 Response from local model for Team {team}:")
    st.write(generated_text)
    
    # Extract JSON from the generated text
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

def update_position(pos: Tuple[int, int], move: str, obstacles: Set[Tuple[int, int]],
                    rows: int, cols: int) -> Tuple[int, int]:
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
    if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
        return pos
    if (new_r, new_c) in obstacles:
        return pos
    return (new_r, new_c)

def is_adjacent(pos: Tuple[int, int], target: Tuple[int, int]) -> bool:
    return max(abs(pos[0] - target[0]), abs(pos[1] - target[1])) == 1

def run_game(team_a_emoji: str, team_b_emoji: str, team_a_llm: str, team_b_llm: str):
    # Initial positions
    team_a = (3, 1)
    team_b = (3, 5)
    ball = (3, 3)
    ball_owner: Optional[str] = None
    
    # Define goal areas (white blocks)
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
        
        move_a = get_team_move_local("A", team_a, team_b, obstacles, ROWS, COLS, ball, ball_owner,
                                     team_a_emoji, team_b_emoji)
        move_b = get_team_move_local("B", team_b, team_a, obstacles, ROWS, COLS, ball, ball_owner,
                                     team_a_emoji, team_b_emoji)
        st.write(f"{team_a_emoji} Team A ({team_a_llm}) moves: **{move_a}**")
        st.write(f"{team_b_emoji} Team B ({team_b_llm}) moves: **{move_b}**")
        
        team_a = update_position(team_a, move_a, obstacles, ROWS, COLS)
        team_b = update_position(team_b, move_b, obstacles, ROWS, COLS)
        
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
            ball = team_a if ball_owner == "A" else team_b
            if ball_owner == "A" and is_adjacent(team_b, ball):
                ball_owner = "B"
                ball = team_b
                st.write(f"{team_b_emoji} Team B steals the ball from {team_a_emoji} Team A.")
            elif ball_owner == "B" and is_adjacent(team_a, ball):
                ball_owner = "A"
                ball = team_a
                st.write(f"{team_a_emoji} Team A steals the ball from {team_b_emoji} Team B.")
        
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
st.title("LLM Sports: LLM Match (Local Inference)")
st.write("A turn-based competition between two teams controlled by a locally loaded Hugging Face model.")

st.markdown("""
**Legend:**
- **Team A:** Must score by moving the ball into the right goal area (the white blocks).
- **Team B:** Must score by moving the ball into the left goal area (the white blocks).
""")

# Extended list of emojis related to LLM, technology, and animals
emoji_options = ["💎", "🌟", "🔍", "🌬️", "🦙", "🤖", "🚀", "💡", "🧠", "🪐", "🤗", "🐋", "🐱", "🐶", "🐼", "🦊"]

# For local inference in this example, we are using our one loaded model.
llm_options = [MODEL_NAME]

team_a_emoji = st.selectbox("Choose emoji for Team A", emoji_options, index=0)
team_b_emoji = st.selectbox("Choose emoji for Team B", emoji_options, index=3)
team_a_llm = st.selectbox("Choose LLM for Team A", llm_options, index=0)
team_b_llm = st.selectbox("Choose LLM for Team B", llm_options, index=0)

if st.button("Start Match"):
    run_game(team_a_emoji, team_b_emoji, team_a_llm, team_b_llm)
