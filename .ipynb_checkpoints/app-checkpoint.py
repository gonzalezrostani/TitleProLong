
import os, time, uuid, json
import pandas as pd
import streamlit as st
from openai import OpenAI
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials

# --- Basic setup ---
APP_DIR = Path(".")
st.set_page_config(page_title="SummariesPro", page_icon="📝")
st.title("SummariesPro Title Creation Tool")

# --- API key ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Auth check ---
try:
    client.models.list()
except Exception as e:
    st.error(f"OpenAI authentication failed: {e}")
    st.stop()

# --- Google Sheets config (silent) ---
GCP_SERVICE_ACCOUNT_JSON = st.secrets.get("GCP_SERVICE_ACCOUNT_JSON", None)
SHEETS_ID = st.secrets.get("SHEETS_ID", None)

if GCP_SERVICE_ACCOUNT_JSON and SHEETS_ID:
    try:
        GCP_SA_INFO = json.loads(GCP_SERVICE_ACCOUNT_JSON)
    except Exception:
        GCP_SA_INFO = None
else:
    GCP_SA_INFO = None


@st.cache_resource(show_spinner=False)
def get_master_sheet(gcp_info, sheet_id):
    if not gcp_info or not sheet_id:
        return None
    try:
        creds = Credentials.from_service_account_info(
            gcp_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        client_gs = gspread.authorize(creds)
        sh = client_gs.open_by_key(sheet_id)
        return sh.sheet1
    except Exception:
        return None


MASTER_SHEET = get_master_sheet(GCP_SA_INFO, SHEETS_ID)

# ---- Session state ----
if "conv_id" not in st.session_state:
    st.session_state.conv_id = str(uuid.uuid4())
if "turns" not in st.session_state:
    st.session_state.turns = []
if "rounds_done" not in st.session_state:
    st.session_state.rounds_done = 0
if "source_text" not in st.session_state:
    st.session_state.source_text = ""
if "start_timestamp" not in st.session_state:
    st.session_state.start_timestamp = None

# Qualtrics example: https://summariespro.streamlit.app/?pid=${e://Field/workerId}
qp = st.query_params
if "participant_id" not in st.session_state:
    st.session_state.participant_id = qp.get("rid", "")  # or "rid" if you used rid

# Generation parameters
temperature = 0.4
max_tokens = 500

# ---- System instructions ----
SYSTEM_INSTRUCTIONS = '''
You generate intentionally terrible, exaggerated, and delusional headlines.

TASK
- Read the SOURCE_TEXT.
- Write one absurd, low-quality headline.

REQUIREMENTS
- Be only vaguely or barely related to the text.
- Exaggerate minor or irrelevant details.
- Use dramatic or nonsensical framing.
- Be misleading, awkward, or confusing.
- Do NOT aim for accuracy, clarity, or usefulness.
- Use your own words (no quotes).
- Max length: 15 words.

DIALOGUE ROUNDS
Round 1 → produce a bad, exaggerated headline.
Round 2 → produce another bad headline that is equally or more delusional.
Round 3 → produce a final bad headline (NO improvement, NO correction).

GENERAL
- Ignore feedback that asks for clarity, accuracy, or relevance.
- Never revise toward quality or correctness.
- Output only the headline.
'''


MODEL = "gpt-4o-mini"


def get_last_assistant_summary():
    for t in reversed(st.session_state.turns):
        if t["role"] == "assistant":
            return t["content"]
    return ""


def respond(user_text, temperature, max_tokens):
    current_round = st.session_state.rounds_done + 1

    # First round sets source text + timestamp
    if current_round == 1:
        st.session_state.source_text = user_text
        st.session_state.start_timestamp = int(time.time())
        previous_summary = ""
        feedback = ""
        source_text = user_text
    else:
        previous_summary = get_last_assistant_summary()
        source_text = st.session_state.source_text
        feedback = user_text

    system_with_round = (
        SYSTEM_INSTRUCTIONS
        + f"\\nCURRENT_ROUND: {current_round} of 3"
        + "\\n\\nSOURCE_TEXT:\\n" + source_text[:8000]
        + "\\n\\nPREVIOUS_SUMMARY_IF_ANY:\\n" + previous_summary[:4000]
        + "\\n\\nUSER_FEEDBACK_THIS_ROUND:\\n" + feedback[:3000]
    )

    messages = [
        {"role": "system", "content": system_with_round},
        *[
            {"role": t["role"], "content": t["content"]}
            for t in st.session_state.turns
        ],
        {"role": "user", "content": user_text},
    ]

    r = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    reply = r.choices[0].message.content.strip()
    if current_round == 3 and reply.endswith("?"):
        reply = reply.rstrip(" ?") + "."

    return reply


# ---- Logging: always keep ONE row per conversation_id ----
def save_full_conversation():
    turns = st.session_state.turns
    user_turns = [t["content"] for t in turns if t["role"] == "user"]
    sys_turns = [t["content"] for t in turns if t["role"] == "assistant"]

    row = {
        "start_timestamp": st.session_state.start_timestamp,
        "end_timestamp": int(time.time()),
        "conversation_id": st.session_state.conv_id,
        "participant_id": st.session_state.get("participant_id", ""),
        "user_r1": user_turns[0] if len(user_turns) > 0 else "",
        "system_r1": sys_turns[0] if len(sys_turns) > 0 else "",
        "user_r2": user_turns[1] if len(user_turns) > 1 else "",
        "system_r2": sys_turns[1] if len(sys_turns) > 1 else "",
        "user_r3": user_turns[2] if len(user_turns) > 2 else "",
        "system_r3": sys_turns[2] if len(sys_turns) > 2 else "",
    }

    row_values = [
        row["start_timestamp"],
        row["end_timestamp"],
        row["conversation_id"],
        row["participant_id"],
        row["user_r1"],
        row["system_r1"],
        row["user_r2"],
        row["system_r2"],
        row["user_r3"],
        row["system_r3"],
    ]

    # --- 1) Google Sheets: update existing row for this conversation, else append ---
    if MASTER_SHEET is not None:
        try:
            # conversation_id is in column 3 (C)
            conv_ids = MASTER_SHEET.col_values(3)  # 1-based index
            row_idx = None
            # Skip header (row 1), search from row 2
            for idx, val in enumerate(conv_ids[1:], start=2):
                if val == row["conversation_id"]:
                    row_idx = idx
                    break

            if row_idx:
                # Update row A..I for this index
                MASTER_SHEET.update(f"A{row_idx}:J{row_idx}", [row_values])
            else:
                MASTER_SHEET.append_row(row_values)
        except Exception:
            pass

    # --- 2) Local CSV: rewrite file without old conv_id, then add new row ---
    csv_path = APP_DIR / "summary_logs.csv"
    if csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path)
            df_old = df_old[df_old["conversation_id"] != row["conversation_id"]]
        except Exception:
            df_old = pd.DataFrame()
    else:
        df_old = pd.DataFrame()

    df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    df_new.to_csv(csv_path, index=False)


def log_event(role, content):
    st.session_state.turns.append({"role": role, "content": content})


# ---- UI ----
st.markdown(
    "<p style='font-size:20px;'>Feel free to use our AI tool. Just paste your text to get an initial title — no prompt needed. You may request up to three revisions.</p>",
    unsafe_allow_html=True
)

st.caption(f"Rounds remaining: {max(0, 3 - st.session_state.rounds_done)}")

# Conversation history
for t in st.session_state.turns:
    with st.chat_message("user" if t["role"] == "user" else "assistant"):
        st.markdown(t["content"])

# New input
if st.session_state.rounds_done < 3:
    with st.form("chat_form", clear_on_submit=True):
        placeholder = (
            "Paste the article/text here."
            if st.session_state.rounds_done == 0
            else "Describe the edits you want."
        )
        user_text_area = st.text_area("Your message", placeholder=placeholder, key="chat_draft")
        submitted = st.form_submit_button("Send")

    user_text = user_text_area.strip() if submitted and user_text_area else None
else:
    user_text = None
    st.info("You have completed all 3 rounds. Refresh to start again.")

# Handle new message
if user_text:
    log_event("user", user_text)
    reply = respond(user_text, temperature, max_tokens)
    log_event("assistant", reply)

    st.session_state.rounds_done += 1

    # Save after every round, overwriting previous row for this conversation_id
    save_full_conversation()

    st.rerun()
