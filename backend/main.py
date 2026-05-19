from fastapi import FastAPI, Header
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
import os
import json
import uuid
import psycopg2
from psycopg2.extras import Json
from psycopg2.extras import RealDictCursor

# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Create FastAPI app
app = FastAPI()

# Create Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

class ChatRequest(BaseModel):
    message: str


# PostgreSQL connection helper
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "ai_ordering"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD"),
        cursor_factory=RealDictCursor
    )


# --- Conversation DB helpers ---

def get_system_message():
    return {
        "role": "system",
        "content": """
        You are an AI telecom ordering assistant.

        Help customers:
        - search telecom plans
        - compare pricing
        - calculate quotes

        Use tools whenever needed.
        Keep responses concise and helpful.
        """
    }


def ensure_conversation(session_id: str):
    query = """
        INSERT INTO conversations (session_id)
        VALUES (%s)
        ON CONFLICT (session_id) DO NOTHING;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (session_id,))
            conn.commit()


def save_message(session_id: str, role: str, content: str | None, tool_call_id: str | None = None, tool_calls=None):
    query = """
        INSERT INTO conversation_messages (session_id, role, content, tool_call_id, tool_calls)
        VALUES (%s, %s, %s, %s, %s);
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    session_id,
                    role,
                    content,
                    tool_call_id,
                    Json(tool_calls) if tool_calls is not None else None
                )
            )
            conn.commit()


def load_messages(session_id: str):
    query = """
        SELECT role, content, tool_call_id, tool_calls
        FROM conversation_messages
        WHERE session_id = %s
        ORDER BY id ASC;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (session_id,))
            rows = cursor.fetchall()

    messages = [get_system_message()]

    for row in rows:
        if row["role"] == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": row["tool_call_id"],
                    "content": row["content"]
                }
            )
        elif row["role"] == "assistant" and row["tool_calls"] is not None:
            messages.append(
                {
                    "role": "assistant",
                    "content": row["content"],
                    "tool_calls": row["tool_calls"]
                }
            )
        else:
            messages.append(
                {
                    "role": row["role"],
                    "content": row["content"]
                }
            )

    return messages


def serialize_tool_calls(tool_calls):
    if not tool_calls:
        return None

    return [
        {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments
            }
        }
        for tool_call in tool_calls
    ]

# Tool 1
def search_offers(max_price: int | None = None, lob: str | None = None):
    query = """
        SELECT id, name, price, lob, speed
        FROM offers
        WHERE (%s IS NULL OR price <= %s)
        AND (%s IS NULL OR lob = %s)
        ORDER BY price ASC;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (max_price, max_price, lob, lob))
            return cursor.fetchall()

# Tool 2
def calculate_quote(offer_ids: list[int]):
    if not offer_ids:
        return {
            "selected_offers": [],
            "monthly_total": 0
        }

    query = """
        SELECT id, name, price, lob, speed
        FROM offers
        WHERE id = ANY(%s)
        ORDER BY price ASC;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(query, (offer_ids,))
            selected = cursor.fetchall()

    total = sum(offer["price"] for offer in selected)

    return {
        "selected_offers": selected,
        "monthly_total": total
    }

# AI Tool Definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_offers",
            "description": "Search available telecom offers by max price and line of business.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_price": {
                        "type": "integer",
                        "description": "Maximum monthly price"
                    },
                    "lob": {
                        "type": "string",
                        "enum": [
                            "internet",
                            "mobile",
                            "bundle"
                        ],
                        "description": "Line of business"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_quote",
            "description": "Calculate quote total for selected offers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "offer_ids": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        }
                    }
                },
                "required": ["offer_ids"]
            }
        }
    }
]

# Map AI tool names to Python functions
available_tools = {
    "search_offers": search_offers,
    "calculate_quote": calculate_quote
}

# Health endpoint
@app.get("/")
def root():
    return {
        "message": "AI Ordering Agent Running"
    }


# Simple HTML UI endpoint
@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AI Ordering Agent</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f5f5f5;
                margin: 0;
                padding: 0;
            }

            .container {
                max-width: 800px;
                margin: 40px auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                padding: 24px;
            }

            h1 {
                margin-top: 0;
                font-size: 24px;
            }

            .session {
                font-size: 12px;
                color: #666;
                margin-bottom: 16px;
                word-break: break-all;
            }

            .chat-box {
                border: 1px solid #ddd;
                border-radius: 8px;
                height: 400px;
                overflow-y: auto;
                padding: 16px;
                background: #fafafa;
                margin-bottom: 16px;
            }

            .message {
                margin-bottom: 12px;
                padding: 10px 12px;
                border-radius: 8px;
                max-width: 80%;
                white-space: pre-wrap;
            }

            .user {
                background: #dff1ff;
                margin-left: auto;
            }

            .assistant {
                background: #eeeeee;
                margin-right: auto;
            }

            .input-row {
                display: flex;
                gap: 8px;
            }

            input {
                flex: 1;
                padding: 12px;
                border: 1px solid #ccc;
                border-radius: 8px;
                font-size: 14px;
            }

            button {
                padding: 12px 18px;
                border: none;
                border-radius: 8px;
                background: #111827;
                color: white;
                cursor: pointer;
                font-size: 14px;
            }

            button:disabled {
                opacity: 0.6;
                cursor: not-allowed;
            }

            .secondary-button {
                margin-top: 12px;
                background: #6b7280;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Ordering Agent</h1>
            <div class="session" id="sessionInfo">Session: none yet</div>

            <div class="chat-box" id="chatBox"></div>

            <div class="input-row">
                <input id="messageInput" placeholder="Ask something like: Find internet plans under 80 dollars" />
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>

            <button class="secondary-button" onclick="resetSession()">Start New Session</button>
        </div>

        <script>
            const chatBox = document.getElementById("chatBox");
            const messageInput = document.getElementById("messageInput");
            const sendButton = document.getElementById("sendButton");
            const sessionInfo = document.getElementById("sessionInfo");

            function getSessionId() {
                return localStorage.getItem("ai_ordering_session_id");
            }

            function setSessionId(sessionId) {
                localStorage.setItem("ai_ordering_session_id", sessionId);
                sessionInfo.innerText = `Session: ${sessionId}`;
            }

            function loadSessionLabel() {
                const sessionId = getSessionId();
                sessionInfo.innerText = sessionId ? `Session: ${sessionId}` : "Session: none yet";
            }

            function addMessage(role, content) {
                const messageDiv = document.createElement("div");
                messageDiv.className = `message ${role}`;
                messageDiv.innerText = content;
                chatBox.appendChild(messageDiv);
                chatBox.scrollTop = chatBox.scrollHeight;
            }

            async function sendMessage() {
                const message = messageInput.value.trim();

                if (!message) {
                    return;
                }

                addMessage("user", message);
                messageInput.value = "";
                sendButton.disabled = true;
                sendButton.innerText = "Sending...";

                try {
                    const headers = {
                        "Content-Type": "application/json"
                    };

                    const sessionId = getSessionId();
                    if (sessionId) {
                        headers["X-Session-Id"] = sessionId;
                    }

                    const response = await fetch("/chat", {
                        method: "POST",
                        headers,
                        body: JSON.stringify({ message })
                    });

                    const data = await response.json();

                    if (data.session_id) {
                        setSessionId(data.session_id);
                    }

                    addMessage("assistant", data.response || "No response returned.");
                } catch (error) {
                    addMessage("assistant", `Error: ${error.message}`);
                } finally {
                    sendButton.disabled = false;
                    sendButton.innerText = "Send";
                    messageInput.focus();
                }
            }

            function resetSession() {
                localStorage.removeItem("ai_ordering_session_id");
                chatBox.innerHTML = "";
                loadSessionLabel();
            }

            messageInput.addEventListener("keydown", function(event) {
                if (event.key === "Enter") {
                    sendMessage();
                }
            });

            loadSessionLabel();
        </script>
    </body>
    </html>
    """


# Main AI Chat Endpoint
@app.post("/chat")
def chat(
    request: ChatRequest,
    x_session_id: str | None = Header(default=None, alias="X-Session-Id")
):

    session_id = x_session_id or str(uuid.uuid4())

    ensure_conversation(session_id)

    save_message(
        session_id=session_id,
        role="user",
        content=request.message
    )

    messages = load_messages(session_id)

    # FIRST AI CALL
    # AI decides whether tools are needed
    first_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    assistant_message = first_response.choices[0].message

    # If AI responds directly without tools
    if not assistant_message.tool_calls:
        save_message(
            session_id=session_id,
            role="assistant",
            content=assistant_message.content
        )

        return {
            "session_id": session_id,
            "response": assistant_message.content
        }

    serialized_tool_calls = serialize_tool_calls(assistant_message.tool_calls)

    save_message(
        session_id=session_id,
        role="assistant",
        content=assistant_message.content,
        tool_calls=serialized_tool_calls
    )

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": serialized_tool_calls
        }
    )

    # Execute tool calls
    for tool_call in assistant_message.tool_calls:

        function_name = tool_call.function.name

        function_args = json.loads(
            tool_call.function.arguments
        )

        print("SESSION ID:", session_id)
        print("TOOL CALLED:", function_name)
        print("ARGS:", function_args)

        # Find actual Python function
        tool_function = available_tools[function_name]

        # Execute backend function
        tool_result = tool_function(**function_args)

        print("TOOL RESULT:", tool_result)

        tool_result_json = json.dumps(tool_result)

        save_message(
            session_id=session_id,
            role="tool",
            content=tool_result_json,
            tool_call_id=tool_call.id
        )

        # Send tool result back to AI
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result_json
            }
        )

    # SECOND AI CALL
    # AI now sees real tool results
    final_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    final_answer = final_response.choices[0].message.content

    save_message(
        session_id=session_id,
        role="assistant",
        content=final_answer
    )

    return {
        "session_id": session_id,
        "response": final_answer
    }