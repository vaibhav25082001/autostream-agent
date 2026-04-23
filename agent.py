"""
AutoStream Conversational AI Agent
====================================
Built for : ServiceHive — ML Intern Assignment
Stack     : Python · LangGraph · LangChain · Local RAG · Tool Execution
LLMs      : Claude 3 Haiku  |  GPT-4o-mini  |  Gemini 1.5 Flash
            (Controlled via LLM_PROVIDER in .env)
"""

import json
import os
import re
import sys
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

from llm_loader import load_llm   # noqa: E402  (after dotenv)


# ─────────────────────────────────────────────────────────────────────────────
# 1. KNOWLEDGE BASE  (Local RAG – JSON file)
# ─────────────────────────────────────────────────────────────────────────────

def load_knowledge_base() -> dict:
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.json")
    with open(kb_path, "r") as f:
        return json.load(f)

KB = load_knowledge_base()


def retrieve_context(query: str) -> str:
    """
    Keyword-based retrieval over the local KB.
    Returns only the sections relevant to the query.
    """
    q = query.lower()
    chunks = []

    pricing_kw = ["price", "plan", "cost", "basic", "pro", "subscription",
                  "how much", "fee", "tier", "4k", "720p", "unlimited", "caption"]
    policy_kw  = ["refund", "support", "cancel", "policy", "return",
                  "24/7", "help", "assistance"]
    product_kw = ["autostream", "what is", "about", "feature", "tool",
                  "video", "editing", "content", "creator"]

    if any(w in q for w in pricing_kw):
        chunks.append("## AutoStream Pricing\n" + json.dumps(KB["pricing"], indent=2))
    if any(w in q for w in policy_kw):
        chunks.append("## Company Policies\n" + json.dumps(KB["policies"], indent=2))
    if any(w in q for w in product_kw):
        chunks.append("## Product Overview\n" + json.dumps(KB["product"], indent=2))

    # Fallback: return everything
    if not chunks:
        chunks.append("## Full Knowledge Base\n" + json.dumps(KB, indent=2))

    return "\n\n".join(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MOCK LEAD CAPTURE TOOL
# ─────────────────────────────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulates saving a qualified lead to the backend.
    Called ONLY when name + email + platform are ALL confirmed.
    """
    print(f"\n{'=' * 58}")
    print(f"  ✅  Lead captured successfully!")
    print(f"      Name     : {name}")
    print(f"      Email    : {email}")
    print(f"      Platform : {platform}")
    print(f"{'=' * 58}\n")
    return f"Lead captured: {name} | {email} | {platform}"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LANGGRAPH STATE
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:        list   # Full conversation history [{"role":…, "content":…}]
    intent:          str    # casual_greeting | product_inquiry | high_intent
    lead_info:       dict   # Accumulates: name, email, platform
    collecting_lead: bool   # True while mid-collection (overrides routing)
    lead_captured:   bool   # True after mock_lead_capture() fires
    response:        str    # The assistant reply for the current turn


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_lc_messages(history: list, system_prompt: str, last_n: int = 8) -> list:
    """Build a LangChain message list from raw conversation history."""
    msgs = [SystemMessage(content=system_prompt)]
    for m in history[-last_n:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(content=m["content"]))
        else:
            msgs.append(AIMessage(content=m["content"]))
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
# 5. NODE — Intent Classifier
# ─────────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM = """You are an intent classifier for AutoStream's AI assistant.

Given the conversation history, classify the LATEST user message into exactly ONE of:
  casual_greeting  — simple greetings, small talk, "hi", "hello", "thanks"
  product_inquiry  — questions about features, pricing, plans, policies
  high_intent      — user clearly wants to sign up, try, buy, subscribe, or get started

Rules:
- Reply with ONLY the label (one of the three above). No explanation, no punctuation.
- If the user says they want to try / sign up / buy / get started → always high_intent.
- When in doubt between product_inquiry and high_intent → choose product_inquiry."""

def node_classify_intent(state: AgentState, llm) -> AgentState:
    msgs   = build_lc_messages(state["messages"], INTENT_SYSTEM, last_n=6)
    result = llm.invoke(msgs)
    raw    = result.content.strip().lower().replace("-", "_").split()[0]

    valid  = {"casual_greeting", "product_inquiry", "high_intent"}
    intent = raw if raw in valid else "product_inquiry"

    return {**state, "intent": intent}


# ─────────────────────────────────────────────────────────────────────────────
# 6. NODE — Greeting
# ─────────────────────────────────────────────────────────────────────────────

GREET_SYSTEM = """You are AutoStream's friendly AI assistant for content creators.
AutoStream provides automated video editing tools for content creators.

Respond warmly and briefly mention you can help with:
- Pricing and plan details
- Product features
- Getting started / signing up

Keep it to 2–3 sentences. No bullet points."""

def node_greet(state: AgentState, llm) -> AgentState:
    msgs   = build_lc_messages(state["messages"], GREET_SYSTEM, last_n=4)
    result = llm.invoke(msgs)
    reply  = result.content.strip()
    return {**state,
            "messages": state["messages"] + [{"role": "assistant", "content": reply}],
            "response": reply}


# ─────────────────────────────────────────────────────────────────────────────
# 7. NODE — RAG Response
# ─────────────────────────────────────────────────────────────────────────────

def node_rag_respond(state: AgentState, llm) -> AgentState:
    last_user = state["messages"][-1]["content"]
    context   = retrieve_context(last_user)

    system = f"""You are AutoStream's knowledgeable AI assistant.
AutoStream is a SaaS product offering automated video editing tools for content creators.

Answer the user's question using ONLY the context below. Do not fabricate information.
If the context does not cover the question, say you'll connect them with the support team.

CONTEXT:
{context}

Be concise, accurate, and friendly. Format prices and features clearly."""

    msgs   = build_lc_messages(state["messages"], system, last_n=8)
    result = llm.invoke(msgs)
    reply  = result.content.strip()
    return {**state,
            "messages": state["messages"] + [{"role": "assistant", "content": reply}],
            "response": reply}


# ─────────────────────────────────────────────────────────────────────────────
# 8. NODE — Lead Handler  (collection + tool execution)
# ─────────────────────────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """Extract user information from the message if explicitly stated.
Look for:
  - name:     person's first or full name
  - email:    a valid email address
  - platform: creator platform such as YouTube, Instagram, TikTok, Twitter, LinkedIn, etc.

Reply ONLY with valid JSON. Example:
{"name": "Alice", "email": "alice@example.com", "platform": "YouTube"}

Use null for any field not found. Never guess or invent values."""

def node_lead_handler(state: AgentState, llm) -> AgentState:
    lead          = dict(state["lead_info"])   # copy existing info
    last_user_msg = state["messages"][-1]["content"]

    # ── Extract any new info from the latest user message ───────────────────
    extract_result = llm.invoke([
        SystemMessage(content=EXTRACT_SYSTEM),
        HumanMessage(content=last_user_msg)
    ])
    try:
        raw_json  = re.sub(r"```json|```", "", extract_result.content).strip()
        extracted = json.loads(raw_json)
        for field in ["name", "email", "platform"]:
            if extracted.get(field) and not lead.get(field):
                lead[field] = extracted[field]
    except (json.JSONDecodeError, AttributeError):
        pass  # extraction failed; keep existing info and continue

    # ── Check what is still missing ─────────────────────────────────────────
    missing = [f for f in ["name", "email", "platform"] if not lead.get(f)]

    # ── All three collected → fire the tool ─────────────────────────────────
    if not missing:
        mock_lead_capture(lead["name"], lead["email"], lead["platform"])
        reply = (
            f"🎉 You're all set, {lead['name']}! "
            f"We've captured your details and our team will reach out at {lead['email']} soon. "
            f"Welcome to AutoStream — excited to see your work on {lead['platform']}! 🚀"
        )
        return {
            **state,
            "messages":        state["messages"] + [{"role": "assistant", "content": reply}],
            "lead_info":       lead,
            "collecting_lead": False,
            "lead_captured":   True,
            "response":        reply
        }

    # ── Still missing info → ask for the next field ─────────────────────────
    field_prompts = {
        "name":     "Could you please share your name?",
        "email":    "What email address should we use to reach you?",
        "platform": "Which creator platform do you mainly use? (e.g., YouTube, Instagram, TikTok)"
    }
    collected_summary = ", ".join(f"{k}: {v}" for k, v in lead.items() if v) or "nothing yet"
    next_field        = missing[0]

    ask_system = f"""You are AutoStream's AI assistant collecting signup details.

Already collected : {collected_summary}
Still needed      : {next_field}
Question to ask   : {field_prompts[next_field]}

Write a warm, natural 1–2 sentence message that:
1. Briefly acknowledges what the user just said (if they provided info).
2. Asks for the '{next_field}' in a friendly way.
Do NOT ask for fields already collected."""

    ask_result = llm.invoke([
        SystemMessage(content=ask_system),
        HumanMessage(content=last_user_msg)
    ])
    reply = ask_result.content.strip()

    return {
        **state,
        "messages":        state["messages"] + [{"role": "assistant", "content": reply}],
        "lead_info":       lead,
        "collecting_lead": True,
        "response":        reply
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. ROUTING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """
    Priority:
    1. Lead already captured        → rag  (answer follow-up questions normally)
    2. Mid-collection               → lead (keep collecting, ignoring intent)
    3. Route by classified intent
    """
    if state["lead_captured"]:
        return "rag"
    if state["collecting_lead"]:
        return "lead"
    if state["intent"] == "casual_greeting":
        return "greet"
    if state["intent"] == "high_intent":
        return "lead"
    return "rag"   # product_inquiry


# ─────────────────────────────────────────────────────────────────────────────
# 10. BUILD LANGGRAPH
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(llm):
    """
    Builds the LangGraph StateGraph.
    Each node is a closure that captures the `llm` object.
    This makes the graph LLM-agnostic at build time.
    """
    builder = StateGraph(AgentState)

    # Register nodes (wrap each function so llm is injected)
    builder.add_node("classify", lambda s: node_classify_intent(s, llm))
    builder.add_node("greet",    lambda s: node_greet(s, llm))
    builder.add_node("rag",      lambda s: node_rag_respond(s, llm))
    builder.add_node("lead",     lambda s: node_lead_handler(s, llm))

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"greet": "greet", "rag": "rag", "lead": "lead"}
    )

    builder.add_edge("greet", END)
    builder.add_edge("rag",   END)
    builder.add_edge("lead",  END)

    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 11. CLI RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Load the LLM (provider picked from .env LLM_PROVIDER)
    try:
        llm = load_llm()
    except (EnvironmentError, ValueError) as e:
        print(f"\n❌  Error: {e}")
        print("    Check your .env file and try again.\n")
        sys.exit(1)

    graph = build_graph(llm)

    # Initial state — persists across all turns in this session
    state: AgentState = {
        "messages":        [],
        "intent":          "",
        "lead_info":       {},
        "collecting_lead": False,
        "lead_captured":   False,
        "response":        ""
    }

    provider_label = os.getenv("LLM_PROVIDER", "claude").upper()

    print("\n" + "=" * 60)
    print(f"  🎬  AutoStream AI Assistant  [LLM: {provider_label}]")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAutoStream: Thanks for chatting! Goodbye! 👋\n")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("\nAutoStream: Thanks for stopping by! Have a great day! 👋\n")
            break

        # Append user message to history
        state["messages"] = state["messages"] + [{"role": "user", "content": user_input}]

        # Run one turn through the graph
        state = graph.invoke(state)

        print(f"\nAutoStream: {state['response']}\n")


if __name__ == "__main__":
    main()
