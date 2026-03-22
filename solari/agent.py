#!/usr/bin/env python3
"""
Solari Agent — Grounded AI assistant that knows before it acts.

Connects to any LLM provider (Claude, GPT, Codex, Ollama, or any
OpenAI-compatible endpoint) and grounds every response in your
local knowledge minds.

Usage:
    solari agent "what should I focus on today" --minds work,research
    solari agent --provider openai --model gpt-4o "explain this codebase"
    solari agent --provider ollama --model qwen2.5:7b "summarize my notes"
    solari agent --provider anthropic "review this code" --minds security

Providers:
    anthropic   — Claude (requires ANTHROPIC_API_KEY)
    openai      — GPT / Codex (requires OPENAI_API_KEY)
    ollama      — Local models via Ollama (default: http://localhost:11434)
    custom      — Any OpenAI-compatible API (requires --base-url)

The agent queries your minds FIRST, then sends the grounded context
to your LLM. Your AI gets real knowledge, not hallucinations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError


DEFAULT_MINDS_DIR = Path("./minds/")

# ── Provider Configs ────────────────────────────────────────────────────

PROVIDERS = {
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com/v1/messages",
        "default_model": "claude-sonnet-4-20250514",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "default_model": "gpt-4o",
    },
    "ollama": {
        "env_key": None,
        "base_url": "http://localhost:11434/api/chat",
        "default_model": "qwen2.5:7b",
    },
    "custom": {
        "env_key": "CUSTOM_API_KEY",
        "base_url": None,
        "default_model": "default",
    },
}


# ── Knowledge Grounding ────────────────────────────────────────────────

def ground_query(query: str, minds_dir: Path, mind_names: Optional[List[str]] = None, top_k: int = 5) -> str:
    """Query local minds for grounding context."""
    from solari.query import query as faiss_query

    results = faiss_query(
        text=query,
        minds_dir=minds_dir,
        mind_names=mind_names,
        top_k=top_k,
    )

    if not results:
        return ""

    lines = ["Relevant knowledge from your minds:\n"]
    for r in results:
        mind = r.get("mind", "unknown")
        sim = r.get("similarity", 0)
        content = r.get("content", "")
        lines.append(f"[{mind}] (relevance: {sim:.2f})\n{content}\n")

    return "\n".join(lines)


# ── Provider Calls ──────────────────────────────────────────────────────

def call_anthropic(prompt: str, grounding: str, model: str, api_key: str) -> str:
    """Call Anthropic Claude API."""
    system = (
        "You are a knowledgeable assistant grounded in the user's own knowledge base. "
        "Use the provided knowledge context to give accurate, sourced answers. "
        "If the knowledge context doesn't cover the question, say so honestly."
    )

    user_content = prompt
    if grounding:
        user_content = f"{grounding}\n\n---\n\nUser question: {prompt}"

    body = json.dumps({
        "model": model,
        "max_tokens": 2048,
        "system": system,
        "messages": [{"role": "user", "content": user_content}],
    }).encode()

    req = Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    resp = urlopen(req, timeout=120)
    data = json.loads(resp.read())
    return data["content"][0]["text"]


def call_openai(prompt: str, grounding: str, model: str, api_key: str, base_url: str = None) -> str:
    """Call OpenAI-compatible API (GPT, Codex, or any compatible endpoint)."""
    url = base_url or "https://api.openai.com/v1/chat/completions"

    system = (
        "You are a knowledgeable assistant grounded in the user's own knowledge base. "
        "Use the provided knowledge context to give accurate, sourced answers. "
        "If the knowledge context doesn't cover the question, say so honestly."
    )

    user_content = prompt
    if grounding:
        user_content = f"{grounding}\n\n---\n\nUser question: {prompt}"

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 2048,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = Request(url, data=body, headers=headers)
    resp = urlopen(req, timeout=120)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def call_ollama(prompt: str, grounding: str, model: str, base_url: str = None) -> str:
    """Call local Ollama instance."""
    url = base_url or "http://localhost:11434/api/chat"

    system = (
        "You are a knowledgeable assistant grounded in the user's own knowledge base. "
        "Use the provided knowledge context to give accurate, sourced answers. "
        "If the knowledge context doesn't cover the question, say so honestly."
    )

    user_content = prompt
    if grounding:
        user_content = f"{grounding}\n\n---\n\nUser question: {prompt}"

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
    }).encode()

    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    resp = urlopen(req, timeout=300)
    data = json.loads(resp.read())
    return data.get("message", {}).get("content", "No response from model.")


# ── Main ────────────────────────────────────────────────────────────────

def run_agent(
    prompt: str,
    provider: str = "ollama",
    model: str = None,
    minds_dir: Path = DEFAULT_MINDS_DIR,
    mind_names: Optional[List[str]] = None,
    base_url: str = None,
    top_k: int = 5,
) -> str:
    """Run the grounded agent: query minds, then call LLM."""

    config = PROVIDERS.get(provider)
    if not config:
        return f"Unknown provider: {provider}. Available: {', '.join(PROVIDERS.keys())}"

    model = model or config["default_model"]
    url = base_url or config["base_url"]

    # Get API key if needed
    api_key = None
    if config["env_key"]:
        api_key = os.environ.get(config["env_key"], "")
        if not api_key and provider not in ("ollama",):
            return (
                f"Missing API key. Set {config['env_key']} environment variable.\n"
                f"Example: export {config['env_key']}=\"your-key-here\""
            )

    # Ground the query in local knowledge
    grounding = ""
    if minds_dir.is_dir():
        grounding = ground_query(prompt, minds_dir, mind_names, top_k)

    # Call the provider
    if provider == "anthropic":
        return call_anthropic(prompt, grounding, model, api_key)
    elif provider == "ollama":
        return call_ollama(prompt, grounding, model, url)
    else:  # openai, custom
        return call_openai(prompt, grounding, model, api_key, url)


def main():
    parser = argparse.ArgumentParser(
        description="Solari Agent — Grounded AI assistant that knows before it acts.",
    )
    parser.add_argument("prompt", nargs="?", help="Your question or instruction")
    parser.add_argument(
        "--provider", "-p",
        choices=list(PROVIDERS.keys()),
        default="ollama",
        help="AI provider (default: ollama)",
    )
    parser.add_argument("--model", "-m", help="Model name (uses provider default if not set)")
    parser.add_argument(
        "--minds-dir",
        type=Path,
        default=DEFAULT_MINDS_DIR,
        help="Directory containing your minds (default: ./minds/)",
    )
    parser.add_argument(
        "--minds",
        type=str,
        default=None,
        help="Comma-separated list of specific minds to query",
    )
    parser.add_argument("--base-url", help="Custom API base URL (for 'custom' provider)")
    parser.add_argument("--top", type=int, default=5, help="Number of knowledge results to ground with")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output as JSON")

    args = parser.parse_args()

    if not args.prompt:
        parser.print_help()
        sys.exit(1)

    mind_names = args.minds.split(",") if args.minds else None

    try:
        response = run_agent(
            prompt=args.prompt,
            provider=args.provider,
            model=args.model,
            minds_dir=args.minds_dir,
            mind_names=mind_names,
            base_url=args.base_url,
            top_k=args.top,
        )
    except HTTPError as e:
        body = e.read().decode()[:200] if hasattr(e, 'read') else ""
        response = f"API error ({e.code}): {body}"
    except Exception as e:
        response = f"Error: {e}"

    if args.json_output:
        print(json.dumps({"response": response, "provider": args.provider}))
    else:
        print(response)


if __name__ == "__main__":
    main()
