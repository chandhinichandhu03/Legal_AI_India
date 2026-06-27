"""
ollama_engine.py — Ollama LLM Integration
==========================================
generate_response(question, context) — calls local Ollama API.
Supports llama3, mistral, gemma3. Includes hallucination detection.
"""

import re
import json
import time
import requests
from typing import Dict, List, Optional, Generator

from config import (
    OLLAMA_API_GENERATE, OLLAMA_API_TAGS, OLLAMA_DEFAULT_MODEL,
    OLLAMA_TIMEOUT, OLLAMA_MAX_TOKENS, OLLAMA_TEMPERATURE,
    ENABLE_HALLUCINATION_CHECK
)
from utils import logger


# ── System Prompts ────────────────────────────────────────────────────────────

LEGAL_SYSTEM_PROMPT = """You are an expert AI Legal Assistant specializing in Indian Law.

Your role:
- Answer questions about Indian law accurately and clearly
- Reference specific IPC sections, Acts, and legal provisions when relevant
- Provide practical legal guidance based on Indian courts and legal system
- Cite relevant Supreme Court and High Court judgments when applicable
- Clearly state when professional legal advice is needed

Guidelines:
- Be precise and cite specific sections (e.g., IPC Section 420, CrPC Section 125)
- Explain legal terms in simple language when needed
- Mention relevant courts (Supreme Court, High Court, District Court, etc.)
- Include important procedural steps when explaining legal processes
- Always recommend consulting a qualified advocate for specific legal matters

Context from uploaded documents will be provided. Use it to give accurate, grounded answers."""

HALLUCINATION_CHECK_PROMPT = """Compare the AI answer to the retrieved context and rate hallucination risk.

Retrieved Context:
{context}

AI Answer:
{answer}

Respond with ONLY a JSON object:
{{"hallucination_score": 0.0-1.0, "is_grounded": true/false, "unsupported_claims": ["list"]}}

Score: 0.0=fully grounded, 1.0=completely hallucinated."""


# ── Ollama API Functions ──────────────────────────────────────────────────────

def check_ollama_status() -> Dict:
    """Check if Ollama server is running and list available models."""
    try:
        resp = requests.get(OLLAMA_API_TAGS, timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"online": True, "models": models}
        return {"online": False, "models": []}
    except requests.exceptions.ConnectionError:
        return {"online": False, "models": [], "error": "Ollama not running"}
    except Exception as e:
        return {"online": False, "models": [], "error": str(e)}


def generate_response(
    question: str,
    context: str = "",
    model: str = None,
    conversation_history: List[Dict] = None,
    max_tokens: int = OLLAMA_MAX_TOKENS,
    temperature: float = OLLAMA_TEMPERATURE,
) -> Dict:
    """
    Generate a legal AI response using Ollama.

    Args:
        question: User's legal question
        context: Retrieved document context from RAG
        model: Ollama model name (default from config)
        conversation_history: List of {"role": "user"/"assistant", "content": "..."} dicts
        max_tokens: Maximum response tokens
        temperature: Model temperature

    Returns:
        Dict with answer, model, response_time, tokens_used
    """
    model = model or OLLAMA_DEFAULT_MODEL
    
    # Pre-check Ollama status and auto-map to installed models
    status = check_ollama_status()
    if not status.get("online"):
        return _fallback_response(question, context, model, "Ollama server is not running. Please start Ollama.")
        
    available_models = status.get("models", [])
    if not available_models:
        return _fallback_response(
            question, 
            context, 
            model, 
            "Ollama is running, but no models are installed. Please run `ollama pull llama3` in your terminal."
        )
        
    # Map preferred model to installed models
    clean_available = [m.split(":")[0] for m in available_models] + available_models
    if model not in clean_available:
        matched_model = None
        for m in available_models:
            if model.lower() in m.lower() or m.lower() in model.lower():
                matched_model = m
                break
        if matched_model:
            model = matched_model
            logger.info(f"Ollama auto-mapped model to '{model}'")
        else:
            model = available_models[0]
            logger.warning(f"Preferred model not found, falling back to installed model '{model}'")
    else:
        # Match exact tagged name from the list
        for m in available_models:
            if m == model or m.split(":")[0] == model:
                model = m
                break

    start_time = time.time()

    # Build prompt
    prompt = _build_prompt(question, context, conversation_history)

    payload = {
        "model": model,
        "prompt": prompt,
        "system": LEGAL_SYSTEM_PROMPT,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }
    }

    try:
        resp = requests.post(
            OLLAMA_API_GENERATE,
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("response", "").strip()
        response_time = time.time() - start_time
        tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        result = {
            "answer": answer,
            "model": model,
            "response_time": round(response_time, 2),
            "tokens_used": tokens_used,
            "success": True,
        }

        # Hallucination check
        if ENABLE_HALLUCINATION_CHECK and context:
            hall_result = check_hallucination(answer, context, model)
            result.update(hall_result)

        return result

    except requests.exceptions.ConnectionError:
        return _fallback_response(question, context, model, "Ollama server not running")
    except requests.exceptions.Timeout:
        return _fallback_response(question, context, model, "Ollama timeout")
    except Exception as e:
        logger.error(f"Ollama generate error: {e}")
        return _fallback_response(question, context, model, str(e))


def generate_summary(text: str, model: str = None) -> str:
    """Generate an AI-powered summary of a legal document."""
    model = model or OLLAMA_DEFAULT_MODEL
    prompt = f"""Summarize the following Indian legal document in 5-6 clear sentences.
Include: case type, main parties, key legal issues, relevant IPC/Act sections, and outcome.

Document:
{text[:3000]}

Summary:"""

    try:
        resp = requests.post(
            OLLAMA_API_GENERATE,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 500, "temperature": 0.2}
            },
            timeout=OLLAMA_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama summary failed: {e}")
        return ""


def check_hallucination(answer: str, context: str, model: str = None) -> Dict:
    """
    Compare LLM answer against retrieved context to detect hallucinations.
    """
    model = model or OLLAMA_DEFAULT_MODEL
    prompt = HALLUCINATION_CHECK_PROMPT.format(
        context=context[:2000],
        answer=answer[:1000]
    )

    try:
        resp = requests.post(
            OLLAMA_API_GENERATE,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 200, "temperature": 0.0}
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")

        # Parse JSON from response
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return {
                "hallucination_score": float(data.get("hallucination_score", 0.3)),
                "hallucination_flag": not data.get("is_grounded", True),
                "unsupported_claims": data.get("unsupported_claims", []),
            }
    except Exception as e:
        logger.debug(f"Hallucination check failed: {e}")

    return {
        "hallucination_score": 0.3,
        "hallucination_flag": False,
        "unsupported_claims": [],
    }


def compute_confidence_score(chunks: List[Dict], answer: str) -> float:
    """
    Compute confidence score based on retrieved chunk quality.
    """
    if not chunks:
        return 0.2
    avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
    # Boost if answer is non-empty and has substantial content
    length_factor = min(len(answer.split()) / 100, 1.0)
    return round(min(avg_score * 0.7 + length_factor * 0.3, 0.99), 4)


def generate_follow_up_questions(question: str, answer: str, model: str = None) -> List[str]:
    """Generate 3 follow-up questions locally and instantly using simple keyword rules."""
    q_lower = question.lower()
    a_lower = answer.lower()
    
    # Check keywords in question or answer and add custom questions
    if "murder" in q_lower or "302" in q_lower or "kill" in q_lower:
        return [
            "What is the penalty under IPC Section 302?",
            "What is the difference between murder and culpable homicide?",
            "How can one apply for bail in a murder case?"
        ]
    elif "cheating" in q_lower or "420" in q_lower or "fraud" in q_lower:
        return [
            "What is the punishment for cheating under IPC Section 420?",
            "Is cheating a bailable offence in India?",
            "What evidence is required to prove criminal breach of trust?"
        ]
    elif "divorce" in q_lower or "marriage" in q_lower or "custody" in q_lower or "maintenance" in q_lower:
        return [
            "What are the grounds for mutual consent divorce under Hindu law?",
            "How is child custody determined by Indian Family Courts?",
            "What is the procedure for claiming maintenance under Section 125 CrPC?"
        ]
    elif "cyber" in q_lower or "hacking" in q_lower or "phishing" in q_lower or "online" in q_lower:
        return [
            "What are the main provisions of Section 66 of the IT Act?",
            "How do I file a complaint with the cyber crime cell?",
            "What is the punishment for identity theft online?"
        ]
    elif "property" in q_lower or "land" in q_lower or "tenant" in q_lower or "rent" in q_lower:
        return [
            "What is the procedure for tenant eviction under state rent laws?",
            "How do I verify the title deed of a property in India?",
            "What are the legal remedies against property encroachment?"
        ]
    elif "accident" in q_lower or "traffic" in q_lower or "compensation" in q_lower:
        return [
            "How is compensation calculated in a MACT claim?",
            "What is the time limit to file an appeal in an accident claim?",
            "What are the penalties for drunk driving under the MV Act?"
        ]
    
    # General legal questions
    return [
        "What is the difference between bailable and non-bailable offences?",
        "What is the procedure to file a case in the High Court?",
        "What are the rights of an arrested person under CrPC?"
    ]


# ── Private Helpers ───────────────────────────────────────────────────────────

def _build_prompt(question: str, context: str, history: List[Dict] = None) -> str:
    """Build the full prompt with context and conversation history."""
    parts = []

    # Conversation history
    if history:
        parts.append("Previous conversation:")
        for turn in history[-5:]:  # Last 5 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        parts.append("")

    # Retrieved context
    if context:
        parts.append("Retrieved Legal Context:")
        parts.append(context[:2500])
        parts.append("")
        parts.append(f"User Question: {question}")
        parts.append("\nProvide a comprehensive, accurate answer based on Indian law and the context above:")
    else:
        parts.append(f"User Question: {question}")
        parts.append("\nProvide a comprehensive, accurate answer based on Indian law. Use your general knowledge of Indian Law since no specific context document was provided:")

    return "\n".join(parts)


def _fallback_response(question: str, context: str, model: str, error: str) -> Dict:
    """Generate a fallback response when Ollama is unavailable."""
    logger.warning(f"Ollama unavailable ({error}), using fallback.")

    # Use context to generate a basic answer
    if context:
        answer = (
            f"⚠️ AI model ({model}) is currently unavailable. "
            f"Based on the retrieved legal documents:\n\n{context[:800]}\n\n"
            f"Please start Ollama with: `ollama run {model}`"
        )
    else:
        answer = (
            f"⚠️ The AI model ({model}) is not available. "
            f"Please ensure Ollama is running: `ollama serve` and `ollama pull {model}`\n\n"
            f"Your question: {question}"
        )

    return {
        "answer": answer,
        "model": model,
        "response_time": 0.0,
        "tokens_used": 0,
        "success": False,
        "error": error,
        "hallucination_score": 0.0,
        "hallucination_flag": False,
    }
