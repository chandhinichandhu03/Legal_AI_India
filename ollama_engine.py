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


def get_predefined_answer(question: str) -> str:
    """Check if the question matches common legal queries and return a high-quality answer."""
    import re
    q = question.lower().strip()
    clean = re.sub(r'[^\w\s]', '', q)
    
    if "pocso" in clean:
        return (
            "### ⚖️ Legal Reference: POCSO Act, 2012 (Local Fallback)\n\n"
            "The **Protection of Children from Sexual Offences (POCSO) Act, 2012** is a comprehensive law enacted to protect children "
            "(defined as persons below 18 years of age) from offences of sexual assault, sexual harassment, and pornography.\n\n"
            "#### Key Features of the Act:\n"
            "- **Child-Friendly Procedures:** Mandates child-friendly mechanisms for reporting, recording, and investigating cases (e.g., police officers must wear civil clothes, and statements must be recorded at the child's residence).\n"
            "- **Special POCSO Courts:** Designated for speedy trials, with a mandate to dispose of cases within one year of reporting.\n"
            "- **Anonymity and Privacy:** Strictly prohibits disclosing the identity, address, school, or photographs of the child in any media outlet (Section 74).\n\n"
            "#### Major Penal provisions:\n"
            "- **Section 4 (Penetrative Sexual Assault):** Imprisonment of not less than 10 years, which may extend to life imprisonment, along with a fine.\n"
            "- **Section 6 (Aggravated Penetrative Sexual Assault):** Imprisonment of not less than 20 years, which may extend to life imprisonment or death penalty.\n"
            "- **Section 12 (Sexual Harassment):** Imprisonment up to 3 years and a fine.\n\n"
            "*(Note: Since Ollama is currently offline, this direct answer was provided by the local Legal database fallback.)*"
        )
        
    if ("bailable" in clean and "nonbailable" in clean) or ("difference" in clean and "bailable" in clean):
        return (
            "### ⚖️ Legal Reference: Bailable vs. Non-Bailable Offences (Local Fallback)\n\n"
            "In Indian Criminal Law (defined under Section 2(a) of the Code of Criminal Procedure - CrPC), offences are classified into two main types:\n\n"
            "#### 1. Bailable Offences\n"
            "- **Definition:** Less serious, minor offences where bail is a matter of **right** for the accused.\n"
            "- **Examples:** Simple hurt (IPC 323), public nuisance (IPC 290), bribery (PCA 7), and writing obscene songs/acts in public (IPC 294).\n"
            "- **Bail Procedure:** The police officer in charge or the court is legally obligated to release the accused on bail as soon as they furnish the required surety or personal bond.\n\n"
            "#### 2. Non-Bailable Offences\n"
            "- **Definition:** Serious, grave offences where bail is **not a matter of right**; instead, it is at the sole discretion of the Court.\n"
            "- **Examples:** Murder (IPC 302), Rape (IPC 376), Cheating/Fraud (IPC 420), and Kidnapping (IPC 363).\n"
            "- **Bail Procedure:** The accused must apply to the Magistrate or Sessions Court. The court decides whether to grant bail based on factors like the nature and gravity of the crime, danger of the accused fleeing, or risk of tampering with evidence.\n\n"
            "*(Note: Since Ollama is currently offline, this direct answer was provided by the local Legal database fallback.)*"
        )
        
    if "high court" in clean and ("file" in clean or "procedure" in clean or "case" in clean):
        return (
            "### ⚖️ Legal Reference: Procedure to File a Case in the High Court (Local Fallback)\n\n"
            "Filing a case (such as a Writ Petition under Article 226/227, an Appeal, or a Revision Petition) in an Indian High Court follows these steps:\n\n"
            "1. **Drafting the Petition:** An advocate drafts the petition detailing the facts of the case, legal questions, grounds of challenge, and the specific relief/remedy sought. \n"
            "2. **Affidavit & Documentation:** The petitioner must sign an affidavit in the presence of an authorized Oath Commissioner. Supporting documents (like impugned orders, annexures) must be indexed and paginated.\n"
            "3. **Payment of Court Fee:** Appropriate court fees in the form of judicial stamp papers must be paid and attached to the file.\n"
            "4. **Filing at the Registry:** The petition is filed physically (or via e-filing) at the High Court's Filing Counter.\n"
            "5. **Scrutiny & Clearing Objections:** The High Court Registry scrutinizes the file for defects (such as missing documents, wrong formats, or unpaid fees). If objections are raised, the advocate has a specified time to rectify them.\n"
            "6. **Listing and Admission:** Once cleared, the case is listed before a Judge/Bench. The judge hears preliminary arguments to decide if the case is fit to be 'admitted' and notices should be issued to the opposite parties.\n\n"
            "*(Note: Since Ollama is currently offline, this direct answer was provided by the local Legal database fallback.)*"
        )
        
    if ("arrest" in clean and ("right" in clean or "rights" in clean)) or ("person" in clean and "arrested" in clean):
        return (
            "### ⚖️ Legal Reference: Rights of an Arrested Person under CrPC (Local Fallback)\n\n"
            "The Code of Criminal Procedure (CrPC) and the Constitution of India protect an arrested person through several fundamental rights:\n\n"
            "1. **Right to Know Grounds of Arrest (Section 50 CrPC / Article 22(1) Constitution):** The police must immediately inform the arrested person of the exact offence and grounds for their arrest.\n"
            "2. **Right to be Informed of Bail (Section 50(2) CrPC):** If arrested for a bailable offence, the police must inform the person that they are entitled to release on bail.\n"
            "3. **Right to be Produced within 24 Hours (Section 56 & 57 CrPC / Article 22(2) Constitution):** The police must bring the arrested person before the nearest judicial magistrate within 24 hours of arrest, excluding travel time.\n"
            "4. **Right to Legal Counsel (Section 41D CrPC / Article 22(1) Constitution):** The arrested person has the right to consult and be defended by a lawyer of their choice, including meeting them during interrogation.\n"
            "5. **Right to Inform a Relative/Friend (Section 50A CrPC):** The police must immediately notify a nominated relative or friend about the arrest and place of detention.\n"
            "6. **Right to Medical Examination (Section 54 CrPC):** The person has a right to be medically examined by a registered medical officer to document any pre-existing injuries or physical state.\n"
            "7. **Right to Remain Silent (Article 20(3) Constitution):** Protects the arrested person against self-incrimination; they cannot be forced to confess or make statements that incriminate themselves.\n\n"
            "*(Note: Since Ollama is currently offline, this direct answer was provided by the local Legal database fallback.)*"
        )
        
    return ""


def query_local_reference(question: str) -> str:
    """Helper to query local CSV datasets and return matching legal references."""
    import csv
    import re
    from config import INDIAN_LAWS_CSV, IPC_SECTIONS_CSV
    
    # Try predefined high-quality answers first
    predefined = get_predefined_answer(question)
    if predefined:
        return predefined
        
    clean_q = re.sub(r'[^\w\s]', ' ', question.lower()).strip()
    words = [w for w in clean_q.split() if len(w) > 2]
    
    stopwords = {
        'what', 'is', 'the', 'how', 'who', 'when', 'where', 'why', 'can', 'you',
        'explain', 'describe', 'about', 'under', 'act', 'law', 'section', 'for',
        'punishment', 'penalty', 'fine', 'offence', 'crime', 'in', 'india', 'indian',
        'defined', 'definition', 'provide', 'provisions'
    }
    keywords = [w for w in words if w not in stopwords]
    sec_numbers = re.findall(r'\b\d+[A-Za-z]?\b', question)
    
    matched_acts = []
    matched_sections = []
    
    # Try querying the DB first (using LegalAct table)
    try:
        from schema import LegalAct
        from sqlalchemy import or_
        
        db_keywords = [f"%{kw}%" for kw in keywords if len(kw) >= 3]
        if db_keywords:
            or_filters = []
            for pat in db_keywords:
                or_filters.append(LegalAct.act_name.ilike(pat))
                or_filters.append(LegalAct.section_title.ilike(pat))
                or_filters.append(LegalAct.section_content.ilike(pat))
                
            db_acts = LegalAct.query.filter(or_(*or_filters)).limit(5).all()
            for a in db_acts:
                matched_sections.append({
                    "act_name": a.act_name,
                    "section": a.section_number or "",
                    "title": a.section_title or "",
                    "description": a.section_content or "",
                    "punishment": a.punishment or "",
                    "bailable": "yes" if a.bailable else ("no" if a.bailable is False else ""),
                    "cognizable": "yes" if a.cognizable else ("no" if a.cognizable is False else "")
                })
        
        if sec_numbers:
            db_secs = LegalAct.query.filter(LegalAct.section_number.in_(sec_numbers)).limit(5).all()
            for s in db_secs:
                # Add to start of list to prioritize
                matched_sections.insert(0, {
                    "act_name": s.act_name,
                    "section": s.section_number or "",
                    "title": s.section_title or "",
                    "description": s.section_content or "",
                    "punishment": s.punishment or "",
                    "bailable": "yes" if s.bailable else ("no" if s.bailable is False else ""),
                    "cognizable": "yes" if s.cognizable else ("no" if s.cognizable is False else "")
                })
    except Exception as db_err:
        logger.debug(f"DB lookup fallback failed: {db_err}")
        
    # If DB lookup yielded nothing or failed, load from CSV files
    if not matched_sections:
        # 1. Search indian_laws.csv
        if INDIAN_LAWS_CSV.exists():
            try:
                with open(INDIAN_LAWS_CSV, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        act_name = row.get("act_name", "")
                        desc = row.get("description", "")
                        key_prov = row.get("key_provisions", "")
                        
                        match_score = 0
                        for kw in keywords:
                            if kw in act_name.lower():
                                match_score += 10
                            if kw in desc.lower():
                                match_score += 2
                        
                        for num in sec_numbers:
                            if num in key_prov.split():
                                match_score += 5
                                
                        if match_score > 0:
                            matched_acts.append((row, match_score))
            except Exception as e:
                logger.debug(f"CSV read error (indian_laws.csv): {e}")
                
        # 2. Search ipc_sections.csv
        if IPC_SECTIONS_CSV.exists():
            try:
                with open(IPC_SECTIONS_CSV, newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        act_name = row.get("act_name", "")
                        sec_num = row.get("section", "")
                        title = row.get("title", "")
                        desc = row.get("description", "")
                        
                        match_score = 0
                        if sec_num in sec_numbers:
                            match_score += 20
                            
                        for kw in keywords:
                            if kw in act_name.lower():
                                match_score += 3
                            if kw in title.lower():
                                match_score += 5
                            if kw in desc.lower():
                                match_score += 2
                                
                        if match_score > 0:
                            matched_sections.append((row, match_score))
            except Exception as e:
                logger.debug(f"CSV read error (ipc_sections.csv): {e}")
                
        matched_acts.sort(key=lambda x: x[1], reverse=True)
        matched_sections.sort(key=lambda x: x[1], reverse=True)
        
        # Extract rows from scoring tuples for rendering
        matched_acts = [act for act, score in matched_acts]
        matched_sections = [sec for sec, score in matched_sections]
        
    if not matched_acts and not matched_sections:
        return ""
        
    answer_parts = []
    answer_parts.append("### ⚖️ Legal Database Fallback Answer (Ollama Offline)\n")
    answer_parts.append(f"Ollama/Llama3 is currently offline, but I found these relevant matches in the local Indian Law reference database:\n")
    
    # Render Act details
    if matched_acts:
        for act in matched_acts[:2]:
            year = act.get("year", "")
            year_str = f" ({year})" if year else ""
            category = act.get("category", "General")
            desc = act.get("description", "")
            court = act.get("court", "")
            key_prov = act.get("key_provisions", "")
            
            answer_parts.append(f"#### 📜 {act.get('act_name', '')}{year_str}")
            answer_parts.append(f"- **Category:** {category}")
            answer_parts.append(f"- **Description:** {desc}")
            if court:
                answer_parts.append(f"- **Jurisdictional Court:** {court}")
            if key_prov:
                answer_parts.append(f"- **Key Provisions:** {key_prov}")
            answer_parts.append("")
            
    # Render Section details
    if matched_sections:
        answer_parts.append("#### 📑 Relevant Provisions & Punishments")
        # Deduplicate matched sections by (act_name, section)
        seen = set()
        unique_secs = []
        for s in matched_sections:
            k = (s.get("act_name", "").strip(), s.get("section", "").strip())
            if k not in seen:
                seen.add(k)
                unique_secs.append(s)
                
        for sec in unique_secs[:4]:
            act_name = sec.get("act_name", "")
            sec_num = sec.get("section", "")
            title = sec.get("title", "")
            desc = sec.get("description", "")
            punish = sec.get("punishment", "")
            
            bail_val = sec.get("bailable", "").strip().lower()
            cog_val = sec.get("cognizable", "").strip().lower()
            
            bail_str = "Bailable" if bail_val in ("yes", "true") else ("Non-bailable" if bail_val in ("no", "false") else "")
            cog_str = "Cognizable" if cog_val in ("yes", "true") else ("Non-cognizable" if cog_val in ("no", "false") else "")
            status_parts = [p for p in [bail_str, cog_str] if p]
            status_str = f" ({', '.join(status_parts)})" if status_parts else ""
            
            answer_parts.append(f"- **{act_name} — Section {sec_num}: {title}**{status_str}")
            answer_parts.append(f"  *Description:* {desc}")
            if punish:
                answer_parts.append(f"  *Punishment:* {punish}")
            answer_parts.append("")
            
    answer_parts.append("*Note: Since the local Ollama LLM server is offline, this structured reference was retrieved directly from the offline Indian Law database.*")
    return "\n".join(answer_parts)


def _fallback_response(question: str, context: str, model: str, error: str) -> Dict:
    """Generate a fallback response when Ollama is unavailable."""
    logger.warning(f"Ollama unavailable ({error}), using fallback.")

    # Try to search local database/reference
    local_ref = query_local_reference(question)
    
    if local_ref:
        answer = local_ref
        if context:
            answer = f"{local_ref}\n\n---\n**Retrieved Case context:**\n{context[:800]}"
    elif context:
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
        "success": True,
        "error": error,
        "hallucination_score": 0.0,
        "hallucination_flag": False,
    }
