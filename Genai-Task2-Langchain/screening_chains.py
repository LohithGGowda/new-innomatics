"""
chains/screening_chains.py
---------------------------
Defines modular LangChain LCEL chains for each pipeline stage.
Uses the pipe operator (|) from LangChain Expression Language.

Pipeline:
  1. extraction_chain   — Resume → Structured Profile (JSON)
  2. matching_chain     — Profile + JD → Match Analysis (JSON)
  3. scoring_chain      — Match Analysis → Score (JSON)
  4. explanation_chain  — All context → Human-readable Report
  5. full_pipeline      — Orchestrates all 4 stages end-to-end
"""

import json
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from prompts.resume_prompts import (
    skill_extraction_prompt,
    matching_prompt,
    scoring_prompt,
    explanation_prompt,
)


# ─────────────────────────────────────────────
# HELPER: Safe JSON Parser
# ─────────────────────────────────────────────

def safe_parse_json(text: str) -> dict:
    """
    Safely parse JSON from LLM output.
    Strips markdown code fences if present.
    Returns dict or error dict if parsing fails.
    """
    # Remove markdown code fences like ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        # Return structured error — never crash silently
        return {
            "parse_error": str(e),
            "raw_output": text[:500]  # first 500 chars for debugging
        }


def json_to_str(data) -> str:
    """Convert dict/JSON to formatted string for downstream prompts."""
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    return str(data)


# ─────────────────────────────────────────────
# STAGE 1: SKILL EXTRACTION CHAIN
# ─────────────────────────────────────────────

def build_extraction_chain(llm):
    """
    Input:  {"resume": str}
    Output: dict (parsed JSON with skills, experience, tools)
    """
    chain = (
        skill_extraction_prompt           # PromptTemplate → formats the prompt
        | llm                              # LLM → generates response
        | StrOutputParser()               # Extracts text from LLM response
        | RunnableLambda(safe_parse_json) # Parses JSON safely
    )
    return chain


# ─────────────────────────────────────────────
# STAGE 2: MATCHING CHAIN
# ─────────────────────────────────────────────

def build_matching_chain(llm):
    """
    Input:  {"extracted_profile": str, "job_description": str}
    Output: dict (match analysis JSON)
    """
    chain = (
        matching_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(safe_parse_json)
    )
    return chain


# ─────────────────────────────────────────────
# STAGE 3: SCORING CHAIN
# ─────────────────────────────────────────────

def build_scoring_chain(llm):
    """
    Input:  {"match_analysis": str, "job_description": str}
    Output: dict (score JSON with breakdown)
    """
    chain = (
        scoring_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(safe_parse_json)
    )
    return chain


# ─────────────────────────────────────────────
# STAGE 4: EXPLANATION CHAIN
# ─────────────────────────────────────────────

def build_explanation_chain(llm):
    """
    Input:  {"candidate_name": str, "score_result": str,
             "match_analysis": str, "extracted_profile": str}
    Output: str (human-readable evaluation report)
    """
    chain = (
        explanation_prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ─────────────────────────────────────────────
# FULL PIPELINE ORCHESTRATOR
# ─────────────────────────────────────────────

class ResumeScreeningPipeline:
    """
    End-to-end Resume Screening Pipeline.

    Orchestrates all 4 stages sequentially:
      Resume → Extract → Match → Score → Explain

    Each stage is a separate LangChain LCEL chain,
    enabling granular LangSmith tracing per step.
    """

    def __init__(self, llm):
        self.llm = llm
        # Build all chains once during initialization
        self.extraction_chain  = build_extraction_chain(llm)
        self.matching_chain    = build_matching_chain(llm)
        self.scoring_chain     = build_scoring_chain(llm)
        self.explanation_chain = build_explanation_chain(llm)

    def run(self, resume: str, job_description: str, candidate_name: str = "Candidate") -> dict:
        """
        Run the full pipeline for one candidate.

        Args:
            resume:          Full resume text
            job_description: Job description text
            candidate_name:  Optional name for logging/reporting

        Returns:
            dict with all intermediate + final results
        """

        # ── STEP 1: Skill Extraction ─────────────────
        print(f"\n  [Step 1] Extracting skills for {candidate_name}...")
        extracted_profile = self.extraction_chain.invoke({"resume": resume})

        # Convert to string for downstream prompt inputs
        extracted_profile_str = json_to_str(extracted_profile)

        # ── STEP 2: Matching ─────────────────────────
        print(f"  [Step 2] Matching profile to job description...")
        match_analysis = self.matching_chain.invoke({
            "extracted_profile": extracted_profile_str,
            "job_description": job_description,
        })

        match_analysis_str = json_to_str(match_analysis)

        # ── STEP 3: Scoring ──────────────────────────
        print(f"  [Step 3] Scoring candidate...")
        score_result = self.scoring_chain.invoke({
            "match_analysis": match_analysis_str,
            "job_description": job_description,
        })

        score_result_str = json_to_str(score_result)

        # ── STEP 4: Explanation ──────────────────────
        print(f"  [Step 4] Generating explanation report...")
        explanation = self.explanation_chain.invoke({
            "candidate_name": candidate_name,
            "score_result": score_result_str,
            "match_analysis": match_analysis_str,
            "extracted_profile": extracted_profile_str,
        })

        # ── Assemble Final Output ─────────────────────
        return {
            "candidate_name":     candidate_name,
            "extracted_profile":  extracted_profile,
            "match_analysis":     match_analysis,
            "score_result":       score_result,
            "explanation":        explanation,
            # Convenience top-level fields
            "overall_score":      score_result.get("overall_score", "N/A"),
            "score_category":     score_result.get("score_category", "N/A"),
            "recommendation":     score_result.get("hiring_recommendation", "N/A"),
        }


# ─────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "ResumeScreeningPipeline",
    "build_extraction_chain",
    "build_matching_chain",
    "build_scoring_chain",
    "build_explanation_chain",
]
