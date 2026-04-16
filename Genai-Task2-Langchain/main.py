
import os
import json
from dotenv import load_dotenv

# ── Load environment variables from .env ─────
load_dotenv()

# ── Verify required env vars before doing anything ───────────────────────────
def check_environment():
    """Validate that all required environment variables are set."""
    required = {
        "GROK_API_KEY":               "Grok API Key (required for alternative LLM)",
        "LANGCHAIN_API_KEY":      "LangSmith API Key (required for tracing)",
    }
    missing = []
    for key, description in required.items():
        if not os.getenv(key):
            missing.append(f"  ❌ {key} — {description}")

    if missing:
        print("\n⚠️  Missing environment variables:")
        for m in missing:
            print(m)
        print("\n👉 Copy .env.example → .env and fill in your keys.")
        raise EnvironmentError("Required environment variables not set.")

    # Confirm LangSmith tracing is enabled
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
    project  = os.getenv("LANGCHAIN_PROJECT", "default")
    print(f"\n✅ LangSmith Tracing: {'ENABLED' if tracing == 'true' else 'DISABLED'}")
    print(f"   Project: {project}")


# ── LangChain + LangSmith imports ────────────────────────────────────────────
from langchain_openai import ChatOpenAI
from langsmith import traceable

from chains.screening_chains import ResumeScreeningPipeline
from data.sample_data import JOB_DESCRIPTION, CANDIDATES


# ─────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────

SEPARATOR = "=" * 70

def print_section(title: str):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def print_score_card(result: dict):
    """Print a formatted score card for a candidate."""
    name  = result["candidate_name"]
    score = result["overall_score"]
    cat   = result["score_category"]
    rec   = result["recommendation"]

    # Visual score bar
    bar_filled = int(score / 5) if isinstance(score, int) else 0
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    print(f"\n  Candidate  : {name}")
    print(f"  Score      : {score}/100  [{bar}]")
    print(f"  Category   : {cat}")
    print(f"  Decision   : {rec}")


def save_results(all_results: list, output_path: str = "results/screening_results.json"):
    """Save all results to a JSON file for review."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Make results JSON-serializable (handle non-serializable types)
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    serializable = make_serializable(all_results)
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")


# ─────────────────────────────────────────────
# TRACEABLE WRAPPER — LangSmith tags each run
# ─────────────────────────────────────────────

@traceable(
    name="resume_screening_run",
    tags=["resume-screening", "pipeline"],
    metadata={"version": "1.0"}
)
def screen_candidate(pipeline: ResumeScreeningPipeline,
                     resume: str,
                     job_description: str,
                     candidate_name: str,
                     candidate_type: str) -> dict:
    """
    LangSmith-traced wrapper for a single candidate screening.
    The @traceable decorator ensures this function and all nested
    LangChain calls appear as a structured trace in LangSmith.
    """
    print_section(f"Screening: {candidate_name}  [{candidate_type} Candidate]")

    result = pipeline.run(
        resume=resume,
        job_description=job_description,
        candidate_name=candidate_name,
    )

    print_score_card(result)
    print(f"\n── Explanation Report ──────────────────────────────────────────")
    print(result["explanation"])

    return result


# ─────────────────────────────────────────────
# DEBUGGING SECTION — Demonstrates LangSmith Debug
# ─────────────────────────────────────────────

@traceable(
    name="debug_incorrect_output",
    tags=["debug", "pipeline-test"],
    metadata={"purpose": "debugging_demo"}
)
def debug_pipeline_step(pipeline: ResumeScreeningPipeline, job_description: str):
    """
    Demonstrates how to debug an incorrect/unexpected output using LangSmith.
    This intentionally tests an edge case: a resume with misleading buzzwords
    but no real substance — to verify the pipeline doesn't hallucinate skills.
    """
    print_section("DEBUG: Buzzword Resume (Hallucination Guard Test)")

    # Edge case: resume with ML buzzwords but no real experience
    tricky_resume = """
    Name: Buzz Wordsmith
    Objective: Passionate about AI, Machine Learning, Big Data, and Cloud Computing.
    Interested in becoming a Data Scientist. Watched many online courses.
    Skills (self-assessed): AI, ML, Deep Learning, Python, TensorFlow (beginner)
    Education: B.A. English Literature, 2023
    Experience: None
    """

    print("\n  Testing: Resume with buzzwords but no real experience.")
    print("  Expected: Low score. Pipeline must NOT invent experience.\n")

    result = pipeline.run(
        resume=tricky_resume,
        job_description=job_description,
        candidate_name="Buzz Wordsmith (Debug Case)",
    )

    score = result.get("overall_score", "N/A")
    print(f"\n  Debug Result Score: {score}/100")

    if isinstance(score, int) and score > 30:
        print("  ⚠️  WARNING: Score may be inflated — review LangSmith trace for prompt adherence.")
    else:
        print("  ✅ PASS: Pipeline correctly scored the buzzword resume low.")

    return result


# ─────────────────────────────────────────────
# COMPARISON SUMMARY TABLE
# ─────────────────────────────────────────────

def print_comparison_table(all_results: list):
    """Print a final ranked comparison table of all candidates."""
    print_section("FINAL COMPARISON SUMMARY")

    # Sort by score descending
    sorted_results = sorted(
        all_results,
        key=lambda r: r.get("overall_score", 0) if isinstance(r.get("overall_score"), int) else 0,
        reverse=True,
    )

    print(f"\n  {'Rank':<5} {'Candidate':<20} {'Score':>6} {'Category':<20} {'Decision'}")
    print(f"  {'-'*4} {'-'*19} {'-'*6} {'-'*20} {'-'*25}")

    for rank, r in enumerate(sorted_results, 1):
        name  = r.get("candidate_name", "Unknown")[:19]
        score = r.get("overall_score", "N/A")
        cat   = r.get("score_category", "N/A")[:19]
        rec   = r.get("recommendation", "N/A")
        print(f"  {rank:<5} {name:<20} {str(score):>6} {cat:<20} {rec}")

    print(f"\n{'─'*70}")
    print("  📊 Full traces available in LangSmith dashboard")
    print(f"     Project: {os.getenv('LANGCHAIN_PROJECT', 'ai-resume-screening')}")
    print(f"     URL:     https://smith.langchain.com")
    print(f"{'─'*70}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("   🤖  AI RESUME SCREENING SYSTEM  |  Powered by LangChain + LangSmith")
    print("=" * 70)

    # 1. Validate environment
    check_environment()

    # 2. Initialize LLM
    print("\n🔧 Initializing LLM (GPT-4o-mini)...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",       # cost-effective, highly capable model
        temperature=0,             # deterministic outputs for consistency
        max_tokens=2000,
    )

    # 3. Build pipeline
    print("🔧 Building screening pipeline...")
    pipeline = ResumeScreeningPipeline(llm=llm)

    # 4. Screen all candidates (3 runs for LangSmith)
    all_results = []

    for candidate in CANDIDATES:
        result = screen_candidate(
            pipeline=pipeline,
            resume=candidate["resume"],
            job_description=JOB_DESCRIPTION,
            candidate_name=candidate["name"],
            candidate_type=candidate["type"],
        )
        all_results.append(result)

    # 5. Debug run (mandatory LangSmith debug trace)
    debug_result = debug_pipeline_step(pipeline, JOB_DESCRIPTION)
    # Not added to main results to keep comparison clean

    # 6. Print comparison table
    print_comparison_table(all_results)

    # 7. Save all results
    save_results(all_results)


if __name__ == "__main__":
    main()
