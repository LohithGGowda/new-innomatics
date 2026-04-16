"""
prompts/resume_prompts.py
--------------------------
Defines all PromptTemplates used in the Resume Screening pipeline.
Each prompt is modular, clearly instructed, and anti-hallucination guarded.

Stages:
  1. skill_extraction_prompt  — extract skills, experience, tools
  2. matching_prompt          — compare candidate vs job requirements
  3. scoring_prompt           — assign a 0-100 score
  4. explanation_prompt       — explain the score with reasoning
"""

from langchain_core.prompts import PromptTemplate

# ─────────────────────────────────────────────
# STAGE 1: SKILL EXTRACTION
# ─────────────────────────────────────────────

skill_extraction_prompt = PromptTemplate(
    input_variables=["resume"],
    template="""You are an expert HR analyst. Your task is to extract structured information from the resume below.

STRICT RULES:
- Extract ONLY what is explicitly stated in the resume.
- Do NOT assume, infer, or invent any skills, tools, or experience not mentioned.
- If a field has no data, write "Not mentioned".

Resume:
\"\"\"
{resume}
\"\"\"

Extract the following in JSON format:
{{
  "candidate_name": "<name from resume>",
  "education": "<highest degree and field>",
  "total_experience_years": <number or 0 if not clear>,
  "skills": ["<skill1>", "<skill2>", ...],
  "tools_and_technologies": ["<tool1>", "<tool2>", ...],
  "programming_languages": ["<lang1>", "<lang2>", ...],
  "ml_frameworks": ["<framework1>", ...],
  "cloud_platforms": ["<cloud1>", ...],
  "databases": ["<db1>", ...],
  "soft_skills": ["<skill1>", ...],
  "certifications": ["<cert1>", ...],
  "notable_achievements": ["<achievement1>", ...]
}}

Return ONLY valid JSON. No extra text, no markdown fences."""
)


# ─────────────────────────────────────────────
# STAGE 2: MATCHING LOGIC
# ─────────────────────────────────────────────

matching_prompt = PromptTemplate(
    input_variables=["extracted_profile", "job_description"],
    template="""You are a senior technical recruiter. Compare the candidate profile against the job description.

STRICT RULES:
- Base your analysis ONLY on the extracted profile provided.
- Do NOT assume skills that are not listed in the profile.
- Be objective and fair in your assessment.

Candidate Profile (extracted):
\"\"\"
{extracted_profile}
\"\"\"

Job Description:
\"\"\"
{job_description}
\"\"\"

Perform a detailed match analysis and return JSON:
{{
  "matched_requirements": ["<requirement that is met>", ...],
  "missing_requirements": ["<requirement that is NOT met>", ...],
  "matched_preferred_qualifications": ["<preferred qual that is met>", ...],
  "missing_preferred_qualifications": ["<preferred qual that is NOT met>", ...],
  "experience_match": {{
    "required_years": <number>,
    "candidate_years": <number>,
    "meets_requirement": <true/false>
  }},
  "education_match": {{
    "required_level": "<from JD>",
    "candidate_level": "<from profile>",
    "meets_requirement": <true/false>
  }},
  "key_strengths": ["<strength1>", ...],
  "key_gaps": ["<gap1>", ...]
}}

Return ONLY valid JSON. No extra text."""
)


# ─────────────────────────────────────────────
# STAGE 3: SCORING
# ─────────────────────────────────────────────

scoring_prompt = PromptTemplate(
    input_variables=["match_analysis", "job_description"],
    template="""You are an objective hiring system. Based on the match analysis, assign a fit score from 0 to 100.

Scoring Rubric:
- 80-100: Excellent fit — meets almost all requirements and most preferred qualifications
- 60-79:  Good fit — meets most core requirements, some gaps in preferred areas
- 40-59:  Average fit — meets some requirements, notable gaps exist
- 20-39:  Below average — meets few requirements, significant skill gaps
- 0-19:   Poor fit — does not meet the core requirements

Match Analysis:
\"\"\"
{match_analysis}
\"\"\"

Job Description Summary:
\"\"\"
{job_description}
\"\"\"

Return JSON:
{{
  "overall_score": <integer 0-100>,
  "score_breakdown": {{
    "technical_skills_score": <integer 0-40>,
    "experience_score": <integer 0-25>,
    "education_score": <integer 0-15>,
    "tools_and_frameworks_score": <integer 0-20>
  }},
  "score_category": "<Excellent/Good/Average/Below Average/Poor>",
  "hiring_recommendation": "<Strongly Recommend/Recommend/Consider/Not Recommended>"
}}

Return ONLY valid JSON. No extra text."""
)


# ─────────────────────────────────────────────
# STAGE 4: EXPLANATION (FEW-SHOT ENHANCED)
# ─────────────────────────────────────────────

explanation_prompt = PromptTemplate(
    input_variables=["candidate_name", "score_result", "match_analysis", "extracted_profile"],
    template="""You are a professional HR advisor writing a clear, structured candidate evaluation report.

RULES:
- Be factual and specific — cite only what is in the profile and match analysis.
- Be professional and constructive.
- Do NOT invent or assume any skills or experience.

Here is an EXAMPLE of a well-structured explanation (few-shot):

---EXAMPLE START---
Candidate: Jane Doe | Score: 78/100 | Category: Good Fit

SUMMARY:
Jane is a strong candidate with 4 years of relevant experience in data science. She meets most of the
core technical requirements but has limited exposure to cloud platforms and MLOps.

STRENGTHS:
1. Strong Python and Scikit-learn expertise directly matches the JD requirements.
2. 4 years of experience exceeds the 3-year minimum requirement.
3. Master's degree in Statistics aligns with educational preferences.

GAPS:
1. No experience with AWS, GCP, or Azure — a key JD requirement.
2. Limited production ML deployment experience (MLOps gap).

RECOMMENDATION: Recommend — Strong technical base; cloud skills can be developed on the job.
---EXAMPLE END---

Now write the evaluation for the following candidate:

Candidate Name: {candidate_name}

Score Result:
{score_result}

Match Analysis:
{match_analysis}

Extracted Profile:
{extracted_profile}

Write a professional evaluation report covering:
1. SUMMARY (2-3 sentences)
2. STRENGTHS (bullet points, specific and factual)
3. GAPS (bullet points, specific and factual)
4. RECOMMENDATION with brief reasoning

Keep it concise but complete. Plain text format (no JSON)."""
)


# ─────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────

__all__ = [
    "skill_extraction_prompt",
    "matching_prompt",
    "scoring_prompt",
    "explanation_prompt",
]
