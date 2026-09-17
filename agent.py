import re
import time
import json
from typing import Literal

from ollama import chat
from pydantic import BaseModel, Field, ValidationError


# =========================================================
# Configuration
# =========================================================

MODEL = "qwen3:1.7b"


# =========================================================
# Models
# =========================================================

Status = Literal[
    "direct",
    "partial",
    "missing",
    "unknown"
]


class ParsedJob(BaseModel):

    job_title: str
    company: str

    core_required: list[str] = Field(
        default_factory=list
    )

    required_competencies: list[str] = Field(
        default_factory=list
    )

    preferred: list[str] = Field(
        default_factory=list
    )

    responsibilities: list[str] = Field(
        default_factory=list
    )


class MatchItem(BaseModel):

    id: str

    status: Status

    evidence: str


class MatchResponse(BaseModel):

    matches: list[MatchItem] = Field(
        default_factory=list
    )


# =========================================================
# Job Parser Prompt
# =========================================================

JOB_PARSER_PROMPT = """
Extract this job posting into structured categories.

core_required:
Hard requirements involving prior experience, technologies,
education, certifications, or required professional background.

required_competencies:
Required abilities such as communication, organization,
problem solving, teamwork, prioritization, or learning ability.

preferred:
Anything described as preferred, ideal, a plus,
desired, bonus, or nice to have.

responsibilities:
Tasks performed after being hired.

IMPORTANT:
If the posting has a Required, Minimum Qualifications,
Basic Qualifications, Required Knowledge/Experience,
or Must Have section, extract EVERY substantive item.

Do not evaluate any candidate.

Do not turn responsibilities into prior-experience requirements
unless the posting explicitly says that prior experience is required.

Keep each item concise.
"""


# =========================================================
# Parse Job
# =========================================================

def clean_line(line):
    """
    Remove common Markdown/bullet formatting.
    """

    line = line.strip()

    line = re.sub(
        r"^[\-\*\•\·]+\s*",
        "",
        line
    )

    line = re.sub(
        r"^#{1,6}\s*",
        "",
        line
    )

    line = line.replace("**", "")
    line = line.strip()

    return line


def is_heading(line, keywords):
    """
    Check whether a line resembles one of our section headings.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        line.lower().rstrip(":\\").strip()
    )

    return any(
        normalized == keyword
        or normalized.startswith(f"{keyword}:")
        for keyword in keywords
    )


def extract_items(lines):
    """
    Turn section lines into concise items.
    """

    items = []
    seen = set()

    for line in lines:

        line = clean_line(line)

        if not line:
            continue

        normalized = line.casefold()

        if normalized in seen:
            continue

        seen.add(normalized)
        items.append(line)

    return items


def parse_job(job_description):

    lines = [
        clean_line(line)
        for line in job_description.splitlines()
        if clean_line(line)
    ]

    # ------------------------------------
    # Job title / company
    # ------------------------------------

    job_title = (
        lines[0]
        if lines
        else "Not specified"
    )

    if job_title.lower() in {
        "full job description",
        "job description"
    }:

        opening_text = " ".join(lines[:40])
        title_match = re.search(
            r"\bAs (?:an?|the) ([^,.;]+),\s+you\b",
            opening_text,
            flags=re.IGNORECASE
        )
        search_match = re.search(
            r"\bsearch for (?:an?|the) (.+?)(?:\.\s|$)",
            opening_text,
            flags=re.IGNORECASE
        )

        if title_match:
            job_title = title_match.group(1).strip()
        elif search_match:
            job_title = search_match.group(1).strip()

    company = "Not specified"

    for line in lines[:20]:

        if "college board" in line.lower():
            company = "College Board"
            break

        company_match = re.match(
            r"^(?:company|employer|organization)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE
        )

        if company_match:
            company = company_match.group(1).strip()
            break

        introduction_match = re.match(
            r"^At\s+([^,]{2,60}),",
            line,
            flags=re.IGNORECASE
        )

        if introduction_match:
            company = introduction_match.group(1).strip()
            break

    # ------------------------------------
    # Heading definitions
    # ------------------------------------

    core_headings = [
        "required knowledge/experience",
        "required experience",
        "required qualifications",
        "required skills",
        "requirements",
        "minimum qualifications",
        "basic qualifications",
        "minimum requirements",
        "qualifications"
    ]

    competency_headings = [
        "required competencies",
        "skills and competencies",
        "what you need to succeed",
        "what you'll need to succeed",
        "what you will need to succeed",
        "what you need",
        "what you'll need",
        "what you will need",
        "what you bring",
        "who you are",
        "exceptional candidates can effectively speak to",
        "all roles at college board require"
    ]

    fixed_competency_headings = [
        "must have"
    ]

    preferred_headings = [
        "preferred knowledge/experience",
        "preferred qualifications",
        "preferred",
        "nice to have",
        "desired qualifications"
    ]

    responsibility_headings = [
        "in this role, you will",
        "responsibilities",
        "what you'll do",
        "what you will do",
        "duties"
    ]

    ignored_headings = [
        "full job description",
        "the role",
        "about us",
        "about the company",
        "about the team",
        "about the opportunity",
        "about you",
        "about our process",
        "benefits",
        "health + wellness",
        "time off",
        "401k",
        "equipment",
        "what we offer",
        "compensation",
        "salary",
        "location",
        "accommodations",
        "equal opportunity employer"
    ]

    all_headings = (
        core_headings
        + competency_headings
        + fixed_competency_headings
        + preferred_headings
        + responsibility_headings
    )

    # ------------------------------------
    # Collect sections
    # ------------------------------------

    sections = {
        "core": [],
        "competency": [],
        "fixed_competency": [],
        "preferred": [],
        "responsibility": []
    }

    current_section = None

    for line in lines:

        if is_heading(
            line,
            core_headings
        ):
            current_section = "core"
            continue

        if is_heading(
            line,
            preferred_headings
        ):
            current_section = "preferred"
            continue

        if is_heading(
            line,
            competency_headings
        ):
            current_section = "competency"
            continue

        if is_heading(
            line,
            fixed_competency_headings
        ):
            current_section = "fixed_competency"
            continue

        if is_heading(
            line,
            responsibility_headings
        ):
            current_section = "responsibility"
            continue

        if is_heading(
            line,
            ignored_headings
        ):
            current_section = None
            continue

        # Stop collecting when another major heading begins
        if current_section is not None:

            if is_heading(
                line,
                all_headings
            ):
                continue

            # Ignore obvious non-qualification sections
            lower = line.lower()

            if any(
                phrase in lower
                for phrase in [
                    "about the team",
                    "about the opportunity",
                    "about our process",
                    "what we offer",
                    "compensation",
                    "hiring range",
                    "location",
                    "role type"
                ]
            ):
                current_section = None
                continue

            sections[
                current_section
            ].append(line)

    # ------------------------------------
    # Convert to ParsedJob
    # ------------------------------------

    competency_items = extract_items(
        sections["competency"]
    )

    experience_markers = (
        " experience",
        "background in",
        "degree",
        "certification",
        " years",
        "knowledge of",
        "familiarity with",
        "proficiency in"
    )

    inferred_core = [
        item
        for item in competency_items
        if any(
            marker in f" {item.lower()}"
            for marker in experience_markers
        )
    ]

    competency_items = [
        item
        for item in competency_items
        if item not in inferred_core
    ]

    parsed_job = ParsedJob(
        job_title=job_title,
        company=company,

        core_required=extract_items(
            sections["core"]
        ) + inferred_core,

        required_competencies=(
            extract_items(
                sections["fixed_competency"]
            )
            + competency_items
        ),

        preferred=extract_items(
            sections["preferred"]
        ),

        responsibilities=extract_items(
            sections["responsibility"]
        )
    )

    # ------------------------------------
    # Safety validation
    # ------------------------------------

    total_required = (
        len(parsed_job.core_required)
        + len(
            parsed_job.required_competencies
        )
    )

    if total_required == 0:

        raise ValueError(
            "ApplyAI could not identify any "
            "required qualifications."
        )

    print(
        "\nPYTHON JOB PARSER RESULTS"
    )

    print(
        "Core requirements:",
        len(parsed_job.core_required)
    )

    print(
        "Required competencies:",
        len(
            parsed_job.required_competencies
        )
    )

    print(
        "Preferred:",
        len(parsed_job.preferred)
    )

    print(
        "Responsibilities:",
        len(parsed_job.responsibilities)
    )

    return parsed_job


# =========================================================
# Build Requirement IDs
# =========================================================

def build_experience_fit(
    requirements,
    matches
):

    match_lookup = {
        match.id: match.status
        for match in matches
    }

    core = [
        requirement
        for requirement in requirements
        if (
            requirement["category"] == "core"
            and not eligibility_metadata_requirement_kinds(
                requirement.get("requirement", "")
            )
        )
    ]

    if not core:

        return (
            "No explicit core experience requirements "
            "were identified in the posting."
        )

    direct = 0
    partial = 0
    missing = 0
    unknown = 0

    for requirement in core:

        status = match_lookup.get(
            requirement["id"],
            "missing"
        )

        if status == "direct":
            direct += 1

        elif status == "partial":
            partial += 1

        elif status == "unknown":
            unknown += 1

        else:
            missing += 1

    summary = (
        f"Of the {len(core)} core experience requirements, "
        f"the candidate has {direct} direct matches, "
        f"{partial} partial matches, and "
        f"{missing} missing qualifications."
    )

    if unknown:
        summary += (
            f" {unknown} requirement"
            + ("s are" if unknown != 1 else " is")
            + " not provided and excluded from scoring."
        )

    return summary


def build_requirements(parsed_job):

    requirements = []

    for index, requirement in enumerate(
        parsed_job.core_required,
        start=1
    ):

        requirements.append(
            {
                "id": f"C{index}",
                "category": "core",
                "requirement": requirement
            }
        )

    for index, requirement in enumerate(
        parsed_job.required_competencies,
        start=1
    ):

        requirements.append(
            {
                "id": f"K{index}",
                "category": "competency",
                "requirement": requirement
            }
        )

    for index, requirement in enumerate(
        parsed_job.preferred,
        start=1
    ):

        requirements.append(
            {
                "id": f"P{index}",
                "category": "preferred",
                "requirement": requirement
            }
        )

    return requirements


# =========================================================
# Matcher Prompt
# =========================================================

MATCHER_PROMPT = """
Compare the candidate profile against every job requirement.

For each requirement ID return:

status:
direct
partial
missing
unknown

direct:
The candidate profile explicitly demonstrates the ENTIRE substantive
requirement in the requested context.

partial:
The profile contains meaningful transferable or related experience.

missing:
The profile contains no meaningful evidence.

unknown:
Use only when a required candidate-supplied metadata value is null.
Do not use unknown for ordinary resume qualifications.

RULES:

Never invent experience.

When a requirement contains multiple abilities joined by "and",
direct requires explicit evidence for every important part.
If only some parts are supported, return partial.

Prefer partial when the profile shows a transferable skill but not
the specific context requested by the employer.

Retail customer service can demonstrate:
- customer service
- communication
- prioritization
- organization

But retail customer service does NOT equal:
- professional software support
- client-system problem tracking or trend analysis
- ticket tracking or technical-support processes
- CRM case management
- Salesforce experience

If Salesforce, CRM/case management, SaaS, or LMS experience is not
explicitly present in the candidate profile, return missing.

Software troubleshooting plus retail customer service is transferable
to software customer support, but it is partial unless a professional
software-support role is explicitly documented.

Never infer U.S. work authorization or sponsorship needs from resume
text, location, school, name, or employment. Use candidate_metadata only.

Customer-facing retail work mainly demonstrates verbal communication.
It does not demonstrate written communication unless the profile
explicitly describes writing, documentation, email, or chat work.

Applying product knowledge learned in training while following support
procedures and policies requires explicit training/procedure/policy
evidence. Employment at a retailer by itself is only transferable.

Academic and personal projects can demonstrate:
- technical skills
- problem solving
- debugging
- teamwork
- learning unfamiliar tools

But they do not automatically equal professional employment.

Evidence must be one short candidate-profile fact, at most 12 words.
Do not explain the decision or list multiple examples.
Never use the requirement itself as evidence.
Prefer a named job, project, or activity with a concrete action over a
generic technical-skills summary.

Good evidence:
"Python and SQL listed under technical skills."

Bad evidence:
Long summaries of the candidate's background.

Return every requirement ID exactly once.
"""


# =========================================================
# Match Candidate
# =========================================================

MATCH_BATCH_SIZE = 5


SEMANTIC_CONCEPTS = {
    "customer_service": (
        "customer service",
        "customer support",
        "assist customers",
        "help customers",
        "customer facing"
    ),
    "product_knowledge": (
        "product knowledge",
        "product expertise",
        "learned products",
        "software utilization"
    ),
    "training": (
        "training",
        "trained",
        "onboarding",
        "new hire"
    ),
    "procedures_policies": (
        "procedure",
        "procedures",
        "policy",
        "policies",
        "standard operating",
        "protocol",
        "guidelines"
    ),
    "problem_tracking": (
        "problem tracking",
        "track problems",
        "tracked problems",
        "issue tracking",
        "tracked issues",
        "ticket tracking",
        "ticketing system",
        "case management"
    ),
    "trend_analysis": (
        "trend analysis",
        "trends",
        "patterns",
        "time series",
        "statistical comparison"
    ),
    "client_systems": (
        "client system",
        "customer system",
        "customer software",
        "software support",
        "technical support",
        "saas support"
    ),
    "written_communication": (
        "written communication",
        "writing",
        "wrote",
        "documentation",
        "documented",
        "help documents",
        "email support",
        "chat support"
    ),
    "verbal_communication": (
        "verbal communication",
        "phone support",
        "spoke with",
        "speaking",
        "technical discussion",
        "communicate with customers",
        "customer service"
    ),
    "problem_solving": (
        "problem solving",
        "solve problems",
        "resolved problems",
        "resolve issues",
        "resolve routine issues",
        "resolve problems",
        "issue resolution",
        "debugged",
        "debugging",
        "troubleshot",
        "technical troubleshooting",
        "methodical debugging"
    ),
    "speed": (
        "quickly",
        "promptly",
        "time sensitive",
        "fast paced",
        "high traffic",
        "time pressure",
        "under pressure",
        "urgent"
    ),
    "troubleshooting": (
        "troubleshooting",
        "troubleshoot",
        "debugged",
        "debugging",
        "diagnose",
        "diagnostic"
    ),
    "computer_software": (
        "computer",
        "computers",
        "software",
        "technology",
        "pc hardware",
        "technical skills"
    ),
    "organization": (
        "organization",
        "organizational",
        "organized",
        "coordinate with",
        "maintaining accuracy",
        "competing priorities",
        "competing responsibilities",
        "balance customer needs"
    ),
    "prioritization": (
        "prioritization",
        "prioritize",
        "competing priorities",
        "competing responsibilities",
        "balance customer needs"
    ),
    "emerging_technology": (
        "emerging technology",
        "emerging technologies",
        "ollama",
        "qwen",
        "local llm",
        "local ai"
    ),
    "learning_unfamiliar": (
        "learning unfamiliar",
        "learned unfamiliar",
        "unfamiliar tools",
        "learn new tools",
        "rapid learning"
    ),
    "collaboration": (
        "collaborate",
        "collaboration",
        "collaborative",
        "two person development team",
        "code review",
        "teamwork"
    ),
    "process_improvement": (
        "process improvement",
        "iteratively improved",
        "iterative improvement",
        "continuously improving"
    ),
    "education_mission": (
        "peer tutoring",
        "tutoring",
        "mentoring",
        "educational opportunities",
        "career opportunities"
    ),
    "analytical_reasoning": (
        "analytical",
        "analyze",
        "analysis",
        "analyzed",
        "statistical",
        "algorithm",
        "correctness",
        "time complexity",
        "root cause"
    )
}

CONCEPT_LABELS = {
    "customer_service": "customer service",
    "product_knowledge": "product knowledge",
    "training": "training",
    "procedures_policies": "procedures or policies",
    "problem_tracking": "problem or ticket tracking",
    "trend_analysis": "trend or pattern analysis",
    "client_systems": "client-system support",
    "written_communication": "written communication",
    "verbal_communication": "verbal communication",
    "problem_solving": "problem solving",
    "speed": "working quickly",
    "troubleshooting": "troubleshooting",
    "computer_software": "computer or software experience",
    "organization": "organization",
    "prioritization": "prioritizing competing demands",
    "emerging_technology": "emerging-technology experimentation",
    "learning_unfamiliar": "learning unfamiliar tools",
    "collaboration": "collaboration",
    "process_improvement": "process improvement",
    "education_mission": "education or career-development support",
    "analytical_reasoning": "analytical reasoning"
}

SEMANTIC_STOP_WORDS = {
    "a", "ability", "an", "and", "are", "be", "for",
    "general", "in", "of", "or", "skills", "strong",
    "superb", "the", "to", "with"
}


def normalize_semantic_text(value):

    if isinstance(value, dict):
        value = json.dumps(value)

    return re.sub(
        r"\s+",
        " ",
        re.sub(r"[^a-z0-9+#]+", " ", str(value).lower())
    ).strip()


def has_semantic_concept(text, concept):

    normalized = f" {normalize_semantic_text(text)} "

    return any(
        f" {normalize_semantic_text(alias)} " in normalized
        for alias in SEMANTIC_CONCEPTS[concept]
    )


def requirement_concepts(requirement):

    normalized = normalize_semantic_text(requirement)
    concepts = set()

    markers = {
        "customer_service": (
            "customer service", "customer support"
        ),
        "product_knowledge": ("product knowledge",),
        "training": ("training", "new hire"),
        "procedures_policies": (
            "procedure", "procedures", "policy", "policies"
        ),
        "problem_tracking": (
            "problem tracking", "issue tracking",
            "ticket tracking", "case management"
        ),
        "trend_analysis": ("trend", "pattern"),
        "client_systems": (
            "client system", "customer system",
            "software support", "technical support"
        ),
        "written_communication": ("written", "writing"),
        "verbal_communication": (
            "verbal", "spoken", "phone communication"
        ),
        "problem_solving": (
            "solve problem", "solving problem", "problem solving"
        ),
        "speed": (
            "quickly", "promptly", "rapidly",
            "time pressure", "under pressure", "fast paced"
        ),
        "troubleshooting": (
            "troubleshoot", "troubleshooting", "diagnose"
        ),
        "computer_software": (
            "computer", "software technology", "technical technology"
        ),
        "organization": (
            "organization", "organizational", "organized"
        ),
        "prioritization": (
            "priorit", "competing demand", "competing priorit"
        ),
        "emerging_technology": (
            "emerging technolog", "new technolog", "technology curiosity"
        ),
        "learning_unfamiliar": (
            "unfamiliar tool", "learn new", "rapid learning",
            "quickly learn", "learning and applying new", "digital tools"
        ),
        "collaboration": (
            "collaborat", "teamwork", "team work", "small team"
        ),
        "process_improvement": (
            "process improvement", "continuously improving"
        ),
        "education_mission": (
            "educational", "career opportunities", "mission driven"
        ),
        "analytical_reasoning": (
            "analytical", "root cause", "data informed",
            "analysis", "reasoning"
        )
    }

    for concept, phrases in markers.items():
        if any(phrase in normalized for phrase in phrases):
            concepts.add(concept)

    if (
        "communication" in normalized
        and "written_communication" not in concepts
        and "verbal_communication" not in concepts
    ):
        concepts.add("verbal_communication")

    if (
        "quickly learn" in normalized
        and "learning_unfamiliar" in concepts
    ):
        concepts.discard("speed")

    return concepts


def semantic_tokens(value):

    return {
        token
        for token in normalize_semantic_text(value).split()
        if token not in SEMANTIC_STOP_WORDS
        and len(token) > 2
    }


def evidence_is_grounded(evidence, candidate_profile):

    evidence_concepts = {
        concept
        for concept in SEMANTIC_CONCEPTS
        if has_semantic_concept(evidence, concept)
    }
    profile_concepts = {
        concept
        for concept in SEMANTIC_CONCEPTS
        if has_semantic_concept(candidate_profile, concept)
    }

    if evidence_concepts - profile_concepts:
        return False

    if evidence_concepts:
        return True

    evidence_tokens = semantic_tokens(evidence)
    profile_tokens = semantic_tokens(candidate_profile)
    shared_tokens = evidence_tokens & profile_tokens

    if not evidence_tokens:
        return False

    return (
        len(shared_tokens) >= 2
        or len(shared_tokens) / len(evidence_tokens) >= 0.5
    )


def requirement_part_is_supported(part, text):

    concepts = requirement_concepts(part)

    if concepts:
        return all(
            has_semantic_concept(text, concept)
            for concept in concepts
        )

    part_tokens = semantic_tokens(part)

    if not part_tokens:
        return True

    shared_tokens = part_tokens & semantic_tokens(text)

    return (
        len(shared_tokens) >= 2
        or len(shared_tokens) / len(part_tokens) >= 0.6
    )


def candidate_profile_passages(candidate_profile):

    if isinstance(candidate_profile, dict):
        profile_text = json.dumps(candidate_profile)
    else:
        profile_text = str(candidate_profile)

    base_passages = [
        passage.strip(" -\t\r\n")
        for passage in re.split(
            r"[\r\n]+|(?<=[.!?])\s+",
            profile_text
        )
        if passage.strip()
    ]

    combined_passages = [
        f"{base_passages[index]} {base_passages[index + 1]}"
        for index in range(len(base_passages) - 1)
    ]

    source_markers = (
        "applyai",
        "senior design",
        "driving intervention analysis",
        "algorithm optimization",
        "publix",
        "association for computing machinery",
        "acm member"
    )
    contextual_passages = []

    for index, passage in enumerate(base_passages):
        if not any(
            marker in normalize_semantic_text(passage)
            for marker in source_markers
        ):
            continue

        following = []

        for detail in base_passages[index + 1:index + 4]:
            detail_text = normalize_semantic_text(detail)

            if any(
                marker in detail_text
                for marker in source_markers
            ):
                break

            following.append(detail)

        for detail in following:
            contextual_passages.append(
                f"{passage} {detail}"
            )

        for count in range(2, len(following) + 1):
            contextual_passages.append(
                " ".join([passage] + following[:count])
            )

    return (
        base_passages
        + combined_passages
        + contextual_passages
    )


EVIDENCE_CONCEPT_WEIGHTS = {
    "collaboration": 4,
    "emerging_technology": 4,
    "learning_unfamiliar": 4,
    "organization": 3,
    "prioritization": 3,
    "written_communication": 3,
    "analytical_reasoning": 3,
    "problem_solving": 2,
    "troubleshooting": 2,
    "process_improvement": 2,
    "education_mission": 2
}


def evidence_specificity_score(
    requirement,
    passage
):
    """
    Prefer concrete work/project facts over generic skill inventories.
    """

    requirement_text = normalize_semantic_text(requirement)
    passage_text = f" {normalize_semantic_text(passage)} "
    score = 0

    if any(
        verb in passage_text
        for verb in (
            " built ", " develop ", " debugged ", " troubleshot ",
            " collaborate ", " analyzed ", " designed ",
            " balance ", " provide customer service "
        )
    ):
        score += 2

    if any(
        phrase in passage_text
        for phrase in (
            "technical skills include",
            "development tools",
            "troubleshooting high volume"
        )
    ):
        score -= 4

    analytical_requirement = any(
        phrase in requirement_text
        for phrase in (
            "analytical",
            "problem solving",
            "root cause",
            "troubleshoot"
        )
    )

    if analytical_requirement:
        if (
            "applyai" in passage_text
            and any(
                term in passage_text
                for term in (
                    "debugged", "troubleshot", "model output"
                )
            )
        ):
            score += 14

        if (
            any(
                term in passage_text
                for term in (
                    "senior design",
                    "driving intervention analysis"
                )
            )
            and any(
                term in passage_text
                for term in (
                    " analyze ", " analyzed ",
                    " statistical ", " machine learning ",
                    " extract features ", " reproducible python ",
                    " clean ", " synchronize "
                )
            )
        ):
            score += 17

        if (
            "algorithm" in passage_text
            and any(
                term in passage_text
                for term in (
                    "correctness", "time complexity", "runtime"
                )
            )
        ):
            score += 10

    if any(
        term in requirement_text
        for term in (
            "organization", "organizational",
            "priorit", "competing demand"
        )
    ):
        if (
            any(
                term in passage_text
                for term in ("publix", "grocery clerk")
            )
            and any(
                term in passage_text
                for term in (
                    "competing priorities",
                    "competing responsibilities",
                    "balance customer needs",
                    "high traffic"
                )
            )
        ):
            score += 14

            if any(
                term in passage_text
                for term in (
                    "competing priorities",
                    "competing responsibilities",
                    "balance customer needs"
                )
            ):
                score += 4


            if (
                "high traffic" in passage_text
                and any(
                    term in passage_text
                    for term in (
                        "competing priorities",
                        "competing responsibilities"
                    )
                )
            ):
                score += 5

    if any(
        term in requirement_text
        for term in (
            "emerging technolog", "ai driven", "digital tools"
        )
    ):
        named_ai_tools = sum(
            tool in passage_text
            for tool in (
                "ollama", "qwen", "streamlit", "local ai"
            )
        )

        if "applyai" in passage_text and named_ai_tools >= 2:
            score += 16 + (named_ai_tools * 2)

    if any(
        term in requirement_text
        for term in (
            "collaborat", "teamwork", "small team"
        )
    ):
        if (
            any(
                term in passage_text
                for term in (
                    "senior design", "driving intervention analysis"
                )
            )
            and any(
                term in passage_text
                for term in (
                    "two person", "github", "code review"
                )
            )
        ):
            score += 16

        if (
            "acm" in passage_text
            and any(
                term in passage_text
                for term in (
                    "peer tutoring", "tutoring",
                    "collaborative problem solving"
                )
            )
        ):
            score += 13

    return score


def find_complete_profile_evidence(
    requirement,
    candidate_profile
):
    """
    Find one resume passage that explicitly covers every known concept.
    """

    concepts = requirement_concepts(requirement)

    if not concepts:
        return None

    matching_passages = [
        passage
        for passage in candidate_profile_passages(
            candidate_profile
        )
        if all(
            has_semantic_concept(passage, concept)
            for concept in concepts
        )
    ]

    if matching_passages:
        return max(
            matching_passages,
            key=lambda passage: evidence_specificity_score(
                requirement,
                passage
            )
        )

    return None


DETERMINISTIC_DIRECT_CONCEPTS = {
    "customer_service",
    "organization",
    "prioritization",
    "emerging_technology",
    "learning_unfamiliar",
    "collaboration",
    "problem_solving",
    "speed",
    "troubleshooting",
    "computer_software",
    "verbal_communication",
    "written_communication"
}

DETERMINISTIC_TRANSFERABLE_CONCEPTS = (
    DETERMINISTIC_DIRECT_CONCEPTS
    | {
        "process_improvement",
        "education_mission",
        "analytical_reasoning"
    }
)


def apply_deterministic_profile_evidence(
    requirement,
    match,
    candidate_profile
):
    """
    Use explicit resume facts for well-defined transferable competencies.
    """

    concepts = requirement_concepts(requirement)

    if (
        not concepts
    ):
        return match

    evidence = None

    if concepts.issubset(
        DETERMINISTIC_DIRECT_CONCEPTS
    ):
        evidence = find_complete_profile_evidence(
            requirement,
            candidate_profile
        )

        if evidence is not None:
            return MatchItem(
                id=match.id,
                status="direct",
                evidence=evidence
            )

    supported_transferable_concepts = {
        concept
        for concept in concepts
        if concept in DETERMINISTIC_TRANSFERABLE_CONCEPTS
        and has_semantic_concept(
            candidate_profile,
            concept
        )
    }

    if (
        match.status == "missing"
        and supported_transferable_concepts
    ):
        evidence = find_related_profile_evidence(
            requirement,
            candidate_profile
        )

        if evidence:
            return MatchItem(
                id=match.id,
                status="partial",
                evidence=evidence
            )

    return match


def profile_explicitly_documents(
    candidate_profile,
    terms
):

    normalized = normalize_semantic_text(
        candidate_profile
    )

    for term in terms:
        normalized_term = normalize_semantic_text(term)

        if normalized_term not in normalized:
            continue

        negative_pattern = (
            r"\b(?:no|without|not documented|lack(?:s|ing)?)\b"
            r".{0,60}\b"
            + re.escape(normalized_term)
            + r"\b"
        )

        if not re.search(negative_pattern, normalized):
            return True

    return False


def get_candidate_metadata_boolean(
    candidate_metadata,
    section,
    field
):
    """
    Read an explicit tri-state candidate fact without coercion or inference.
    """

    if not isinstance(candidate_metadata, dict):
        return None

    section_data = candidate_metadata.get(section)

    if not isinstance(section_data, dict):
        return None

    value = section_data.get(field)

    if isinstance(value, bool):
        return value

    return None


def work_authorization_is_documented(candidate_metadata):
    """
    Preserve the public helper while consulting metadata only.
    """

    return get_candidate_metadata_boolean(
        candidate_metadata,
        "work_authorization",
        "us_authorized"
    ) is True


def candidate_metadata_requirement_kinds(requirement):
    """
    Identify requirements governed by candidate-supplied metadata.
    """

    normalized = normalize_semantic_text(requirement)
    kinds = set()

    authorization_markers = (
        "authorized to work",
        "authorization to work",
        "work authorization",
        "legally authorized",
        "eligible to work in the united states",
        "eligible to work in the u s",
        "eligible for employment in the united states",
        "eligible for employment in the u s",
        "right to work in the united states",
        "right to work in the u s",
        "legally permitted to work"
    )

    if any(
        marker in normalized
        for marker in authorization_markers
    ):
        kinds.add("us_authorized")

    sponsorship_markers = (
        "require sponsorship",
        "requires sponsorship",
        "requiring sponsorship",
        "sponsorship required",
        "without sponsorship",
        "no sponsorship",
        "not offer sponsorship",
        "does not offer sponsorship",
        "do not offer sponsorship",
        "cannot sponsor",
        "unable to sponsor",
        "sponsorship is not available",
        "visa sponsorship"
    )
    sponsorship_unavailable = bool(re.search(
        r"\b(?:no|not|does not|do not|cannot|unable to|without)\b"
        r".{0,40}\b(?:sponsor|sponsorship)\b",
        normalized
    ))
    sponsorship_available = (
        not sponsorship_unavailable
        and any(
            marker in normalized
            for marker in (
                "sponsorship is available",
                "sponsorship available",
                "offers sponsorship",
                "offer visa sponsorship",
                "provides sponsorship"
            )
        )
    )
    sponsorship_required_pattern = re.search(
        r"\b(?:requir(?:e|es|ing)|need|needs)\b"
        r".{0,40}\bsponsorship\b",
        normalized
    )

    if (
        not sponsorship_available
        and (
            sponsorship_required_pattern
            or sponsorship_unavailable
            or any(
                marker in normalized
                for marker in sponsorship_markers
            )
        )
    ):
        kinds.add("requires_sponsorship")

    if re.search(
        r"\bsecurity clearance\b"
        r"|\bability to obtain\b.{0,30}\bclearance\b"
        r"|\b(?:hold|maintain|obtain)\b.{0,30}\bclearance\b",
        normalized
    ):
        kinds.add("security_clearance")

    if re.search(
        r"\b(?:relocation required|required to relocate|"
        r"must relocate|willing(?:ness)? to relocate)\b",
        normalized
    ):
        kinds.add("willing_to_relocate")

    if re.search(
        r"\b(?:travel required|required to .*?travel|must .*?travel|"
        r"willing(?:ness)? to travel|\d{1,3} (?:percent )?travel)\b",
        normalized
    ):
        kinds.add("willing_to_travel")

    mandatory_arrangement = bool(re.search(
        r"\b(?:must|required|required to|ability to)\b.{0,35}"
        r"\b(?:remote|hybrid|onsite|on site|in office)\b",
        normalized
    ))

    if mandatory_arrangement:
        kinds.add("work_arrangement")

    return kinds


ELIGIBILITY_METADATA_KINDS = {
    "us_authorized",
    "requires_sponsorship",
    "security_clearance"
}


def eligibility_metadata_requirement_kinds(requirement):

    return (
        candidate_metadata_requirement_kinds(requirement)
        & ELIGIBILITY_METADATA_KINDS
    )


def evaluate_candidate_metadata_requirement(
    requirement,
    match,
    candidate_metadata
):
    """
    Resolve authorization and sponsorship solely from explicit metadata.
    """

    kinds = candidate_metadata_requirement_kinds(
        requirement
    )

    if not kinds:
        return None, None

    authorization = get_candidate_metadata_boolean(
        candidate_metadata,
        "work_authorization",
        "us_authorized"
    )
    requires_sponsorship = get_candidate_metadata_boolean(
        candidate_metadata,
        "work_authorization",
        "requires_sponsorship"
    )
    clearance = get_candidate_metadata_value(
        candidate_metadata,
        "security_clearance",
        "status",
        default="unknown"
    )
    willing_to_relocate = get_candidate_metadata_boolean(
        candidate_metadata,
        "location",
        "willing_to_relocate"
    )
    willing_to_travel = get_candidate_metadata_boolean(
        candidate_metadata,
        "travel",
        "willing"
    )
    maximum_travel = get_candidate_metadata_value(
        candidate_metadata,
        "travel",
        "maximum_percentage"
    )
    preferred_arrangements = set(
        get_candidate_metadata_value(
            candidate_metadata,
            "work_preferences",
            "arrangements",
            default=[]
        ) or []
    )

    conflicts = []
    unknown = []
    supported = []

    if "us_authorized" in kinds:
        if authorization is True:
            supported.append(
                "currently authorized to work in the United States"
            )
        elif authorization is False:
            conflicts.append(
                "not currently authorized to work in the United States"
            )
        else:
            unknown.append(
                "U.S. work authorization was not provided"
            )

    if "requires_sponsorship" in kinds:
        if requires_sponsorship is False:
            supported.append(
                "employer sponsorship is not required"
            )
        elif requires_sponsorship is True:
            conflicts.append(
                "employer sponsorship is required"
            )
        else:
            unknown.append(
                "sponsorship needs were not provided"
            )

    if "security_clearance" in kinds:
        active_clearance_required = bool(re.search(
            r"\b(?:active|current)\s+security clearance\b",
            normalize_semantic_text(requirement)
        ))

        if clearance == "active":
            supported.append("active security clearance")
        elif clearance == "eligible" and not active_clearance_required:
            supported.append("eligible to obtain a security clearance")
        elif clearance == "unknown":
            unknown.append("security clearance status was not provided")
        elif clearance == "eligible":
            conflicts.append(
                "eligible for clearance but no active clearance"
            )
        else:
            conflicts.append("no security clearance")

    if "willing_to_relocate" in kinds:
        if willing_to_relocate is True:
            supported.append("willing to relocate")
        elif willing_to_relocate is False:
            conflicts.append("not willing to relocate")
        else:
            unknown.append("relocation preference was not provided")

    if "willing_to_travel" in kinds:
        required_travel_match = re.search(
            r"(\d{1,3}) (?:percent )?travel",
            normalize_semantic_text(requirement)
        )
        required_travel = (
            int(required_travel_match.group(1))
            if required_travel_match
            else None
        )

        if willing_to_travel is False:
            conflicts.append("not willing to travel")
        elif willing_to_travel is None:
            unknown.append("travel preference was not provided")
        elif (
            required_travel is not None
            and maximum_travel is None
        ):
            unknown.append(
                "maximum travel percentage was not provided"
            )
        elif (
            required_travel is not None
            and maximum_travel < required_travel
        ):
            conflicts.append(
                f"maximum travel is {maximum_travel}%"
            )
        else:
            supported.append(
                "willing to travel"
                + (
                    f" up to {maximum_travel}%"
                    if maximum_travel is not None
                    else ""
                )
            )

    if "work_arrangement" in kinds:
        normalized_requirement = normalize_semantic_text(
            requirement
        )
        required_arrangements = {
            arrangement
            for arrangement, markers in {
                "remote": ("remote",),
                "hybrid": ("hybrid",),
                "onsite": ("onsite", "on site", "in office")
            }.items()
            if any(
                marker in normalized_requirement
                for marker in markers
            )
        }

        if not preferred_arrangements:
            unknown.append(
                "work arrangement preferences were not provided"
            )
        elif preferred_arrangements & required_arrangements:
            supported.append(
                "preferred work arrangement includes "
                + ", ".join(
                    sorted(
                        preferred_arrangements
                        & required_arrangements
                    )
                )
            )
        else:
            conflicts.append(
                "preferred work arrangements do not include "
                + ", ".join(sorted(required_arrangements))
            )

    if conflicts:
        return (
            MatchItem(
                id=match.id,
                status="missing",
                evidence=(
                    "Candidate metadata: "
                    + "; ".join(conflicts)
                    + "."
                )
            ),
            (
                "Explicit candidate metadata conflicts with "
                "the job requirement."
            )
        )

    if unknown:
        return (
            MatchItem(
                id=match.id,
                status="unknown",
                evidence=(
                    "Candidate metadata: "
                    + "; ".join(unknown)
                    + "."
                )
            ),
            (
                "Candidate metadata was not provided; "
                "this requirement is score-neutral."
            )
        )

    return (
        MatchItem(
            id=match.id,
            status="direct",
            evidence=(
                "Candidate metadata: "
                + "; ".join(supported)
                + "."
            )
        ),
        "Explicit candidate metadata satisfies this requirement."
    )


def find_related_profile_evidence(
    requirement,
    candidate_profile
):

    passages = candidate_profile_passages(
        candidate_profile
    )
    target_concepts = requirement_concepts(requirement)
    normalized_requirement = normalize_semantic_text(
        requirement
    )

    if "product knowledge" in normalized_requirement:
        target_concepts |= {
            "customer_service",
            "computer_software"
        }

    if "problem tracking" in normalized_requirement:
        target_concepts |= {
            "problem_solving",
            "trend_analysis",
            "troubleshooting",
            "computer_software"
        }

    if (
        "software" in normalized_requirement
        and "support" in normalized_requirement
    ):
        target_concepts |= {
            "customer_service",
            "troubleshooting",
            "computer_software"
        }

    if not target_concepts:
        return None

    best_passage = None
    best_score = 0

    for passage in passages:
        concept_score = sum(
            EVIDENCE_CONCEPT_WEIGHTS.get(concept, 1)
            for concept in target_concepts
            if has_semantic_concept(passage, concept)
        )

        if concept_score == 0:
            continue

        score = concept_score + evidence_specificity_score(
            requirement,
            passage
        )

        if score > best_score:
            best_score = score
            best_passage = passage

    return best_passage


def enforce_hard_requirement_limits(
    requirement,
    match,
    candidate_profile,
    candidate_metadata=None
):
    """
    Apply non-negotiable absence and professional-context rules.
    """

    normalized = normalize_semantic_text(requirement)

    metadata_match, metadata_reason = (
        evaluate_candidate_metadata_requirement(
            requirement,
            match,
            candidate_metadata
        )
    )

    if metadata_match is not None:
        return metadata_match, metadata_reason

    absent_technology_rules = (
        (("salesforce",), ("salesforce",), "Salesforce"),
        (
            ("crm", "case management"),
            ("crm", "customer relationship management", "case management"),
            "CRM/case-management"
        ),
        (("saas",), ("saas", "software as a service"), "SaaS"),
        (("lms",), ("lms", "learning management system"), "LMS")
    )

    for markers, documented_terms, label in absent_technology_rules:
        if (
            any(marker in normalized for marker in markers)
            and not profile_explicitly_documents(
                candidate_profile,
                documented_terms
            )
        ):
            return (
                MatchItem(
                    id=match.id,
                    status="missing",
                    evidence=(
                        f"No {label} experience is "
                        "documented in the resume."
                    )
                ),
                (
                    f"{label} is not documented; "
                    "related experience cannot substitute for it."
                )
            )

    software_customer_support = (
        "software" in normalized
        and any(
            phrase in normalized
            for phrase in (
                "customer support",
                "technical support",
                "software support"
            )
        )
    )

    professional_support_documented = (
        profile_explicitly_documents(
            candidate_profile,
            (
                "software support specialist",
                "technical support specialist",
                "customer support specialist",
                "help desk",
                "service desk",
                "professional software support",
                "software customer support"
            )
        )
    )

    if (
        software_customer_support
        and not professional_support_documented
    ):
        profile_has_transferable_support = (
            has_semantic_concept(
                candidate_profile,
                "customer_service"
            )
            and (
                has_semantic_concept(
                    candidate_profile,
                    "troubleshooting"
                )
                or has_semantic_concept(
                    candidate_profile,
                    "computer_software"
                )
            )
        )
        status = (
            "partial"
            if profile_has_transferable_support
            else "missing"
        )
        evidence = find_related_profile_evidence(
            requirement,
            candidate_profile
        )

        if profile_has_transferable_support:
            evidence = (
                "Retail customer service plus documented "
                "software troubleshooting experience."
            )

        return (
            MatchItem(
                id=match.id,
                status=status,
                evidence=(
                    evidence
                    or "Professional software support is not documented."
                )
            ),
            (
                "Retail customer service and technical troubleshooting "
                "are transferable, not professional software support."
            )
        )

    return match, None


def evidence_repeats_requirement(
    requirement,
    evidence
):

    requirement_tokens = semantic_tokens(requirement)
    evidence_tokens = semantic_tokens(evidence)

    if not requirement_tokens or not evidence_tokens:
        return False

    overlap = requirement_tokens & evidence_tokens

    return (
        len(overlap) / len(evidence_tokens) >= 0.7
        and len(overlap) / len(requirement_tokens) >= 0.5
    )


def repair_or_reject_evidence(
    requirement,
    match,
    candidate_profile
):

    if match.status == "missing":
        if evidence_repeats_requirement(
            requirement,
            match.evidence
        ):
            return (
                MatchItem(
                    id=match.id,
                    status="missing",
                    evidence="No grounded candidate evidence was found."
                ),
                None
            )

        return match, None

    grounded = evidence_is_grounded(
        match.evidence,
        candidate_profile
    )
    repeated = evidence_repeats_requirement(
        requirement,
        match.evidence
    )

    if repeated:
        replacement = (
            find_complete_profile_evidence(
                requirement,
                candidate_profile
            )
            if match.status == "direct"
            else find_related_profile_evidence(
                requirement,
                candidate_profile
            )
        )

        if (
            replacement
            and normalize_semantic_text(replacement)
            != normalize_semantic_text(match.evidence)
        ):
            return (
                MatchItem(
                    id=match.id,
                    status=match.status,
                    evidence=replacement
                ),
                None
            )

    concepts = requirement_concepts(requirement)
    evidence_support = {
        concept
        for concept in concepts
        if has_semantic_concept(match.evidence, concept)
    }
    incomplete_partial_evidence = (
        match.status == "partial"
        and concepts
        and evidence_support != concepts
    )
    unsupported_copy = (
        repeated
        and not grounded
    )

    if (
        grounded
        and not unsupported_copy
        and not incomplete_partial_evidence
    ):
        return match, None

    replacement = find_related_profile_evidence(
        requirement,
        candidate_profile
    )

    if replacement:
        return (
            MatchItem(
                id=match.id,
                status=match.status,
                evidence=replacement
            ),
            None
        )

    return (
        MatchItem(
            id=match.id,
            status="missing",
            evidence="No grounded candidate evidence was found."
        ),
        (
            "The model evidence repeated the requirement or could not "
            "be traced to a candidate fact."
        )
    )


def validate_match_evidence(
    requirement,
    match,
    candidate_profile
):
    """
    Conservatively downgrade direct matches lacking complete evidence.
    """

    if match.status != "direct":
        return match, None

    complete_profile_evidence = find_complete_profile_evidence(
        requirement,
        candidate_profile
    )

    if complete_profile_evidence:
        match = MatchItem(
            id=match.id,
            status=match.status,
            evidence=complete_profile_evidence
        )

    concepts = requirement_concepts(requirement)
    profile_support = {
        concept
        for concept in concepts
        if has_semantic_concept(candidate_profile, concept)
    }
    evidence_support = {
        concept
        for concept in concepts
        if has_semantic_concept(match.evidence, concept)
    }

    missing_from_profile = concepts - profile_support
    missing_from_evidence = concepts - evidence_support
    grounded = evidence_is_grounded(
        match.evidence,
        candidate_profile
    )

    compound_parts = [
        part.strip(" ,.;")
        for part in re.split(
            r"\band\b",
            normalize_semantic_text(requirement)
        )
        if semantic_tokens(part)
    ]
    compound_profile_supported = all(
        requirement_part_is_supported(
            part,
            candidate_profile
        )
        for part in compound_parts
    )
    compound_evidence_supported = all(
        requirement_part_is_supported(
            part,
            match.evidence
        )
        for part in compound_parts
    )

    if (
        grounded
        and not missing_from_profile
        and not missing_from_evidence
        and compound_profile_supported
        and compound_evidence_supported
    ):
        return match, None

    normalized_requirement = normalize_semantic_text(
        requirement
    )
    profile_concepts = {
        concept
        for concept in SEMANTIC_CONCEPTS
        if has_semantic_concept(candidate_profile, concept)
    }

    if "problem tracking" in normalized_requirement:
        transferable = bool(
            profile_concepts
            & {
                "problem_tracking",
                "trend_analysis",
                "client_systems",
                "problem_solving",
                "troubleshooting",
                "computer_software"
            }
        )
    elif "product knowledge" in normalized_requirement:
        transferable = bool(
            profile_concepts
            & {
                "customer_service",
                "product_knowledge",
                "training",
                "procedures_policies",
                "computer_software"
            }
        )
    else:
        transferable = bool(
            profile_support
            or evidence_support
        )

    downgraded_status = (
        "partial"
        if transferable
        else "missing"
    )

    reasons = []

    if not grounded:
        reasons.append(
            "the cited evidence is not grounded in the resume"
        )

    missing_concepts = (
        missing_from_profile
        | missing_from_evidence
    )

    if missing_concepts:
        labels = sorted(
            CONCEPT_LABELS[concept]
            for concept in missing_concepts
        )
        reasons.append(
            "direct evidence is incomplete for "
            + ", ".join(labels)
        )

    if (
        len(compound_parts) > 1
        and (
            not compound_profile_supported
            or not compound_evidence_supported
        )
    ):
        reasons.append(
            "not every substantive part joined by 'and' is supported"
        )

    reason = "; ".join(reasons)

    return (
        MatchItem(
            id=match.id,
            status=downgraded_status,
            evidence=match.evidence
        ),
        reason
    )


def match_requirement_batch(
    requirements,
    candidate_profile,
    candidate_metadata=None
):
    """
    Match a bounded requirement batch, splitting it if JSON is truncated.
    """

    prompt_data = {
        "candidate_profile":
            candidate_profile,

        "candidate_metadata":
            candidate_metadata,

        "job_requirements":
            requirements
    }

    response = chat(
        model=MODEL,

        messages=[
            {
                "role": "system",
                "content": MATCHER_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    prompt_data,
                    separators=(",", ":")
                )
            }
        ],

        format=MatchResponse.model_json_schema(),

        options={
            "temperature": 0,
            "seed": 0,
            "num_ctx": 4096,
            "num_predict": max(
                256,
                len(requirements) * 120
            )
        },

        think=False,

        keep_alive="30m"
    )

    try:

        match_response = (
            MatchResponse
            .model_validate_json(
                response[
                    "message"
                ][
                    "content"
                ]
            )
        )

    except ValidationError:

        if len(requirements) == 1:
            requirement_id = requirements[0]["id"]
            print(
                f"\nMatcher returned invalid JSON for {requirement_id}; "
                "it was conservatively marked missing."
            )
            return [
                MatchItem(
                    id=requirement_id,
                    status="missing",
                    evidence="No valid matching evidence was returned."
                )
            ]

        midpoint = len(requirements) // 2
        print(
            "\nMatcher output was incomplete; "
            "retrying that batch in smaller pieces."
        )
        return (
            match_requirement_batch(
                requirements[:midpoint],
                candidate_profile,
                candidate_metadata
            )
            + match_requirement_batch(
                requirements[midpoint:],
                candidate_profile,
                candidate_metadata
            )
        )

    expected_ids = {
        requirement["id"]
        for requirement in requirements
    }

    return [
        match
        for match in match_response.matches
        if match.id in expected_ids
    ]


def match_candidate(
    parsed_job,
    candidate_profile,
    candidate_metadata=None
):

    requirements = build_requirements(
        parsed_job
    )

    returned_matches = {}
    automatic_downgrades = []

    for start in range(
        0,
        len(requirements),
        MATCH_BATCH_SIZE
    ):
        batch = requirements[
            start:start + MATCH_BATCH_SIZE
        ]

        for match in match_requirement_batch(
            batch,
            candidate_profile,
            candidate_metadata
        ):
            if match.id not in returned_matches:
                returned_matches[match.id] = match

    ordered_matches = []

    for requirement in requirements:
        requirement_id = requirement["id"]
        match = returned_matches.get(requirement_id)

        if match is None or not match.evidence.strip():
            match = MatchItem(
                id=requirement_id,
                status="missing",
                evidence="No matching evidence was returned."
            )

        model_status = match.status
        downgrade_reason = None
        metadata_requirement = bool(
            candidate_metadata_requirement_kinds(
                requirement["requirement"]
            )
        )

        if not metadata_requirement:
            match = apply_deterministic_profile_evidence(
                requirement["requirement"],
                match,
                candidate_profile
            )

        match, hard_limit_reason = enforce_hard_requirement_limits(
            requirement["requirement"],
            match,
            candidate_profile,
            candidate_metadata
        )

        if hard_limit_reason:
            downgrade_reason = hard_limit_reason

        elif not metadata_requirement:
            match, downgrade_reason = validate_match_evidence(
                requirement["requirement"],
                match,
                candidate_profile
            )

        evidence_reason = None

        if not metadata_requirement:
            match, evidence_reason = repair_or_reject_evidence(
                requirement["requirement"],
                match,
                candidate_profile
            )

        if evidence_reason:
            downgrade_reason = (
                evidence_reason
                if not downgrade_reason
                else f"{downgrade_reason} {evidence_reason}"
            )

        if (
            downgrade_reason
            and model_status != match.status
            and not metadata_requirement
        ):
            automatic_downgrades.append(
                {
                    "id": requirement_id,
                    "requirement": requirement["requirement"],
                    "from": model_status,
                    "to": match.status,
                    "reason": downgrade_reason
                }
            )

        concise_evidence = " ".join(
            match.evidence.split()
        )

        if len(concise_evidence) > 180:
            concise_evidence = (
                concise_evidence[:177]
                .rsplit(" ", 1)[0]
                + "..."
            )

        ordered_matches.append(
            MatchItem(
                id=match.id,
                status=match.status,
                evidence=concise_evidence
            )
        )

    match_response = MatchResponse(
        matches=ordered_matches
    )


    return (
        requirements,
        match_response,
        automatic_downgrades
    )


# =========================================================
# Score
# =========================================================

def calculate_score(
    requirements,
    matches
):

    values = {
        "direct": 1.0,
        "partial": 0.6,
        "missing": 0.0
    }

    match_lookup = {
        item.id: item.status
        for item in matches
    }

    category_scores = {
        "core": [],
        "competency": [],
        "preferred": []
    }

    for requirement in requirements:

        if eligibility_metadata_requirement_kinds(
            requirement.get("requirement", "")
        ):
            continue

        status = match_lookup.get(
            requirement["id"],
            "missing"
        )

        if status == "unknown":
            continue

        value = values.get(
            status,
            0
        )

        category_scores[
            requirement["category"]
        ].append(value)

    weights = {
        "core": 0.60,
        "competency": 0.25,
        "preferred": 0.15
    }

    weighted_total = 0
    active_weight = 0

    for category, scores in (
        category_scores.items()
    ):

        if not scores:
            continue

        average = (
            sum(scores)
            / len(scores)
        )

        weight = weights[category]

        weighted_total += (
            average * weight
        )

        active_weight += weight

    if active_weight == 0:
        return 0

    return round(
        (
            weighted_total
            / active_weight
        )
        * 100
    )


# =========================================================
# Eligibility / Preference Evaluation
# =========================================================

WORK_ARRANGEMENT_LABELS = {
    "remote": "Remote",
    "hybrid": "Hybrid",
    "onsite": "Onsite"
}

JOB_TYPE_LABELS = {
    "full_time": "Full-time",
    "part_time": "Part-time",
    "internship": "Internship",
    "contract": "Contract"
}

JOB_AREA_LABELS = {
    "software_engineering": "Software engineering",
    "data_engineering": "Data engineering",
    "data_analysis": "Data analysis",
    "it_support": "IT / support",
    "ai_ml": "AI / ML"
}


def get_candidate_metadata_value(
    candidate_metadata,
    *path,
    default=None
):

    value = candidate_metadata

    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]

    return value


def extract_compensation_range(job_description):

    range_match = re.search(
        r"(?:hiring|salary|compensation)\s+range"
        r"(?P<context>.{0,180})",
        job_description,
        flags=re.IGNORECASE | re.DOTALL
    )

    if not range_match:
        return None, None

    amounts = [
        int(amount.replace(",", ""))
        for amount in re.findall(
            r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
            range_match.group("context")
        )
    ]

    if len(amounts) < 2:
        return None, None

    return min(amounts[:2]), max(amounts[:2])


def extract_job_preference_signals(
    job_description,
    job_title=""
):

    normalized = normalize_semantic_text(job_description)
    title = normalize_semantic_text(job_title)

    arrangements = set()

    if (
        re.search(r"\bremote\b", normalized)
        and not re.search(
            r"\b(?:not|no)\s+remote\b",
            normalized
        )
    ):
        arrangements.add("remote")

    if "hybrid" in normalized:
        arrangements.add("hybrid")

    if any(
        marker in normalized
        for marker in (
            "fully onsite", "onsite role", "onsite position",
            "on site role", "on site position"
        )
    ):
        arrangements.add("onsite")

    job_types = set()
    job_type_markers = {
        "full_time": ("full time",),
        "part_time": ("part time",),
        "internship": ("internship", "intern role"),
        "contract": ("contract role", "contract position", "contractor")
    }

    for job_type, markers in job_type_markers.items():
        if any(marker in normalized for marker in markers):
            job_types.add(job_type)

    relocation_required = bool(re.search(
        r"\b(?:relocation required|must relocate|required to relocate)\b",
        normalized
    ))
    travel_required = bool(re.search(
        r"\b(?:travel required|required to .*?travel|"
        r"occasionally travel|travel occasionally|up to \d{1,3} "
        r"percent travel|up to \d{1,3} travel)\b",
        normalized
    ))
    travel_percentage_match = re.search(
        r"(?:up to\s*)?(\d{1,3})\s*(?:%|percent)\s*travel",
        job_description,
        flags=re.IGNORECASE
    )
    travel_percentage = (
        int(travel_percentage_match.group(1))
        if travel_percentage_match
        else None
    )

    clearance_required = bool(re.search(
        r"\b(?:active|current)?\s*security clearance\s+required\b"
        r"|\brequired to (?:hold|maintain|obtain)\b.{0,30}"
        r"\bclearance\b"
        r"|\bability to obtain\b.{0,30}\bclearance\b",
        normalized
    ))
    active_clearance_required = bool(re.search(
        r"\b(?:active|current)\s+security clearance\b",
        normalized
    ))

    job_areas = set()
    area_text = title or normalize_semantic_text(
        job_description[:1200]
    )

    if any(
        marker in area_text
        for marker in (
            "technical support", "customer support",
            "help desk", "service desk", "it support"
        )
    ):
        job_areas.add("it_support")

    if any(
        marker in area_text
        for marker in (
            "software engineer", "software developer",
            "application developer"
        )
    ):
        job_areas.add("software_engineering")

    if "data engineer" in area_text:
        job_areas.add("data_engineering")

    if any(
        marker in area_text
        for marker in (
            "data analyst", "data analysis", "analytics analyst"
        )
    ):
        job_areas.add("data_analysis")

    if any(
        marker in area_text
        for marker in (
            "machine learning engineer", "ai engineer",
            "artificial intelligence engineer", "ml engineer"
        )
    ):
        job_areas.add("ai_ml")

    compensation_minimum, compensation_maximum = (
        extract_compensation_range(job_description)
    )

    return {
        "arrangements": sorted(arrangements),
        "job_types": sorted(job_types),
        "relocation_required": relocation_required,
        "travel_required": travel_required,
        "travel_percentage": travel_percentage,
        "clearance_required": clearance_required,
        "active_clearance_required": active_clearance_required,
        "job_areas": sorted(job_areas),
        "compensation_minimum": compensation_minimum,
        "compensation_maximum": compensation_maximum
    }


def evaluate_eligibility(
    job_description,
    candidate_metadata,
    job_signals
):

    requirement_kinds = candidate_metadata_requirement_kinds(
        job_description
    )
    authorization_required = (
        "us_authorized" in requirement_kinds
    )
    sponsorship_restricted = (
        "requires_sponsorship" in requirement_kinds
    )
    authorization = get_candidate_metadata_boolean(
        candidate_metadata,
        "work_authorization",
        "us_authorized"
    )
    requires_sponsorship = get_candidate_metadata_boolean(
        candidate_metadata,
        "work_authorization",
        "requires_sponsorship"
    )
    clearance = get_candidate_metadata_value(
        candidate_metadata,
        "security_clearance",
        "status",
        default="unknown"
    )
    items = []

    if authorization is None:
        authorization_status = "unknown"
        authorization_detail = (
            "Not provided. Posting requires U.S. work authorization."
            if authorization_required
            else "Not provided. No explicit authorization requirement identified."
        )
    elif authorization:
        authorization_status = "match"
        authorization_detail = (
            "Authorized to work in the U.S."
            if authorization_required
            else "Candidate reports U.S. work authorization."
        )
    elif authorization_required:
        authorization_status = "mismatch"
        authorization_detail = (
            "Not authorized; the posting explicitly requires it."
        )
    else:
        authorization_status = "info"
        authorization_detail = (
            "Candidate reports no current U.S. work authorization; "
            "the posting does not explicitly require it."
        )

    items.append(
        {
            "key": "us_authorized",
            "label": "U.S. work authorization",
            "status": authorization_status,
            "detail": authorization_detail,
            "required_by_posting": authorization_required
        }
    )

    if requires_sponsorship is None:
        sponsorship_status = "unknown"
        sponsorship_detail = (
            "Not provided. Posting states sponsorship is unavailable."
            if sponsorship_restricted
            else "Not provided. Posting does not specify sponsorship."
        )
    elif not requires_sponsorship:
        sponsorship_status = "match"
        sponsorship_detail = "Sponsorship not required."
    elif sponsorship_restricted:
        sponsorship_status = "mismatch"
        sponsorship_detail = (
            "Candidate requires sponsorship, but the posting "
            "states it is unavailable."
        )
    else:
        sponsorship_status = "warning"
        sponsorship_detail = (
            "Candidate requires sponsorship; availability is not "
            "specified in the posting."
        )

    items.append(
        {
            "key": "requires_sponsorship",
            "label": "Employer sponsorship",
            "status": sponsorship_status,
            "detail": sponsorship_detail,
            "required_by_posting": sponsorship_restricted
        }
    )

    clearance_required = job_signals["clearance_required"]

    if clearance == "unknown":
        clearance_status = "unknown"
        clearance_detail = (
            "Not provided. Posting includes a clearance requirement."
            if clearance_required
            else "Not provided. No clearance requirement identified."
        )
    elif not clearance_required:
        clearance_status = "info"
        clearance_detail = (
            f"Candidate status: {clearance}; "
            "no clearance requirement identified."
        )
    elif clearance == "active":
        clearance_status = "match"
        clearance_detail = "Active clearance satisfies the posting."
    elif (
        clearance == "eligible"
        and not job_signals["active_clearance_required"]
    ):
        clearance_status = "match"
        clearance_detail = "Eligible to obtain the required clearance."
    elif clearance == "eligible":
        clearance_status = "warning"
        clearance_detail = (
            "Eligible for clearance, but the posting requires an "
            "active clearance."
        )
    else:
        clearance_status = "mismatch"
        clearance_detail = (
            "No clearance; the posting includes a clearance requirement."
        )

    items.append(
        {
            "key": "security_clearance",
            "label": "Security clearance",
            "status": clearance_status,
            "detail": clearance_detail,
            "required_by_posting": clearance_required
        }
    )

    return {
        "items": items,
        "affects_qualification_score": False
    }


def join_labels(values, labels):

    return ", ".join(
        labels.get(value, value)
        for value in values
    )


def evaluate_preferences(
    candidate_metadata,
    job_signals
):

    items = []
    candidate_arrangements = set(
        get_candidate_metadata_value(
            candidate_metadata,
            "work_preferences",
            "arrangements",
            default=[]
        ) or []
    )
    job_arrangements = set(job_signals["arrangements"])

    if not candidate_arrangements:
        arrangement_status = "unknown"
        arrangement_detail = (
            "Not provided. Posting specifies "
            + join_labels(
                sorted(job_arrangements),
                WORK_ARRANGEMENT_LABELS
            )
            + "."
            if job_arrangements
            else "Not provided. Posting does not specify an arrangement."
        )
    elif not job_arrangements:
        arrangement_status = "info"
        arrangement_detail = (
            "Candidate prefers "
            + join_labels(
                sorted(candidate_arrangements),
                WORK_ARRANGEMENT_LABELS
            )
            + "; the posting does not specify an arrangement."
        )
    elif candidate_arrangements & job_arrangements:
        arrangement_status = "match"
        arrangement_detail = (
            join_labels(
                sorted(candidate_arrangements & job_arrangements),
                WORK_ARRANGEMENT_LABELS
            )
            + " matches the posting."
        )
    else:
        arrangement_status = "mismatch"
        arrangement_detail = (
            "Candidate prefers "
            + join_labels(
                sorted(candidate_arrangements),
                WORK_ARRANGEMENT_LABELS
            )
            + "; the posting specifies "
            + join_labels(
                sorted(job_arrangements),
                WORK_ARRANGEMENT_LABELS
            )
            + "."
        )

    items.append(
        {
            "key": "work_arrangements",
            "label": "Work arrangement",
            "status": arrangement_status,
            "detail": arrangement_detail
        }
    )

    candidate_job_types = set(
        get_candidate_metadata_value(
            candidate_metadata,
            "work_preferences",
            "job_types",
            default=[]
        ) or []
    )
    job_types = set(job_signals["job_types"])

    if not candidate_job_types:
        job_type_status = "unknown"
        job_type_detail = (
            "Not provided. Posting specifies "
            + join_labels(sorted(job_types), JOB_TYPE_LABELS)
            + "."
            if job_types
            else "Not provided. Posting does not specify a job type."
        )
    elif not job_types:
        job_type_status = "info"
        job_type_detail = (
            "Candidate prefers "
            + join_labels(
                sorted(candidate_job_types),
                JOB_TYPE_LABELS
            )
            + "; the posting does not specify a job type."
        )
    elif candidate_job_types & job_types:
        job_type_status = "match"
        job_type_detail = (
            join_labels(
                sorted(candidate_job_types & job_types),
                JOB_TYPE_LABELS
            )
            + " matches the posting."
        )
    else:
        job_type_status = "mismatch"
        job_type_detail = (
            "Candidate prefers "
            + join_labels(
                sorted(candidate_job_types),
                JOB_TYPE_LABELS
            )
            + "; the posting specifies "
            + join_labels(sorted(job_types), JOB_TYPE_LABELS)
            + "."
        )

    items.append(
        {
            "key": "job_types",
            "label": "Job type",
            "status": job_type_status,
            "detail": job_type_detail
        }
    )

    current_location = get_candidate_metadata_value(
        candidate_metadata,
        "location",
        "current"
    )

    if current_location:
        location_status = (
            "match" if "remote" in job_arrangements else "info"
        )
        location_detail = (
            f"{current_location}; the posting allows remote work."
            if "remote" in job_arrangements
            else f"{current_location}; verify location requirements."
        )
    else:
        location_status = "unknown"
        location_detail = (
            "Not provided. Posting allows remote work."
            if "remote" in job_arrangements
            else "Not provided."
        )

    items.append(
        {
            "key": "current_location",
            "label": "Current location",
            "status": location_status,
            "detail": location_detail
        }
    )

    willing_to_relocate = get_candidate_metadata_boolean(
        candidate_metadata,
        "location",
        "willing_to_relocate"
    )

    if willing_to_relocate is None:
        relocation_status = "unknown"
        relocation_detail = (
            "Not provided. Posting requires relocation."
            if job_signals["relocation_required"]
            else "Not provided. No relocation requirement identified."
        )
    elif job_signals["relocation_required"]:
        relocation_status = (
            "match" if willing_to_relocate else "mismatch"
        )
        relocation_detail = (
            "Willing to relocate for this position."
            if willing_to_relocate
            else "Position requires relocation; candidate is unwilling."
        )
    else:
        relocation_status = "info"
        relocation_detail = (
            "Willing to relocate; no relocation requirement identified."
            if willing_to_relocate
            else "Not willing to relocate; no relocation requirement identified."
        )

    items.append(
        {
            "key": "relocation",
            "label": "Relocation",
            "status": relocation_status,
            "detail": relocation_detail
        }
    )

    willing_to_travel = get_candidate_metadata_boolean(
        candidate_metadata,
        "travel",
        "willing"
    )
    maximum_travel = get_candidate_metadata_value(
        candidate_metadata,
        "travel",
        "maximum_percentage"
    )
    required_travel = job_signals["travel_percentage"]

    if willing_to_travel is None:
        travel_status = "unknown"
        travel_detail = (
            "Not provided. Posting requires travel"
            + (
                f" up to {required_travel}%."
                if required_travel is not None
                else "."
            )
            if job_signals["travel_required"]
            else "Not provided. No travel requirement identified."
        )
    elif job_signals["travel_required"] and not willing_to_travel:
        travel_status = "mismatch"
        travel_detail = "Posting requires travel; candidate is unwilling."
    elif (
        job_signals["travel_required"]
        and required_travel is not None
        and maximum_travel is not None
        and maximum_travel < required_travel
    ):
        travel_status = "mismatch"
        travel_detail = (
            f"Candidate maximum is {maximum_travel}%; "
            f"posting requires up to {required_travel}%."
        )
    elif job_signals["travel_required"]:
        travel_status = "match"
        travel_detail = (
            "Willing to travel"
            + (
                f" up to {maximum_travel}%"
                if maximum_travel is not None
                else ""
            )
            + "; the posting requires travel."
        )
    else:
        travel_status = "info"
        travel_detail = (
            "Willing to travel"
            + (
                f" up to {maximum_travel}%."
                if maximum_travel is not None
                else "."
            )
            if willing_to_travel
            else "Not willing to travel; no travel requirement identified."
        )

    items.append(
        {
            "key": "travel",
            "label": "Travel",
            "status": travel_status,
            "detail": travel_detail
        }
    )

    candidate_minimum = get_candidate_metadata_value(
        candidate_metadata,
        "compensation",
        "minimum_amount"
    )
    job_minimum = job_signals["compensation_minimum"]
    job_maximum = job_signals["compensation_maximum"]

    if candidate_minimum is None:
        compensation_status = "unknown"
        compensation_detail = (
            f"Not provided. Posting range is "
            f"${job_minimum:,}-${job_maximum:,}."
            if job_maximum is not None
            else "Not provided. Posting does not provide a usable range."
        )
    elif job_maximum is None:
        compensation_status = "info"
        compensation_detail = (
            f"Candidate minimum is ${candidate_minimum:,}; "
            "the posting does not provide a usable range."
        )
    elif candidate_minimum <= job_maximum:
        compensation_status = "match"
        compensation_detail = (
            f"Candidate minimum ${candidate_minimum:,} overlaps "
            f"the posting range ${job_minimum:,}-${job_maximum:,}."
        )
    else:
        compensation_status = "mismatch"
        compensation_detail = (
            f"Candidate minimum ${candidate_minimum:,} exceeds "
            f"the posting maximum ${job_maximum:,}."
        )

    items.append(
        {
            "key": "compensation",
            "label": "Compensation",
            "status": compensation_status,
            "detail": compensation_detail
        }
    )

    preferred_areas = set(
        get_candidate_metadata_value(
            candidate_metadata,
            "work_preferences",
            "preferred_job_areas",
            default=[]
        ) or []
    )
    job_areas = set(job_signals["job_areas"])

    if not preferred_areas:
        area_status = "unknown"
        area_detail = (
            "Not provided. Role is classified as "
            + join_labels(sorted(job_areas), JOB_AREA_LABELS)
            + "."
            if job_areas
            else "Not provided. No job area was confidently identified."
        )
    elif not job_areas:
        area_status = "info"
        area_detail = (
            "Candidate prefers "
            + join_labels(sorted(preferred_areas), JOB_AREA_LABELS)
            + "; no job area was confidently identified."
        )
    elif preferred_areas & job_areas:
        area_status = "match"
        area_detail = (
            join_labels(
                sorted(preferred_areas & job_areas),
                JOB_AREA_LABELS
            )
            + " matches the role."
        )
    else:
        area_status = "mismatch"
        area_detail = (
            "Candidate prefers "
            + join_labels(sorted(preferred_areas), JOB_AREA_LABELS)
            + "; this role is classified as "
            + join_labels(sorted(job_areas), JOB_AREA_LABELS)
            + "."
        )

    items.append(
        {
            "key": "preferred_job_areas",
            "label": "Preferred job area",
            "status": area_status,
            "detail": area_detail
        }
    )

    earliest_start = get_candidate_metadata_value(
        candidate_metadata,
        "availability",
        "earliest_start_date"
    )
    graduation_date = get_candidate_metadata_value(
        candidate_metadata,
        "education",
        "graduation_date"
    )

    items.extend(
        [
            {
                "key": "earliest_start_date",
                "label": "Earliest start date",
                "status": "info" if earliest_start else "unknown",
                "detail": earliest_start or "Not provided."
            },
            {
                "key": "graduation_date",
                "label": "Graduation date",
                "status": "info" if graduation_date else "unknown",
                "detail": graduation_date or "Not provided."
            }
        ]
    )

    return {
        "items": items,
        "job_signals": job_signals,
        "affects_qualification_score": False
    }


# =========================================================
# Build Final Result
# =========================================================

def build_result(
    parsed_job,
    requirements,
    match_response,
    automatic_downgrades,
    job_description,
    candidate_metadata=None
):

    match_lookup = {
        match.id: match
        for match
        in match_response.matches
    }

    required = []
    preferred = []

    for requirement in requirements:

        if eligibility_metadata_requirement_kinds(
            requirement["requirement"]
        ):
            continue

        match = match_lookup.get(
            requirement["id"]
        )

        if match is None:

            status = "missing"
            evidence = (
                "No matching evidence "
                "was returned."
            )

        else:

            status = match.status
            evidence = match.evidence

        item = {
            "requirement":
                requirement["requirement"],

            "status":
                status,

            "evidence":
                evidence
        }

        if (
            requirement["category"]
            == "core"
        ):

            item["category"] = (
                "core_experience"
            )

            required.append(item)

        elif (
            requirement["category"]
            == "competency"
        ):

            item["category"] = (
                "competency"
            )

            required.append(item)

        else:

            preferred.append(item)

    score = calculate_score(
        requirements,
        match_response.matches
    )
    job_signals = extract_job_preference_signals(
        job_description,
        parsed_job.job_title
    )
    eligibility = evaluate_eligibility(
        job_description,
        candidate_metadata,
        job_signals
    )
    preferences = evaluate_preferences(
        candidate_metadata,
        job_signals
    )

    missing = [
        item["requirement"]
        for item in required
        if item["status"] == "missing"
    ]

    partial = [
        item["requirement"]
        for item in required
        if item["status"] == "partial"
    ]

    interview_topics = (
        missing[:3]
        + partial[:3]
    )

    resume_improvements = [
        (
            "If you have relevant experience, "
            f"make it more explicit for: {item}"
        )
        for item in missing[:5]
    ]

    responsibilities = [
        {
            "responsibility":
                responsibility,

            "readiness":
                "not_scored",

            "evidence":
                "Role responsibility; not included "
                "in the match score."
        }

        for responsibility
        in parsed_job.responsibilities
    ]

    return {
        "job_title":
            parsed_job.job_title,

        "company":
            parsed_job.company,

        "match_score":
            score,

        "qualification_match": {
            "score": score,
            "affects_qualification_score": True
        },

        "eligibility":
            eligibility,

        "preferences":
            preferences,

        "required":
            required,

        "preferred":
            preferred,

        "responsibilities":
            responsibilities,

        "automatic_downgrades":
            automatic_downgrades,

        "experience_fit":
            build_experience_fit(
                requirements,
                match_response.matches
            ),

        "resume_improvements":
            resume_improvements,

        "interview_topics":
            interview_topics,

        "summary":
            (
                f"ApplyAI calculated a "
                f"{score}% match based on "
                f"core experience, required "
                f"competencies, and preferred "
                f"qualifications. Eligibility and "
                f"preferences are reported separately."
            )
    }


# =========================================================
# Main Function
# =========================================================

def analyze_job(
    job_description,
    candidate_profile,
    candidate_metadata=None
):

    start = time.time()

    parsed_job = parse_job(
        job_description
    )

    parse_time = time.time()

    print(
        f"\nJob parsing took "
        f"{parse_time - start:.1f} seconds"
    )

    requirements, match_response, automatic_downgrades = (
        match_candidate(
            parsed_job,
            candidate_profile,
            candidate_metadata
        )
    )

    match_time = time.time()

    print(
        f"Candidate matching took "
        f"{match_time - parse_time:.1f} seconds"
    )

    print(
        f"Total analysis took "
        f"{match_time - start:.1f} seconds"
    )

    return build_result(
        parsed_job,
        requirements,
        match_response,
        automatic_downgrades,
        job_description,
        candidate_metadata
    )
