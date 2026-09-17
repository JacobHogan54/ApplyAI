import unittest
import json
from pathlib import Path

from agent import (
    MatchItem,
    apply_deterministic_profile_evidence,
    build_experience_fit,
    build_requirements,
    enforce_hard_requirement_limits,
    find_complete_profile_evidence,
    find_related_profile_evidence,
    parse_job,
    repair_or_reject_evidence,
    work_authorization_is_documented,
)


CANDIDATE_RESUME = """
Customer-facing work at Publix adds high-volume service, communication,
organization, and reliability in fast-paced environments.
Publix Grocery Clerk: Provide customer service in a high-traffic environment.
Balance customer needs, stocking responsibilities, and competing priorities
while maintaining accuracy under time-sensitive conditions.
Technical skills include software troubleshooting, PC hardware troubleshooting,
methodical debugging, and issue resolution.
ApplyAI: Built and debugged a local AI application using Python, Streamlit,
Ollama, and Qwen, learning unfamiliar tools and integration workflows.
Driving Intervention Analysis - Senior Design: Collaborate in a two-person
development team using GitHub, code review, testing, and documentation.
Analyze high-frequency telemetry using statistical and machine-learning methods.
Algorithm Optimization: Designed multiple approaches and analyzed correctness,
time complexity, and runtime results.
ACM member: Support peers through tutoring, technical discussion, and
collaborative technical problem solving.
"""


COLLEGE_BOARD_POSTING = """
Job Description
College Board – OCEO
About the Opportunity
The AP Program is conducting a search for a Technical Support Specialist, AP Classroom User Support. You'll support AP Classroom users.
In this role, you will:
Review and triage incoming support requests.
About You:
Required Knowledge/Experience:
A background in software customer support
Experience with online customer inquiries including high-volume seasonal support demands and support case management processes
Experience using Salesforce for case management and email templates
Preferred Knowledge/Experience:
Experience with process improvement
Experience supporting SaaS applications and LMS technical queries
Exceptional candidates can effectively speak to:
Must have:
Strong organizational skills with a process-oriented mindset
Ability to prioritize multiple competing support demands, using recommended operating procedures
Ability to quickly learn complex, unfamiliar tools
All roles at College Board require:
Curiosity and enthusiasm for emerging technologies and AI-driven solutions
A collaborative and empathetic approach
Authorization to work in the United States
About Our Process
Application review begins immediately.
"""


class CollegeBoardRegressionTests(unittest.TestCase):

    def model_match(self, status="direct", evidence="Copied requirement"):
        return MatchItem(id="K1", status=status, evidence=evidence)

    def test_parser_preserves_three_core_items_and_company(self):
        parsed = parse_job(COLLEGE_BOARD_POSTING)
        requirements = build_requirements(parsed)

        self.assertEqual(parsed.company, "College Board")
        self.assertEqual(
            parsed.job_title,
            "Technical Support Specialist, AP Classroom User Support",
        )
        self.assertEqual(len(parsed.core_required), 3)
        self.assertEqual(len(parsed.required_competencies), 6)
        self.assertEqual(len(parsed.preferred), 2)
        self.assertTrue(
            all(item["category"] == "core" for item in requirements[:3])
        )
        self.assertIn(
            "Of the 3 core experience requirements",
            build_experience_fit(requirements, []),
        )

    def test_software_customer_support_is_partial_at_most(self):
        match, reason = enforce_hard_requirement_limits(
            "A background in software customer support",
            self.model_match("direct", "Publix customer service"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "partial")
        self.assertIn("transferable", reason)

    def test_salesforce_is_missing_when_not_documented(self):
        match, reason = enforce_hard_requirement_limits(
            "Experience using Salesforce for case management",
            self.model_match("partial", "Related database experience"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "missing")
        self.assertIn("not documented", reason)

    def test_case_management_is_missing_when_not_documented(self):
        match, _ = enforce_hard_requirement_limits(
            "Experience with support case management processes",
            self.model_match("partial", "Resolved retail issues"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "missing")

    def test_work_authorization_is_not_inferred(self):
        match, reason = enforce_hard_requirement_limits(
            "Authorization to work in the United States",
            self.model_match("direct", "Lives in Florida"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "unknown")
        self.assertIn("score-neutral", reason)

    def test_structured_work_authorization_field_is_explicit(self):
        profile = json.loads(
            Path("candidate_profile.json").read_text(
                encoding="utf-8"
            )
        )["candidate_metadata"]

        self.assertEqual(
            profile["work_authorization"]["us_authorized"],
            None,
        )
        self.assertEqual(
            profile["work_authorization"]["requires_sponsorship"],
            None,
        )
        self.assertFalse(
            work_authorization_is_documented(profile)
        )

        profile["work_authorization"]["us_authorized"] = True
        self.assertTrue(
            work_authorization_is_documented(profile)
        )

    def test_organization_is_recovered_from_publix(self):
        requirement = "Strong organizational skills"
        match = apply_deterministic_profile_evidence(
            requirement,
            self.model_match("missing", "No evidence"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "direct")
        self.assertIn("competing priorities", match.evidence.lower())

    def test_prioritization_is_not_left_missing(self):
        requirement = (
            "Ability to prioritize multiple competing support demands, "
            "using recommended operating procedures"
        )
        match = apply_deterministic_profile_evidence(
            requirement,
            self.model_match("missing", "No evidence"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "partial")
        self.assertIn("competing priorities", match.evidence.lower())

    def test_emerging_technology_is_recovered_from_applyai(self):
        requirement = "Curiosity and enthusiasm for emerging technologies"
        match = apply_deterministic_profile_evidence(
            requirement,
            self.model_match("missing", "No evidence"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "direct")
        self.assertTrue(
            any(
                tool in match.evidence
                for tool in ("ApplyAI", "Ollama", "Qwen", "Streamlit")
            )
        )

    def test_quickly_learning_unfamiliar_tools_uses_applyai(self):
        requirement = (
            "Ability to quickly learn and navigate a feature-rich "
            "application, with comfort ramping up on complex, "
            "unfamiliar tools"
        )
        match = apply_deterministic_profile_evidence(
            requirement,
            self.model_match("missing", "No evidence"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "direct")
        self.assertIn("learning unfamiliar tools", match.evidence)

    def test_collaboration_is_recovered_from_senior_design(self):
        requirement = "A collaborative approach in a small team"
        match = apply_deterministic_profile_evidence(
            requirement,
            self.model_match("missing", "No evidence"),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "direct")
        self.assertIn("two-person", match.evidence)

    def test_analytical_evidence_prefers_named_project(self):
        evidence = find_related_profile_evidence(
            "Strong analytical thinking and structured problem-solving "
            "with root-cause investigation",
            CANDIDATE_RESUME,
        )

        self.assertTrue(
            any(
                source in evidence
                for source in (
                    "ApplyAI",
                    "Senior Design",
                    "Algorithm Optimization",
                )
            )
        )
        self.assertTrue(
            any(
                action in evidence.lower()
                for action in (
                    "debugged",
                    "analyzed correctness",
                    "statistical",
                    "machine-learning",
                    "extract features",
                    "correctness",
                    "time complexity",
                )
            )
        )
        self.assertNotIn("Technical skills include", evidence)

    def test_organization_evidence_prefers_publix_priorities(self):
        evidence = find_complete_profile_evidence(
            "Strong organizational skills",
            CANDIDATE_RESUME,
        )

        self.assertIn("Publix", evidence)
        self.assertIn("competing priorities", evidence)

    def test_emerging_technology_evidence_prefers_applyai_tools(self):
        evidence = find_complete_profile_evidence(
            "Curiosity about emerging technologies",
            CANDIDATE_RESUME,
        )

        self.assertIn("ApplyAI", evidence)
        self.assertIn("Ollama", evidence)
        self.assertIn("Qwen", evidence)

    def test_collaboration_evidence_prefers_senior_design(self):
        evidence = find_complete_profile_evidence(
            "A collaborative approach in a small team",
            CANDIDATE_RESUME,
        )

        self.assertIn("Senior Design", evidence)
        self.assertIn("two-person", evidence)

    def test_copied_requirement_evidence_is_replaced(self):
        requirement = "Strong organizational skills"
        match, _ = repair_or_reject_evidence(
            requirement,
            self.model_match("direct", requirement),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "direct")
        self.assertNotEqual(match.evidence, requirement)
        self.assertIn("competing priorities", match.evidence.lower())

    def test_unsupported_copied_evidence_is_rejected(self):
        requirement = "Experience creating Tableau dashboards"
        match, reason = repair_or_reject_evidence(
            requirement,
            self.model_match("partial", requirement),
            CANDIDATE_RESUME,
        )

        self.assertEqual(match.status, "missing")
        self.assertIsNotNone(reason)


if __name__ == "__main__":
    unittest.main()
