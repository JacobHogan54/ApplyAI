import unittest

from agent import (
    MatchItem,
    MatchResponse,
    ParsedJob,
    build_result,
    calculate_score,
    candidate_metadata_requirement_kinds,
    enforce_hard_requirement_limits,
    evaluate_eligibility,
    evaluate_preferences,
    extract_job_preference_signals,
)


RESUME_WITH_INDIRECT_AUTHORIZATION_CLAIMS = """
Lives in Florida and attends a U.S. university.
Resume text says authorized to work in the United States.
"""


class CandidateMetadataTests(unittest.TestCase):

    def model_match(self):
        return MatchItem(
            id="K1",
            status="missing",
            evidence="No resume evidence",
        )

    def metadata(self, authorized=None, sponsorship=None):
        return {
            "schema_version": 2,
            "work_authorization": {
                "us_authorized": authorized,
                "requires_sponsorship": sponsorship,
            },
            "location": {
                "current": None,
                "willing_to_relocate": None,
            },
            "work_preferences": {
                "arrangements": [],
                "job_types": [],
                "preferred_job_areas": [],
            },
            "availability": {"earliest_start_date": None},
            "education": {"graduation_date": None},
            "travel": {
                "willing": None,
                "maximum_percentage": None,
            },
            "security_clearance": {"status": "unknown"},
            "compensation": {
                "minimum_amount": None,
                "currency": "USD",
                "period": "annual",
            },
        }

    def evaluate(self, requirement, metadata):
        return enforce_hard_requirement_limits(
            requirement,
            self.model_match(),
            RESUME_WITH_INDIRECT_AUTHORIZATION_CLAIMS,
            metadata,
        )

    def test_authorized_true_is_direct(self):
        match, _ = self.evaluate(
            "Authorization to work in the United States",
            self.metadata(authorized=True),
        )

        self.assertEqual(match.status, "direct")
        self.assertIn("Candidate metadata", match.evidence)

    def test_authorized_false_is_missing(self):
        match, _ = self.evaluate(
            "Must be legally authorized to work in the United States",
            self.metadata(authorized=False),
        )

        self.assertEqual(match.status, "missing")

    def test_unknown_authorization_is_not_inferred_from_resume(self):
        match, reason = self.evaluate(
            "U.S. work authorization is required",
            self.metadata(),
        )

        self.assertEqual(match.status, "unknown")
        self.assertIn("score-neutral", reason)

    def test_sponsorship_is_evaluated_independently(self):
        match, _ = self.evaluate(
            "Must be able to work without employer sponsorship",
            self.metadata(
                authorized=False,
                sponsorship=False,
            ),
        )

        self.assertEqual(match.status, "direct")
        self.assertIn("sponsorship is not required", match.evidence)

    def test_required_sponsorship_conflicts_with_no_sponsorship_role(self):
        match, _ = self.evaluate(
            "We do not offer visa sponsorship for this role",
            self.metadata(
                authorized=True,
                sponsorship=True,
            ),
        )

        self.assertEqual(match.status, "missing")
        self.assertIn("sponsorship is required", match.evidence)

    def test_combined_requirement_needs_both_metadata_values(self):
        match, _ = self.evaluate(
            "Must be authorized to work in the U.S. without sponsorship",
            self.metadata(
                authorized=True,
                sponsorship=None,
            ),
        )

        self.assertEqual(match.status, "unknown")

    def test_employer_sponsorship_question_is_detected(self):
        kinds = candidate_metadata_requirement_kinds(
            "Will you now or in the future require employer sponsorship?"
        )

        self.assertEqual(kinds, {"requires_sponsorship"})

    def test_security_clearance_uses_metadata_only(self):
        metadata = self.metadata()
        requirement = "Must hold an active security clearance"

        unknown_match, _ = self.evaluate(
            requirement,
            metadata,
        )
        self.assertEqual(unknown_match.status, "unknown")

        metadata["security_clearance"]["status"] = "active"
        active_match, _ = self.evaluate(
            requirement,
            metadata,
        )
        self.assertEqual(active_match.status, "direct")

    def test_unknown_metadata_does_not_reduce_score(self):
        requirements = [
            {"id": "C1", "category": "core"},
            {"id": "K1", "category": "competency"},
            {"id": "K2", "category": "competency"},
        ]
        matches = [
            MatchItem(id="C1", status="direct", evidence="Experience"),
            MatchItem(id="K1", status="direct", evidence="Skill"),
            MatchItem(id="K2", status="unknown", evidence="Not provided"),
        ]

        self.assertEqual(
            calculate_score(requirements, matches),
            100,
        )

    def test_eligibility_requirement_never_changes_qualification_score(self):
        requirements = [
            {
                "id": "C1",
                "category": "core",
                "requirement": "Three years of customer support",
            },
            {
                "id": "K1",
                "category": "competency",
                "requirement": (
                    "Authorization to work in the United States"
                ),
            },
        ]
        matches = [
            MatchItem(id="C1", status="direct", evidence="Support"),
            MatchItem(id="K1", status="missing", evidence="Not authorized"),
        ]

        self.assertEqual(
            calculate_score(requirements, matches),
            100,
        )

    def test_college_board_preference_signals_are_separate(self):
        posting = """
        This is a remote role with a hybrid option.
        This is a full-time position.
        Employees are required to occasionally travel.
        Authorization to work in the United States is required.
        The hiring range for this role is $44,000-$75,000.
        Technical Support Specialist
        """
        signals = extract_job_preference_signals(
            posting,
            "Technical Support Specialist",
        )

        self.assertEqual(
            set(signals["arrangements"]),
            {"remote", "hybrid"},
        )
        self.assertEqual(signals["job_types"], ["full_time"])
        self.assertTrue(signals["travel_required"])
        self.assertEqual(signals["compensation_minimum"], 44000)
        self.assertEqual(signals["compensation_maximum"], 75000)
        self.assertEqual(signals["job_areas"], ["it_support"])

        eligibility = evaluate_eligibility(
            posting,
            self.metadata(),
            signals,
        )
        preferences = evaluate_preferences(
            self.metadata(),
            signals,
        )

        self.assertFalse(eligibility["affects_qualification_score"])
        self.assertFalse(preferences["affects_qualification_score"])
        self.assertEqual(
            eligibility["items"][0]["status"],
            "unknown",
        )

    def test_preference_matches_and_mismatches_are_informational(self):
        posting = """
        This is a remote, full-time role requiring up to 25% travel.
        The salary range is $50,000-$70,000.
        """
        metadata = self.metadata(
            authorized=True,
            sponsorship=False,
        )
        metadata["work_preferences"].update(
            {
                "arrangements": ["remote"],
                "job_types": ["full_time"],
                "preferred_job_areas": [],
            }
        )
        metadata["travel"].update(
            {"willing": True, "maximum_percentage": 10}
        )
        metadata["compensation"]["minimum_amount"] = 80000
        signals = extract_job_preference_signals(posting)
        preferences = evaluate_preferences(metadata, signals)
        item_statuses = {
            item["key"]: item["status"]
            for item in preferences["items"]
        }

        self.assertEqual(item_statuses["work_arrangements"], "match")
        self.assertEqual(item_statuses["job_types"], "match")
        self.assertEqual(item_statuses["travel"], "mismatch")
        self.assertEqual(item_statuses["compensation"], "mismatch")

    def test_result_separates_eligibility_from_qualifications(self):
        posting = (
            "Authorization to work in the United States is required."
        )
        parsed_job = ParsedJob(
            job_title="Support Specialist",
            company="Example",
            core_required=["Customer support experience"],
            required_competencies=[
                "Authorization to work in the United States"
            ],
        )
        requirements = [
            {
                "id": "C1",
                "category": "core",
                "requirement": "Customer support experience",
            },
            {
                "id": "K1",
                "category": "competency",
                "requirement": (
                    "Authorization to work in the United States"
                ),
            },
        ]
        matches = MatchResponse(
            matches=[
                MatchItem(
                    id="C1",
                    status="direct",
                    evidence="Customer support role",
                ),
                MatchItem(
                    id="K1",
                    status="missing",
                    evidence="Not authorized",
                ),
            ]
        )
        result = build_result(
            parsed_job,
            requirements,
            matches,
            [],
            posting,
            self.metadata(authorized=False),
        )

        self.assertEqual(result["match_score"], 100)
        self.assertEqual(len(result["required"]), 1)
        self.assertEqual(
            result["eligibility"]["items"][0]["status"],
            "mismatch",
        )
        self.assertFalse(
            result["eligibility"]["affects_qualification_score"]
        )


if __name__ == "__main__":
    unittest.main()
