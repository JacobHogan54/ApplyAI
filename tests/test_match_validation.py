import unittest

from agent import (
    MatchItem,
    calculate_score,
    validate_match_evidence,
)


RETAIL_AND_TECH_RESUME = """
Grocery Clerk at Publix. Provide customer service in a high-traffic
environment, answer customer questions, resolve routine issues, and
communicate with customers and teammates. Built Python projects and
performed PC hardware and software troubleshooting.
"""


class BloomerangMatchValidationTests(unittest.TestCase):

    def validate_direct(self, requirement, evidence, resume=None):
        match = MatchItem(
            id="K1",
            status="direct",
            evidence=evidence,
        )
        return validate_match_evidence(
            requirement,
            match,
            resume or RETAIL_AND_TECH_RESUME,
        )

    def test_customer_service_background_is_direct(self):
        match, reason = self.validate_direct(
            "A background in providing top-notch customer service",
            "Customer service at Publix",
        )

        self.assertEqual(match.status, "direct")
        self.assertIsNone(reason)

    def test_product_training_and_procedures_is_partial_without_evidence(self):
        match, reason = self.validate_direct(
            "Ability to apply product knowledge obtained in new hire "
            "training and daily activities, support procedures and policies",
            "Customer service at Publix",
        )

        self.assertEqual(match.status, "partial")
        self.assertIsNotNone(reason)

    def test_problem_tracking_is_not_direct_from_retail(self):
        retail_only_resume = (
            "Grocery Clerk at Publix providing customer service, "
            "answering questions, and helping customers."
        )
        match, reason = self.validate_direct(
            "Problem tracking skills to determine trends or patterns "
            "to client system problems",
            "Customer service at Publix",
            retail_only_resume,
        )

        self.assertIn(match.status, {"partial", "missing"})
        self.assertNotEqual(match.status, "direct")
        self.assertIsNotNone(reason)

    def test_written_and_verbal_is_partial_without_written_evidence(self):
        match, reason = self.validate_direct(
            "Superb written and verbal communication",
            "Customer service and communication at Publix",
        )

        self.assertEqual(match.status, "partial")
        self.assertIsNotNone(reason)

    def test_computer_and_software_troubleshooting_is_direct(self):
        match, reason = self.validate_direct(
            "Keen troubleshooting ability, and general comfort with "
            "computers and software technology",
            "PC and software troubleshooting",
        )

        self.assertEqual(match.status, "direct")
        self.assertIsNone(reason)

    def test_explicit_product_training_and_procedures_remains_direct(self):
        resume = (
            "Applied product knowledge after formal training and followed "
            "support procedures and policies during daily work."
        )
        match, reason = self.validate_direct(
            "Ability to apply product knowledge obtained in new hire "
            "training and daily activities, support procedures and policies",
            "Product knowledge, training, support procedures and policies",
            resume,
        )

        self.assertEqual(match.status, "direct")
        self.assertIsNone(reason)

    def test_revised_bloomerang_statuses_reduce_score(self):
        requirements = [
            {"id": "C1", "category": "core"},
            {"id": "K1", "category": "competency"},
            {"id": "K2", "category": "competency"},
            {"id": "K3", "category": "competency"},
            {"id": "K4", "category": "competency"},
        ]
        matches = [
            MatchItem(id="C1", status="direct", evidence="customer service"),
            MatchItem(id="K1", status="partial", evidence="transferable"),
            MatchItem(id="K2", status="partial", evidence="transferable"),
            MatchItem(id="K3", status="partial", evidence="verbal only"),
            MatchItem(id="K4", status="direct", evidence="troubleshooting"),
        ]

        self.assertEqual(
            calculate_score(requirements, matches),
            91,
        )


if __name__ == "__main__":
    unittest.main()
