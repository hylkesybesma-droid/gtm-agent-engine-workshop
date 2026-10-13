import json
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, get_prospect, score_prospect


class ProspectDataSafetyTests(unittest.TestCase):
    prospect_id = "LEAD-TEST"
    sensitive_values = {
        "tax_id": "541-88-2307",
        "date_of_birth": "1984-03-11",
        "card_on_file": "4539882100437712",
        "credit_check_ref": "EXPN-88213",
    }

    def setUp(self):
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        data_service._PROFILES.pop(self.prospect_id, None)

    def prospect_record(self):
        return {
            "name": "Omar Okafor",
            "email": "omar@example.com",
            "billing_qualification": self.sensitive_values.copy(),
            "disqualified": True,
            "engagement_history": [{"channel": "Inbound demo request"}],
            "account_details": [{"account": "Lakeside Analytics"}],
            "tech_stack": ["Snowflake", "AWS"],
            "annual_revenue": 90000000,
            "enrichment_source": "Apollo",
            "unapproved_field": "must not pass through",
        }

    def assert_sensitive_data_absent(self, value):
        serialized = json.dumps(value)
        self.assertNotIn("billing_qualification", serialized)
        for sensitive_value in self.sensitive_values.values():
            self.assertNotIn(sensitive_value, serialized)

    def test_get_prospect_returns_only_contact_allowlist(self):
        with patch.object(data_service, "get_prospect_record", return_value=self.prospect_record()):
            result = get_prospect.invoke({"prospect_id": self.prospect_id})

        self.assertEqual(result["prospect"], {
            "prospect_id": self.prospect_id,
            "name": "Omar Okafor",
            "email": "omar@example.com",
            "disqualified": True,
        })
        self.assert_sensitive_data_absent(result)

    def test_build_profile_keeps_required_fields_without_billing_data(self):
        record = self.prospect_record()
        with (
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=record["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=record["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=record["tech_stack"]),
        ):
            result = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        profile = result["prospect_profile"]
        self.assertEqual(profile["prospect_id"], self.prospect_id)
        self.assertEqual(profile["name"], "Omar Okafor")
        self.assertEqual(profile["email"], "omar@example.com")
        self.assertEqual(profile["disqualified"], True)
        self.assertEqual(profile["annual_revenue"], 90000000)
        self.assertEqual(profile["engagement_history"], record["engagement_history"])
        self.assertEqual(profile["account_details"], record["account_details"])
        self.assertEqual(profile["tech_stack"], record["tech_stack"])
        self.assertEqual(profile["enrichment_source"], "Apollo")
        self.assertNotIn("unapproved_field", profile)
        self.assertEqual(data_service._PROFILES[self.prospect_id], profile)
        self.assert_sensitive_data_absent(result)
        self.assert_sensitive_data_absent(data_service._PROFILES[self.prospect_id])

    def test_legacy_cache_hit_is_sanitized_and_updated(self):
        legacy_profile = {
            "prospect_id": self.prospect_id,
            "name": "Omar Okafor",
            "email": "omar@example.com",
            "disqualified": False,
            "annual_revenue": 90000000,
            "tech_stack": ["Snowflake", "AWS"],
            "billing_qualification": self.sensitive_values.copy(),
            "unapproved_field": "must not pass through",
        }
        data_service._PROFILES[self.prospect_id] = legacy_profile

        cached = data_service.get_profile_from_db(self.prospect_id)
        self.assert_sensitive_data_absent(cached)
        self.assertNotIn("billing_qualification", data_service._PROFILES[self.prospect_id])

        result = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        profile = result["prospect_profile"]
        self.assertEqual(profile["annual_revenue"], 90000000)
        self.assertEqual(profile["tech_stack"], ["Snowflake", "AWS"])
        self.assertNotIn("unapproved_field", profile)
        self.assertIs(data_service._PROFILES[self.prospect_id], profile)
        self.assert_sensitive_data_absent(result)
        self.assert_sensitive_data_absent(data_service._PROFILES[self.prospect_id])

    def test_score_prompt_omits_billing_data_and_keeps_scoring_inputs(self):
        llm = MagicMock()
        llm.invoke.return_value.model_dump.return_value = {"score": 80}
        profile = {
            "prospect_id": self.prospect_id,
            "name": "Omar Okafor",
            "annual_revenue": 90000000,
            "tech_stack": ["Snowflake", "AWS"],
            "billing_qualification": self.sensitive_values.copy(),
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 50000000,
            "description": "Managed analytics platform",
        }

        with patch("gtm_agent.gtm_agent._scoring_llm", llm), patch.object(
            data_service, "fetch_tech_stack", return_value=["Snowflake", "AWS"]
        ):
            score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        messages = llm.invoke.call_args.args[0]
        prompt = messages[1]["content"]
        self.assertIn('"annual_revenue": 90000000', prompt)
        self.assertIn('"tech_stack": [\n    "Snowflake",\n    "AWS"\n  ]', prompt)
        self.assert_sensitive_data_absent(prompt)


if __name__ == "__main__":
    unittest.main()
