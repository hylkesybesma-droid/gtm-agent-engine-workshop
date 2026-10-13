import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent.gtm_agent import send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def call_tool(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="A subject",
            body="A message",
            runtime=SimpleNamespace(config={}),
            from_rep={"email": "rep@example.com", "name": "Rep"},
        )

    def test_blocks_disqualified_prospect_without_sending(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "email": "priya.nair@brightwaveapps.com",
            "name": "Priya Nair",
        }

        with patch("gtm_agent.gtm_agent.uuid.uuid4") as generate_message_id:
            result = self.call_tool(prospect)

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )
        generate_message_id.assert_not_called()

    def test_sends_non_disqualified_prospect(self):
        prospect = {
            "prospect_id": "LEAD-ELIGIBLE",
            "email": "lead@example.com",
            "name": "Eligible Lead",
        }

        with patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value={"disqualified": False}):
            result = self.call_tool(prospect)

        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(
            {key: result[key] for key in ("to", "to_name", "from", "from_name", "subject", "body")},
            {
                "to": "lead@example.com",
                "to_name": "Eligible Lead",
                "from": "rep@example.com",
                "from_name": "Rep",
                "subject": "A subject",
                "body": "A message",
            },
        )

    def test_sends_prospect_without_id(self):
        prospect = {"email": "lead@example.com", "name": "Lead"}

        with patch("gtm_agent.gtm_agent.data_service.get_prospect_record") as get_prospect_record:
            result = self.call_tool(prospect)

        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], "lead@example.com")
        get_prospect_record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
