import unittest

from app.services.llm.vlm_client import normalize_lmp_cloud_endpoint


class LmpCloudEndpointNormalizationTests(unittest.TestCase):
    def test_host_root_uses_default_lmp_cloud_endpoint(self):
        self.assertEqual(
            normalize_lmp_cloud_endpoint("http://llm-gateway:8080"),
            "http://llm-gateway:8080/lmp-cloud-ias-server/api/vlm/chat/completions",
        )

    def test_lmp_cloud_service_root_appends_vlm_endpoint(self):
        self.assertEqual(
            normalize_lmp_cloud_endpoint("http://llm-gateway:8080/lmp-cloud-ias-server"),
            "http://llm-gateway:8080/lmp-cloud-ias-server/api/vlm/chat/completions",
        )

    def test_vlm_root_appends_chat_completions(self):
        self.assertEqual(
            normalize_lmp_cloud_endpoint("http://llm-gateway:8080/api/vlm"),
            "http://llm-gateway:8080/api/vlm/chat/completions",
        )

    def test_custom_complete_endpoint_is_used_as_is(self):
        self.assertEqual(
            normalize_lmp_cloud_endpoint("http://llm-gateway:8080/V2"),
            "http://llm-gateway:8080/V2",
        )
        self.assertEqual(
            normalize_lmp_cloud_endpoint("http://llm-gateway:8080/custom/lmp/chat"),
            "http://llm-gateway:8080/custom/lmp/chat",
        )

    def test_any_chat_completions_endpoint_is_used_as_is(self):
        self.assertEqual(
            normalize_lmp_cloud_endpoint("http://llm-gateway:8080/v1/chat/completions"),
            "http://llm-gateway:8080/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
