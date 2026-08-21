import unittest

from fastapi.testclient import TestClient

from app.main import UI_BUILD_ID, create_app


class UiCacheContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(create_app())

    def test_root_redirect_preserves_relative_proxy_path(self):
        response = self.client.get("/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers.get("location"), "./ui/")
        self.assertEqual(response.headers.get("x-qa-ui-build"), UI_BUILD_ID)
        self.assertIn("no-store", response.headers.get("cache-control", ""))

        response = self.client.get("/ui", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers.get("location"), "./ui/")
        self.assertEqual(response.headers.get("x-qa-ui-build"), UI_BUILD_ID)

    def test_ui_entry_and_assets_share_no_cache_contract(self):
        for path in ("/ui/", "/ui/index.html", "/ui/app.js?v=test", "/ui/styles.css?v=test"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers.get("x-qa-ui-build"), UI_BUILD_ID)
                self.assertIn("no-store", response.headers.get("cache-control", ""))

        html = self.client.get("/ui/").text
        self.assertIn(f'content="{UI_BUILD_ID}"', html)
        self.assertIn(f"app.js?v={UI_BUILD_ID}", html)
        self.assertNotIn("兜底测试逻辑", html)

    def test_desktop_sidebar_matches_reference_scale(self):
        css = self.client.get("/ui/styles.css").text
        self.assertIn("--sidebar-width: 248px", css)
        self.assertIn("min-height: 112px", css)
        self.assertIn("min-height: 44px", css)
        self.assertIn("calc(100% - var(--sidebar-width))", css)
        self.assertNotIn("224px", css)


if __name__ == "__main__":
    unittest.main()
