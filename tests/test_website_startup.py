"""启动器的服务复用、端口冲突及环境回退检查。"""
import unittest
from unittest.mock import Mock, patch

import start_website


class WebsiteStartupTests(unittest.TestCase):
    def test_reuses_running_website_without_spawning(self):
        with patch.object(start_website, "website_ready", return_value=True), \
                patch.object(start_website.subprocess, "Popen") as spawn:
            self.assertFalse(start_website.ensure_website())
            spawn.assert_not_called()

    def test_does_not_replace_service_on_occupied_port(self):
        with patch.object(start_website, "website_ready", return_value=False), \
                patch.object(start_website, "port_in_use", return_value=True), \
                patch.object(start_website.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "5000"):
                start_website.ensure_website()
            spawn.assert_not_called()

    def test_health_response_must_identify_this_website(self):
        response = Mock()
        response.read.return_value = b'{"status": "ok", "service": "another-app"}'
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        with patch.object(start_website.OPENER, "open", return_value=context):
            self.assertFalse(start_website.website_ready())
            response.read.return_value = b'{"status": "ok", "service": "opd-web"}'
            self.assertTrue(start_website.website_ready())
            response.read.return_value = b'<html>Not a JSON API</html>'
            self.assertFalse(start_website.website_ready())

    def test_falls_back_to_usable_python_environment(self):
        failed = Mock(returncode=1, stderr="ModuleNotFoundError: flask", stdout="")
        working = Mock(returncode=0)
        with patch.dict(start_website.os.environ, {"OPD_PYTHON": "missing-deps-python"}), \
                patch.object(start_website.shutil, "which", side_effect=lambda value: value), \
                patch.object(start_website.subprocess, "run", side_effect=[failed, working]):
            self.assertEqual(start_website.find_python(), start_website.sys.executable)

    def test_missing_dependencies_give_actionable_error(self):
        with patch.object(start_website.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "requirements.txt"):
                start_website.find_python()


if __name__ == "__main__":
    unittest.main()
