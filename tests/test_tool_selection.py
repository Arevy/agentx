import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import agent_cli
from agent_cli import CLIProgrammingAgent, CommandExecutor, DocFragment


class DummyAgent(CLIProgrammingAgent):
    """Lightweight stub to access _extract_actions without loading models."""

    def __init__(self):
        self._executed_global_signatures = set()


class ToolSelectionTests(unittest.TestCase):
    def setUp(self):
        self.agent = DummyAgent.__new__(DummyAgent)
        DummyAgent.__init__(self.agent)

    def test_directory_requests_prefer_run_command(self):
        llm_output = json.dumps(
            {
                "tool_calls": [
                    {"name": "run_command", "arguments": {"command": "ls -la"}},
                    {"name": "open_file", "arguments": {"path": "/workspace"}},
                ]
            }
        )

        actions = CLIProgrammingAgent._extract_actions(self.agent, llm_output)
        self.assertTrue(actions)
        tool, args = actions[0]
        self.assertEqual(tool, "run_command")
        self.assertTrue(args["command"].startswith("ls"))

    def test_duplicate_tool_calls_are_ignored(self):
        llm_output = json.dumps(
            {
                "tool_calls": [
                    {"name": "run_command", "arguments": {"command": "ls -la"}},
                    {"name": "run_command", "arguments": {"command": "ls -la"}},
                ]
            }
        )

        first_pass = CLIProgrammingAgent._extract_actions(self.agent, llm_output)
        self.assertEqual(len(first_pass), 1)
        tool, args = first_pass[0]
        self.assertEqual(tool, "run_command")

        signature = self.agent._action_signature(tool, args)
        self.agent._executed_global_signatures.add(signature)
        second_pass = CLIProgrammingAgent._extract_actions(self.agent, llm_output)
        self.assertEqual(second_pass, [])


class StubMemory:
    def __init__(self):
        self._notes: list[str] = []

    def append(self, note: str) -> None:
        self._notes.append(note)

    def snapshot(self, max_lines: int = 20) -> str:
        return "\n".join(self._notes[-max_lines:])

    def entries(self) -> list[str]:
        return self._notes


class StubVectorMemory:
    def __init__(self, search_results=None):
        self.metadata = []
        self._results = search_results or []

    def add(self, text: str, metadata: dict) -> None:
        self.metadata.append({"text": text, "metadata": metadata})

    def search(self, query: str, top_k: int = 5):
        return self._results[:top_k]


class StubRetriever:
    def __init__(self, docs=None):
        self._docs = docs or []

    def search(self, query: str, top_k: int = 5):
        return self._docs[:top_k]


class ToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        workspace = Path(self.tempdir.name)
        self.agent = CLIProgrammingAgent.__new__(CLIProgrammingAgent)
        self.agent.executor = CommandExecutor(workspace)
        self.agent.memory = StubMemory()
        self.agent.vector_memory = StubVectorMemory()
        self.agent.retriever = StubRetriever()
        self.agent.observation_summary_chars = 256
        self.agent.max_action_output_chars = 4000
        self.agent._log_audit = lambda *_, **__: None
        self.agent.config = type(
            "Cfg",
            (),
            {
                "docs_path": workspace,
                "workspace": workspace,
                "context7_url": None,
                "context7_api_key": None,
            },
        )()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_run_python_executes_simple_code(self):
        result = self.agent._dispatch_action("run_python", {"code": "print(1+1)"})
        self.assertIn("2", result)

    def test_run_python_blocks_dangerous_code(self):
        result = self.agent._dispatch_action("run_python", {"code": "import os"})
        self.assertIn("blocked", result.lower())

    def test_run_python_blocks_from_import(self):
        result = self.agent._dispatch_action("run_python", {"code": "from os import path"})
        self.assertIn("blocked", result.lower())

    def test_run_python_blocks_builtins_import(self):
        result = self.agent._dispatch_action("run_python", {"code": "import builtins"})
        self.assertIn("blocked", result.lower())

    def test_search_web_returns_local_matches(self):
        doc = DocFragment(path=Path("docs/sample.md"), text="List files with ls -la", score=10.0)
        self.agent.retriever = StubRetriever([doc])
        self.agent.vector_memory = StubVectorMemory(
            [{"text": "Remember to run_command for listings.", "metadata": {"source": "memory"}}]
        )
        output = self.agent._search_web("list files")
        self.assertIn("run_command", output)
        self.assertIn("memory", output)

    def test_search_web_fallbacks_to_context7(self):
        self.agent.retriever = StubRetriever([])
        self.agent.vector_memory = StubVectorMemory([])
        self.agent.config.context7_url = "https://context7.local"

        class DummyResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {"results": [{"text": "Doc fragment", "source": "ctx7"}]}

        with mock.patch.object(agent_cli.requests, "post", return_value=DummyResponse()), \
             mock.patch.object(agent_cli.requests, "get", side_effect=agent_cli.requests.RequestException("offline")):
            output = self.agent._search_web("nonexistent topic")
        self.assertIn("Doc fragment", output)

    def test_download_file_sanitizes_name(self):
        class DummyResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"Content-Length": "4"}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=8192):
                yield b"data"

            def close(self):
                return None

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch.object(agent_cli.requests, "get", return_value=DummyResponse()):
            result = self.agent._dispatch_action(
                "download_file",
                {"url": "https://example.com/test file.txt"},
            )
        self.assertIn("downloads", result)
        downloads_dir = self.agent.executor.workspace / "downloads"
        files = list(downloads_dir.glob("*"))
        self.assertTrue(files)
        self.assertNotIn(" ", files[0].name)
        self.assertTrue(files[0].read_bytes().startswith(b"data"))

    def test_download_file_retries_and_reports_failure(self):
        class FailingResponse:
            def __init__(self):
                self.status_code = 503

            def raise_for_status(self):
                raise agent_cli.requests.HTTPError("503 Server Error")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch.object(agent_cli.requests, "get", side_effect=[FailingResponse(), FailingResponse(), FailingResponse()]):
            result = self.agent._dispatch_action("download_file", {"url": "https://example.com/file.txt"})
        self.assertIn("failed to download", result.lower())

    def test_download_file_blocks_large_content_length(self):
        class LargeResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"Content-Length": str(agent_cli.DOWNLOAD_MAX_BYTES + 1)}

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=8192):
                yield b"data"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        with mock.patch.object(agent_cli.requests, "get", return_value=LargeResponse()):
            result = self.agent._dispatch_action("download_file", {"url": "https://example.com/large.bin"})
        self.assertIn("5mb limit", result.lower())

    def test_call_context7_success(self):
        self.agent.config.context7_url = "https://context7.local"

        class DummyResponse:
            def __init__(self):
                self.status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "results": [
                        {"text": "Function foo does bar", "source": "docs/api"},
                        {"text": "Class Baz handles qux", "source": "docs/api"},
                    ]
                }

        with mock.patch.object(agent_cli.requests, "post", return_value=DummyResponse()):
            result = self.agent._dispatch_action("call_context7", {"query": "foo"})
        self.assertIn("Function foo", result)

    def test_call_context7_no_config(self):
        self.agent.config.context7_url = None
        result = self.agent._dispatch_action("call_context7", {"query": "foo"})
        self.assertIn("context7 unavailable", result)

    def test_call_context7_handles_http_error(self):
        self.agent.config.context7_url = "https://context7.local"

        class ErrorResponse:
            def raise_for_status(self):
                raise agent_cli.requests.HTTPError("500 Server Error")

        with mock.patch.object(agent_cli.requests, "post", return_value=ErrorResponse()):
            result = self.agent._dispatch_action("call_context7", {"query": "foo"})
        self.assertIn("request failed", result.lower())


if __name__ == "__main__":
    unittest.main()
