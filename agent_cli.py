#!/usr/bin/env python3
"""
Local CLI Programming Agent powered by Hugging Face transformers.

Features
--------
* Loads a causal language model (default: gpt2-medium) for instruction following.
* Maintains a lightweight long-term memory in `memory_bank/global.md`.
* Retrieves supporting context from Markdown docs stored in `docs/`.
* Supports tool use by parsing LLM-suggested actions such as:
    - Action: run_command("ls")
    - Action: open_file("path/to/file.py")
    - Action: store_memory("Remember to...")
* Executes in a Reason -> Act -> Observe loop until the model produces a final reply.
* Blocks destructive shell commands for safety.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import contextlib
from urllib.parse import urlparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except ImportError:  # pragma: no cover - optional dependency
    BitsAndBytesConfig = None

from vector_memory import VectorMemory

SAFE_PYTHON_DENYLIST = [
    "import os",
    "from os import",
    "import subprocess",
    "from subprocess import",
    "subprocess",
    "__import__",
    "__builtins__",
    "builtins.",
    "import builtins",
    "from builtins import",
    "import sys",
    "from sys import",
    "open(",
    "os.system",
    "import importlib",
    "from importlib import",
    "socket",
    "requests",
    "urllib",
    "shutil",
    "sys.exit",
    "eval(",
    "exec(",
    "os.",
    "sys.modules",
    "globals(",
    "locals(",
    "compile(",
    "input(",
]
DOWNLOAD_MAX_BYTES = 5 * 1024 * 1024  # 5 MB safety cap
DEFAULT_CONTEXT7_API_KEY = "ctx7sk-840f286d-465e-4343-aadb-b4453dbcbae3"

logger = logging.getLogger("agent_cli")


def utcnow_iso() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")


DEFAULT_SYSTEM_PROMPT = (
    "You are a local CLI programming assistant. Follow the Reason→Act→Observe loop strictly. "
    "When you need a tool, emit exactly one JSON object like:\n"
    "{\n"
    '  "tool_calls": [{"name": "...", "arguments": {...}}]\n'
    "}\n"
    "Tools:\n"
    "- run_command  → shell tasks (e.g. {\"command\": \"ls -la\"}). Use for listings or scripts.\n"
    "- open_file    → read an existing file inside the workspace (relative path only; never directories).\n"
    "- store_memory → append important notes (provide {\"note\": \"...\"}).\n"
    "- run_python   → execute small Python snippets safely ({\"code\": \"print(42)\"}).\n"
    "- search_web   → query local docs/memory (\"{\\\"query\\\": \\\"...\\\", \\\"k\\\": 5}\").\n"
    "- call_context7 → request documentation fragments via context7 when locals are insufficient.\n"
    "- install_package → install Python packages ({\"package\": \"fastapi\"}).\n"
    "- download_file → fetch HTTPS resources into downloads/ ({\"url\": \"https://...\"}).\n"
    "Never repeat a previously executed tool_call, never run open_file on directories, and respond concisely after each observation."
)


@dataclass
class AgentConfig:
    model_name_or_path: str = "gpt2-medium"
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    history_turns: int = 4
    action_retries: int = 3
    docs_path: Path = Path("docs")
    memory_file: Path = Path("memory_bank/global.md")
    memory_vector_dir: Path = Path("memory_bank/vector_store")
    workspace: Path = Path(".")
    max_action_output_chars: int = 4000
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
    enable_thinking: Optional[bool] = None
    context_window_tokens: Optional[int] = None
    quantization: str = "none"
    max_actions_per_turn: int = 1
    observation_summary_chars: int = 256
    audit_log: Optional[Path] = None
    context7_url: Optional[str] = None
    context7_api_key: Optional[str] = None


class MemoryBank:
    def __init__(self, memory_file: Path):
        self.path = memory_file
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("# Global Memory\n", encoding="utf-8")

    def snapshot(self, max_lines: int = 20) -> str:
        content = self.path.read_text(encoding="utf-8")
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return "No stored memories yet."
        return "\n".join(lines[-max_lines:])

    def append(self, note: str) -> None:
        timestamp = utcnow_iso()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(f"- {timestamp} – {note.strip()}\n")
        logger.info("Stored memory entry.")

    def entries(self) -> List[str]:
        try:
            content = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        return [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("#")]


@dataclass
class DocFragment:
    path: Path
    text: str
    score: float


class DocumentationRetriever:
    def __init__(self, docs_dir: Path):
        self.docs_dir = docs_dir
        self.documents: List[Tuple[Path, str]] = []
        if docs_dir.exists():
            for file in sorted(docs_dir.rglob("*.md")):
                try:
                    self.documents.append((file, file.read_text(encoding="utf-8")))
                except UnicodeDecodeError:
                    logger.warning("Skipping non-UTF8 doc: %s", file)
        else:
            logger.info("Docs directory %s does not exist; continuing without docs.", docs_dir)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z]+", text.lower())

    def search(self, query: str, top_k: int = 3) -> List[DocFragment]:
        if not self.documents or not query.strip():
            return []
        query_terms = set(self._tokenize(query))
        fragments: List[DocFragment] = []
        for path, text in self.documents:
            doc_terms = self._tokenize(text)
            overlap = len(query_terms.intersection(doc_terms))
            if overlap == 0:
                continue
            preview = "\n".join(text.splitlines()[:40])
            fragments.append(DocFragment(path=path, text=preview, score=float(overlap)))
        fragments.sort(key=lambda f: f.score, reverse=True)
        return fragments[:top_k]


class CommandExecutor:
    BLOCKED = {
        "rm",
        "shutdown",
        "reboot",
        "poweroff",
        "halt",
        "mkfs",
        "dd",
        "killall",
        ":(){",
    }

    def __init__(self, workspace: Path):
        self.workspace = workspace

    def run(self, command_line: str) -> str:
        tokens = shlex.split(command_line)
        if not tokens:
            return "Command was empty."
        if tokens[0] in self.BLOCKED:
            return f"Blocked command: {tokens[0]}"
        logger.info("Executing command: %s", command_line)
        process = subprocess.run(
            command_line,
            cwd=self.workspace,
            shell=True,
            capture_output=True,
            text=True,
        )
        output = process.stdout.strip()
        error = process.stderr.strip()
        response = output
        if error:
            response = f"{response}\nSTDERR:\n{error}" if response else f"STDERR:\n{error}"
        if not response:
            response = "(no output)"
        return response

    def open_file(self, file_path: str) -> str:
        path = (self.workspace / file_path).resolve()
        if not str(path).startswith(str(self.workspace.resolve())):
            return "Access denied: file must be inside the workspace."
        if not path.exists():
            return f"File not found: {file_path}"
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "File is not UTF-8 text."
        return content[:8000]


ACTION_PATTERN = re.compile(
    r'Action:\s*(?P<name>run_command|open_file|store_memory|search_web|call_context7|install_package|run_python|download_file)\((?P<args>.*?)\)',
    re.IGNORECASE | re.DOTALL,
)


def _parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    normalized = name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[normalized]


class TransformersBackend:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        extra_kwargs: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.extra_kwargs = extra_kwargs or {}

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if hasattr(self.model, "hf_device_map"):
            target_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(target_device, str) and target_device != "disk":
                target_device = torch.device(target_device)
            if isinstance(target_device, torch.device):
                inputs = inputs.to(target_device)
        else:
            inputs = inputs.to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                **self.extra_kwargs,
            )
        decoded = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return decoded.strip()


class CLIProgrammingAgent:
    def __init__(
        self,
        config: AgentConfig,
        model: AutoModelForCausalLM,
        tokenizer: Optional[AutoTokenizer],
        memory: MemoryBank,
        vector_memory: VectorMemory,
        retriever: DocumentationRetriever,
        executor: CommandExecutor,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory
        self.vector_memory = vector_memory
        self.retriever = retriever
        self.executor = executor
        self.llm_backend = None
        self.history: List[Tuple[str, str]] = []
        self.system_prompt = config.system_prompt
        self.max_context_tokens = config.context_window_tokens
        self.max_actions_per_turn = max(1, config.max_actions_per_turn)
        self.observation_summary_chars = max(32, config.observation_summary_chars)
        self.audit_log_path = config.audit_log
        if self.audit_log_path:
            try:
                self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("Unable to create audit log directory: %s", exc)
                self.audit_log_path = None
        if self.max_context_tokens and self.tokenizer:
            try:
                self.tokenizer.model_max_length = self.max_context_tokens
            except Exception:
                logger.warning("Tokenizer does not allow overriding model_max_length to %s", self.max_context_tokens)
            if self.model is not None and hasattr(self.model.config, "max_position_embeddings"):
                self.model.config.max_position_embeddings = max(
                    getattr(self.model.config, "max_position_embeddings", 0),
                    self.max_context_tokens,
                )
        self.extra_generate_kwargs = {}
        if config.enable_thinking:
            self.extra_generate_kwargs["enable_thinking"] = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is not None and not hasattr(self.model, "hf_device_map"):
            self.model.to(self.device)
        self.llm_backend = TransformersBackend(self.model, self.tokenizer, self.device, self.extra_generate_kwargs)
        self._last_action_signature: Optional[str] = None
        self._executed_global_signatures: set[str] = set()
        if not self.vector_memory.metadata:
            for entry in self.memory.entries():
                self.vector_memory.add(entry, {"source": "memory_bank"})
        self._index_repository()

    def _format_history(self) -> str:
        turns = self.history[- self.config.history_turns * 2 :]
        return "\n".join(f"{role}: {text}" for role, text in turns)

    def _format_docs(self, query: str) -> str:
        fragments = self.retriever.search(query)
        if not fragments:
            return "No relevant local documentation found."
        lines = []
        for fragment in fragments:
            lines.append(f"[{fragment.path} | score={fragment.score:.0f}]")
            lines.append(fragment.text.strip())
        return "\n".join(lines)

    def _build_prompt(self, user_input: str) -> str:
        memory_snapshot = self.memory.snapshot(max_lines=10)
        vector_hits: List[str] = []
        vector_hit_canonicals: Set[str] = set()
        if hasattr(self, "vector_memory"):
            hits = self.vector_memory.search(user_input, top_k=3)
            if hits:
                for item in hits:
                    snippet = (item["text"] or "").replace("\n", " ").strip()
                    if not snippet:
                        continue
                    source = item.get("metadata", {}).get("source") or item.get("metadata", {}).get("role", "memory")
                    trimmed = snippet[:200]
                    vector_hits.append(f"[{source}] {trimmed}")
                    vector_hit_canonicals.add(trimmed.lower())
        docs_block = self._format_docs(user_input)
        context7_block: Optional[str] = None
        need_context7 = not vector_hits and docs_block == "No relevant local documentation found."
        if need_context7 and self.config.context7_url:
            ctx7_entries, ctx7_error = self._fetch_context7(user_input, top_k=3)
            if ctx7_entries:
                lines: List[str] = []
                for entry in ctx7_entries:
                    raw_text = (entry.get("text") or "").replace("\n", " ").strip()
                    snippet = raw_text[:256]
                    if len(raw_text) > 256:
                        snippet += "..."
                    if not snippet:
                        continue
                    canonical = raw_text[:200].lower()
                    if canonical in vector_hit_canonicals:
                        continue
                    vector_hit_canonicals.add(canonical)
                    lines.append(f"[{entry.get('source', 'context7')}] {snippet}")
                if lines:
                    context7_block = "\n".join(lines)
                    if len(context7_block) > 800:
                        context7_block = context7_block[:797] + "..."
            elif ctx7_error:
                context7_block = ctx7_error[:800]
        history_block = self._format_history()
        sections: List[str] = []
        if self.system_prompt:
            sections.append(f"System: {self.system_prompt}")
        memory_lines = [memory_snapshot]
        if vector_hits:
            memory_lines.append("Vector memory suggestions:")
            memory_lines.extend(vector_hits)
        sections.append("Memory:\n" + "\n".join(memory_lines))
        if context7_block:
            sections.append(f"Context7:\n{context7_block}")
        sections.append(f"Documentation:\n{docs_block}")
        if history_block:
            sections.append(f"Recent conversation:\n{history_block}")
        sections.append(f"User: {user_input}")
        sections.append("Assistant:")
        prompt = "\n\n".join(sections)
        logger.debug("Prompt built with %d characters", len(prompt))
        return prompt

    def _generate(self, prompt: str) -> str:
        raw = self.llm_backend.generate(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return self._post_process(raw)

    @staticmethod
    def _post_process(text: str) -> str:
        cleaned = text.strip()
        # Remove repeated prompt echoes.
        cleaned = re.sub(r"(?:System:.*\n?)+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:Memory:.*\n?)+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:Documentation:.*\n?)+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:Recent conversation:.*\n?)+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"(?:User:.*\n?)+", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        if not cleaned:
            cleaned = "Nu am gasit un raspuns util pe baza contextului curent."
        return cleaned

    def _find_json_blocks(self, text: str) -> List[dict]:
        blocks: List[dict] = []
        if "tool_calls" not in text and "function_call" not in text:
            return blocks
        decoder = json.JSONDecoder()
        idx = 0
        while idx < len(text):
            start = text.find("{", idx)
            if start == -1:
                break
            try:
                data, offset = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                idx = start + 1
                continue
            if isinstance(data, dict):
                blocks.append(data)
            idx = start + offset
        return blocks

    def _extract_actions(self, text: str) -> List[Tuple[str, object]]:
        actions: List[Tuple[str, object]] = []
        seen: set[str] = set()
        # JSON tool calls (Qwen, Devstral style)
        for obj in self._find_json_blocks(text):
            tool_calls = []
            if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                tool_calls = obj["tool_calls"]
            elif "function_call" in obj:
                tool_calls = [obj["function_call"]]
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = call.get("name") or call.get("tool_name")
                arguments = call.get("arguments", call.get("params", {}))
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                if name:
                    signature = self._action_signature(name, arguments)
                    if signature in seen or signature in self._executed_global_signatures:
                        continue
                    seen.add(signature)
                    actions.append((name.lower(), arguments))
            if tool_calls:
                return actions

        # Backward-compatible Action: lines
        for line in text.splitlines():
            match = ACTION_PATTERN.search(line.strip())
            if not match:
                continue
            name = match.group("name").lower()
            args_raw = match.group("args").strip()
            try:
                parsed_args = json.loads(f"[{args_raw}]")
            except json.JSONDecodeError:
                parsed_args = [args_raw.strip().strip('"').strip("'")]
            argument = parsed_args[0] if parsed_args else ""
            signature = self._action_signature(name, argument)
            if signature in seen or signature in self._executed_global_signatures:
                continue
            seen.add(signature)
            actions.append((name, argument))
        return actions

    @staticmethod
    def _action_signature(name: str, argument: object) -> str:
        if isinstance(argument, dict):
            normalized = json.dumps(argument, sort_keys=True)
        else:
            normalized = str(argument)
        return f"{name.lower()}::{normalized}"

    @staticmethod
    def _dedupe_consecutive_lines(text: str) -> str:
        lines = text.splitlines()
        deduped: List[str] = []
        prev: Optional[str] = None
        for line in lines:
            canonical = line.strip()
            if prev is not None and canonical and canonical == prev:
                continue
            deduped.append(line)
            prev = canonical if canonical else prev
        return "\n".join(deduped)

    @staticmethod
    def _sanitize_filename(name: str, fallback: str = "downloaded.bin") -> str:
        if not name:
            return fallback
        cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        return cleaned or fallback

    def _search_web(self, query: str, top_k: int = 5) -> str:
        query = (query or "").strip()
        if not query:
            return "search_web requires a non-empty query."
        top_k = max(1, min(top_k, 10))
        results: List[str] = []
        external_msgs: List[str] = []
        try:
            headers = {"User-Agent": "CLI-Agent/1.0"}
            resp = requests.get(
                "https://duckduckgo.com/ac/",
                params={"q": query},
                headers=headers,
                timeout=5,
            )
            resp.raise_for_status()
            suggestions = resp.json()
            if suggestions:
                results.append("Web suggestions:")
                for item in suggestions[:top_k]:
                    phrase = item.get("phrase")
                    if phrase:
                        results.append(f"- {phrase}")
        except requests.RequestException as exc:
            external_msgs.append(f"External search unavailable: {exc}")
        except ValueError:
            external_msgs.append("External search returned invalid JSON.")
        doc_hits = self.retriever.search(query, top_k=top_k)
        if doc_hits:
            results.append("Local documentation matches:")
            for hit in doc_hits:
                snippet = hit.text.replace("\n", " ")[:200]
                results.append(f"- {hit.path} (score {hit.score:.0f}): {snippet}")
        memory_hits = []
        if hasattr(self, "vector_memory"):
            memory_hits = self.vector_memory.search(query, top_k=top_k)
            if memory_hits:
                results.append("Memory matches:")
                for hit in memory_hits:
                    snippet = hit["text"].replace("\n", " ")[:200]
                    source = hit.get("metadata", {}).get("source", "memory")
                    results.append(f"- [{source}] {snippet}")
        had_matches = bool(results)
        if self.config.context7_url and not had_matches:
            ctx7_output = self._call_context7(query, top_k)
            if ctx7_output:
                results.append(ctx7_output)
                had_matches = True
        if not results:
            if external_msgs:
                results.extend(external_msgs)
            else:
                results.append(
                    "No results found locally. Integrate an external search provider (e.g. DuckDuckGo API) "
                    "or ensure network access is available."
                )
        else:
            results.extend(external_msgs)
        return "\n".join(results)

    def _fetch_context7(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, str]], Optional[str]]:
        query = (query or "").strip()
        if not query:
            return [], "call_context7 requires a non-empty query."
        url = (self.config.context7_url or os.getenv("CONTEXT7_URL") or "").strip()
        if not url:
            return [], "context7 unavailable: set CONTEXT7_URL or provide --context7-url."
        headers = {"Content-Type": "application/json"}
        api_key = self.config.context7_api_key or os.getenv("CONTEXT7_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {"query": query, "k": max(1, min(top_k, 5))}
        endpoint = url.rstrip("/") + "/v1/query"
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            return [], f"context7 request failed: {exc}"
        except ValueError:
            return [], "context7 returned invalid JSON."
        results = data.get("results") or data.get("passages") or []
        entries: List[Dict[str, str]] = []
        for item in results[: payload["k"]]:
            text = (item.get("text") or item.get("content") or "").strip()
            if not text:
                continue
            source = item.get("source") or item.get("metadata", {}).get("source") or "context7"
            entries.append({"text": text, "source": source})
        if not entries:
            return [], "context7 returned no matches."
        return entries, None

    def _call_context7(self, query: str, top_k: int = 5) -> str:
        entries, error = self._fetch_context7(query, top_k)
        if error:
            return error
        lines: List[str] = []
        for entry in entries:
            text = entry["text"].replace("\n", " ").strip()
            if not text:
                continue
            truncated = text[:200]
            if len(text) > 200:
                truncated += "..."
            lines.append(f"[{entry['source']}] {truncated}")
        return "\n".join(lines) if lines else "context7 returned empty passages."

    def _install_package(self, package: str) -> str:
        package = (package or "").strip()
        if not package:
            return "install_package requires a package name."
        log_dir = Path("logs/install_logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", package)
        log_file = log_dir / f"{utcnow_iso().replace(':', '-')}_{safe_name}.log"
        env = os.environ.copy()
        env.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
        process = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            cwd=self.executor.workspace,
            capture_output=True,
            text=True,
            env=env,
        )
        output = process.stdout.strip()
        error = process.stderr.strip()
        combined = output
        if error:
            combined = f"{combined}\nSTDERR:\n{error}" if combined else f"STDERR:\n{error}"
        try:
            log_file.write_text(combined, encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to write install log: %s", exc)
        if process.returncode != 0:
            message = (
                f"pip install failed (exit {process.returncode}). Check {log_file}. "
                "Verify network access and package name."
            )
        else:
            excerpt = combined[: self.observation_summary_chars] or "(no output)"
            message = f"pip install succeeded. Log saved to {log_file}. Output excerpt:\n{excerpt}"
        return message

    def _run_python(self, argument: object) -> str:
        if isinstance(argument, dict):
            code = argument.get("code") or argument.get("source") or ""
        else:
            code = str(argument)
        code = code.strip()
        if not code:
            return "run_python requires Python code in the 'code' field."
        lowered = code.lower()
        for forbidden in SAFE_PYTHON_DENYLIST:
            if forbidden in lowered:
                return f"run_python blocked: usage of '{forbidden}' is not permitted for safety."
        if len(code) > 4000:
            return "run_python code too long (limit 4000 characters)."
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": "",
            }
        )
        process = subprocess.run(
            [sys.executable, "-"],
            input=code,
            text=True,
            capture_output=True,
            cwd=self.executor.workspace,
            env=env,
        )
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        if process.returncode != 0:
            message = f"Python execution failed (exit {process.returncode})."
            if stderr:
                message += f" STDERR:\n{stderr[: self.observation_summary_chars]}"
            return message
        if not stdout:
            stdout = "(no output)"
        return f"Python execution output:\n{stdout[: self.observation_summary_chars]}"

    def _download_file(self, argument: object) -> str:
        if isinstance(argument, dict):
            url = argument.get("url") or argument.get("href") or ""
        else:
            url = str(argument)
        url = url.strip()
        if not url:
            return "download_file requires a 'url'."
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return "Only HTTP/HTTPS URLs are allowed for download_file."
        downloads_dir = self.executor.workspace / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        original_name = os.path.basename(parsed.path) or "downloaded.bin"
        filename = self._sanitize_filename(original_name)
        base, ext = os.path.splitext(filename)
        if not base:
            base = "download"
        target_path = downloads_dir / f"{base}{ext}"
        counter = 1
        while target_path.exists():
            candidate = f"{base}_{counter}{ext}"
            target_path = downloads_dir / candidate
            counter += 1
        last_error = None
        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                with requests.get(url, stream=True, timeout=15) as resp:
                    resp.raise_for_status()
                    content_length = resp.headers.get("Content-Length")
                    if content_length:
                        try:
                            if int(content_length) > DOWNLOAD_MAX_BYTES:
                                return (
                                    "Download aborted: file exceeds 5MB limit "
                                    f"(Content-Length: {content_length} bytes)."
                                )
                        except ValueError:
                            pass
                    size = 0
                    with target_path.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            size += len(chunk)
                            if size > DOWNLOAD_MAX_BYTES:
                                fh.close()
                                target_path.unlink(missing_ok=True)
                                return "Download aborted: file exceeds 5MB limit."
                            fh.write(chunk)
                return f"File downloaded to {target_path}"
            except requests.RequestException as exc:
                last_error = exc
                if attempt < attempts:
                    time.sleep(min(2, attempt))  # brief backoff
                    continue
            except OSError as exc:
                last_error = exc
            break
        if target_path.exists():
            with contextlib.suppress(OSError):
                target_path.unlink()
        return f"Failed to download file after {attempts} attempts: {last_error}"

    def _log_audit(self, event: str, payload: dict) -> None:
        if not self.audit_log_path:
            return
        record = {
            "timestamp": utcnow_iso(),
            "event": event,
            **payload,
        }
        logger.debug("Audit event: %s", record)
        try:
            with self.audit_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Failed to write audit log: %s", exc)

    def _dispatch_action(self, name: str, argument: object) -> str:
        if name == "run_command":
            if isinstance(argument, dict):
                command = argument.get("command") or argument.get("cmd") or ""
            else:
                command = str(argument)
            self._log_audit("tool_call", {"name": "run_command", "arguments": command})
            return self.executor.run(command)
        if name == "open_file":
            if isinstance(argument, dict):
                path = argument.get("path") or argument.get("file") or ""
            else:
                path = str(argument)
            self._log_audit("tool_call", {"name": "open_file", "arguments": path})
            return self.executor.open_file(path)
        if name == "store_memory":
            if isinstance(argument, dict):
                note = argument.get("note") or argument.get("text") or ""
            else:
                note = str(argument)
            self.memory.append(note)
            if hasattr(self, "vector_memory"):
                self.vector_memory.add(note, {"role": "Memory"})
            self._log_audit("tool_call", {"name": "store_memory", "arguments": note})
            return f"Stored memory: {note}"
        if name == "search_web":
            if isinstance(argument, dict):
                query = argument.get("query") or argument.get("text") or ""
                k = int(argument.get("k", 5))
            else:
                query = str(argument)
                k = 5
            self._log_audit("tool_call", {"name": "search_web", "arguments": {"query": query, "k": k}})
            return self._search_web(query, k)
        if name == "call_context7":
            if isinstance(argument, dict):
                query = argument.get("query") or argument.get("text") or ""
                k = int(argument.get("k", 5))
            else:
                query = str(argument)
                k = 5
            self._log_audit("tool_call", {"name": "call_context7", "arguments": {"query": query, "k": k}})
            return self._call_context7(query, k)
        if name == "install_package":
            if isinstance(argument, dict):
                package = argument.get("package") or argument.get("name") or ""
            else:
                package = str(argument)
            self._log_audit("tool_call", {"name": "install_package", "arguments": package})
            return self._install_package(package)
        if name == "run_python":
            return self._run_python(argument)
        if name == "download_file":
            return self._download_file(argument)
        message = f"Unknown action: {name}"
        self._log_audit("tool_call", {"name": name, "arguments": argument, "error": message})
        return message

    def interact(self, user_input: str) -> str:
        self.history.append(("User", user_input))
        if hasattr(self, "vector_memory"):
            self.vector_memory.add(user_input, {"role": "User"})
        loops = 0
        prompt = self._build_prompt(user_input)
        reply = self._generate(prompt)
        executed_signatures: set[str] = set()
        self._last_action_signature = None

        while loops < self.config.action_retries:
            actions = self._extract_actions(reply)
            logger.debug("Extracted actions: %s", actions)
            if not actions:
                break
            chosen: Optional[Tuple[str, object, str]] = None
            for name, argument in actions:
                signature = self._action_signature(name, argument)
                if signature in executed_signatures or signature == self._last_action_signature:
                    continue
                chosen = (name, argument, signature)
                break
            if not chosen:
                break
            name, argument, signature = chosen
            executed_signatures.add(signature)
            self._executed_global_signatures.add(signature)
            if len(executed_signatures) > self.max_actions_per_turn:
                break
            self.history.append(("Assistant", reply))
            if hasattr(self, "vector_memory"):
                self.vector_memory.add(reply, {"role": "Assistant-draft"})
            observation = self._dispatch_action(name, argument)
            observation = observation[: self.config.max_action_output_chars]
            logger.info("Observation: %s", observation)
            self._log_audit("observation", {"name": name, "content": observation})
            summary = observation
            if len(summary) > self.observation_summary_chars:
                summary = summary[: self.observation_summary_chars - 3] + "..."
            self.history.append(("Observation", summary))
            if hasattr(self, "vector_memory"):
                self.vector_memory.add(summary, {"role": "Observation", "tool": name})
            prompt = self._build_prompt(f"Observation: {summary}")
            reply = self._generate(prompt)
            loops += 1
            self._last_action_signature = signature
            if len(executed_signatures) >= self.max_actions_per_turn:
                break

        final_reply = reply
        obs_summary = next((text for role, text in reversed(self.history) if role == "Observation"), None)
        logger.debug("Final reply pre-normalisation: %s", final_reply)
        logger.debug("Last observation summary: %s", obs_summary)
        if "tool_calls" in final_reply and self._last_action_signature and obs_summary:
            final_reply = f"Observation handled: {obs_summary}"
        elif final_reply.lstrip().lower().startswith("observation:") and obs_summary:
            final_reply = obs_summary
        final_reply = self._dedupe_consecutive_lines(final_reply)
        obs_marker = final_reply.lower().find("observation:")
        if final_reply.lower().count("observation:") > 1 and obs_marker != -1:
            second = final_reply.lower().find("observation:", obs_marker + 1)
            if second != -1:
                final_reply = final_reply[:second].strip()
        logger.debug("Final reply post-normalisation: %s", final_reply)
        self.history.append(("Assistant", final_reply))
        if hasattr(self, "vector_memory"):
            self.vector_memory.add(final_reply, {"role": "Assistant"})
        return final_reply

    def _index_repository(self) -> None:
        try:
            existing_paths = {
                item.get("metadata", {}).get("path")
                for item in getattr(self.vector_memory, "metadata", [])
                if isinstance(item, dict)
            }
        except Exception:
            existing_paths = set()
        new_entries = 0

        def add_path(path: Path, source: str) -> None:
            nonlocal new_entries
            str_path = str(path)
            if str_path in existing_paths:
                return
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                return
            snippet = text[:2000]
            if not snippet.strip():
                return
            try:
                self.vector_memory.add(snippet, {"source": source, "path": str_path})
                existing_paths.add(str_path)
                new_entries += 1
            except Exception as exc:
                logger.debug("Failed to index %s: %s", path, exc)

        if isinstance(self.vector_memory, VectorMemory):
            if self.config.docs_path.exists():
                for path in sorted(self.config.docs_path.rglob("*.md"))[:100]:
                    add_path(path, "doc")
            code_exts = {".py", ".js", ".ts", ".tsx", ".md", ".rst"}
            scanned = 0
            for path in self.config.workspace.rglob("*"):
                if scanned >= 200:
                    break
                if path.is_file() and path.suffix in code_exts:
                    add_path(path, "code")
                    scanned += 1
        if new_entries:
            logger.debug("Indexed %d new fragments into vector memory.", new_entries)


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device_map: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
    low_cpu_mem_usage: bool = False,
    quantization: str = "none",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading tokenizer and model from %s", model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    load_kwargs = {}
    if device_map:
        load_kwargs["device_map"] = device_map
    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    if low_cpu_mem_usage:
        load_kwargs["low_cpu_mem_usage"] = True
    if quantization != "none":
        if BitsAndBytesConfig is None:
            raise RuntimeError("bitsandbytes is required for quantized loading but is not installed.")
        if quantization == "8bit":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16 if torch_dtype is None else torch_dtype,
                bnb_4bit_quant_type="nf4",
            )
        load_kwargs["quantization_config"] = quant_config
        load_kwargs.setdefault("device_map", "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **load_kwargs,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local CLI programming agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="gpt2-medium", help="Model name or path (local fine-tuned directories supported).")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum tokens to generate each turn.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling value.")
    parser.add_argument("--docs", type=Path, default=Path("docs"), help="Directory with Markdown reference docs.")
    parser.add_argument("--memory", type=Path, default=Path("memory_bank/global.md"), help="Path to the persistent memory file.")
    parser.add_argument("--memory-vector-dir", type=Path, default=Path("memory_bank/vector_store"), help="Directory for FAISS vector memory.")
    parser.add_argument("--workspace", type=Path, default=Path("."), help="Workspace root for file access and commands.")
    parser.add_argument("--once", type=str, help="Run a single turn with the provided prompt and exit.")
    parser.add_argument("--remember", type=str, help="Append a memory note and exit.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING...).")
    parser.add_argument("--device-map", type=str, default=None, help="Device map hint for loading large models (e.g. 'auto', 'cpu', 'mps', 'cuda').")
    parser.add_argument("--torch-dtype", type=str, default=None, help="Override torch dtype when loading model (e.g. float16, bfloat16).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow execution of remote code while loading custom models.")
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Use low CPU memory loading path (recommended for very large models).")
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none", help="Apply bitsandbytes quantization when VRAM is limited.")
    parser.add_argument("--context7-url", type=str, help="Context7 endpoint base URL for documentation lookups.")
    parser.add_argument("--context7-api-key", type=str, help="Context7 API key (optional).")
    parser.add_argument("--system-prompt", type=str, help="Override the default system prompt text.")
    parser.add_argument("--system-prompt-file", type=Path, help="Load system prompt text from file.")
    parser.add_argument("--disable-system-prompt", action="store_true", help="Disable system prompt injection (useful for models with baked-in instructions).")
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", help="Pass enable_thinking=True to the model generate call.")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false", help="Pass enable_thinking=False to the model generate call.")
    parser.set_defaults(enable_thinking=None)
    parser.add_argument("--context-window", type=int, help="Override tokenizer context window (token limit) when supported.")
    parser.add_argument("--max-actions-per-turn", type=int, default=1, help="Maximum tool calls to execute per assistant turn.")
    parser.add_argument("--observation-summary-chars", type=int, default=512, help="Truncate observations to this many characters when feeding back into the model.")
    parser.add_argument("--audit-log", type=Path, help="Path to append JSON audit events (tool_calls and observations).")
    parser.epilog = textwrap.dedent(
        """Examples:
  # Devstral-Small in 4-bit on a single GPU
  python agent_cli.py --model defog/Devstral-small-1.0 \
      --device-map auto --quantization 4bit --torch-dtype float16 \
      --trust-remote-code --low-cpu-mem-usage --context-window 131072

  # Qwen2.5-Coder-32B on dual GPUs with large context
  python agent_cli.py --model Qwen/Qwen2.5-Coder-32B \
      --device-map auto --torch-dtype bfloat16 --max-new-tokens 2048 \
      --context-window 131072 --system-prompt-file prompts/qwen.txt

  # CPU-only debug run with default prompt disabled
  python agent_cli.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
      --device-map cpu --torch-dtype float32 --disable-system-prompt --once "ping"

  # With context7 integration (requires CONTEXT7_URL)
  CONTEXT7_URL=https://context7.local/v1 python agent_cli.py --model defog/Devstral-small-1.0 \
      --device-map auto --quantization 4bit --context7-url https://context7.local/v1
"""
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    memory = MemoryBank(args.memory)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.disable_system_prompt:
        system_prompt = None
    elif args.system_prompt or args.system_prompt_file:
        if args.system_prompt:
            system_prompt = args.system_prompt
        else:
            try:
                system_prompt = args.system_prompt_file.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Failed to load system prompt file: %s", exc)
                return 1

    if args.remember:
        memory.append(args.remember)
        print(f"Stored note: {args.remember}")
        return 0

    audit_log = args.audit_log
    if audit_log is None:
        audit_log = Path("logs/agent_audit.jsonl")
    if audit_log is not None:
        audit_log.parent.mkdir(parents=True, exist_ok=True)

    try:
        vector_memory = VectorMemory(args.memory_vector_dir)
    except Exception as exc:
        logger.warning("Vector memory disabled: %s", exc)

        class _NullVectorMemory:
            def __init__(self):
                self.metadata = []

            def add(self, *_, **__):
                return None

            def search(self, *_args, **_kwargs):
                return []

        vector_memory = _NullVectorMemory()

    torch_dtype = _parse_dtype(args.torch_dtype)
    try:
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            device_map=args.device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            quantization=args.quantization,
        )
    except Exception as exc:
        logger.error("Failed to load model %s: %s", args.model, exc)
        print(
            "Model load failed. Verify the model name, ensure you have network access or a Hugging Face token (huggingface-cli login), "
            "and download checkpoints in advance if necessary.",
        )
        return 1
    context7_url = args.context7_url or os.getenv("CONTEXT7_URL")
    context7_api_key = args.context7_api_key or os.getenv("CONTEXT7_API_KEY") or DEFAULT_CONTEXT7_API_KEY

    retriever = DocumentationRetriever(args.docs)
    executor = CommandExecutor(args.workspace)

    config = AgentConfig(
        model_name_or_path=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        docs_path=args.docs,
        memory_file=args.memory,
        memory_vector_dir=args.memory_vector_dir,
        workspace=args.workspace,
        system_prompt=system_prompt,
        enable_thinking=args.enable_thinking,
        context_window_tokens=args.context_window,
        quantization=args.quantization,
        max_actions_per_turn=max(1, args.max_actions_per_turn),
        observation_summary_chars=max(32, args.observation_summary_chars),
        audit_log=audit_log,
        context7_url=context7_url,
        context7_api_key=context7_api_key,
    )

    agent = CLIProgrammingAgent(config, model, tokenizer, memory, vector_memory, retriever, executor)

    if args.once:
        reply = agent.interact(args.once)
        print(reply)
        return 0

    print("Local CLI agent ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession ended.")
            break
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not user_input:
            continue
        reply = agent.interact(user_input)
        print(f"Agent> {reply}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
