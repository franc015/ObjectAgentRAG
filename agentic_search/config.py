"""Chargement de la configuration (TOML + variables d'environnement)."""

import copy
import os
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "data_dir": "data",
    "llm": {
        "provider": "anthropic",          # "anthropic" ou "openai_compat" (serveur local)
        "model": "claude-sonnet-5-5",
        "base_url": "",                   # vide = adresse par défaut du fournisseur
        "api_key_env": "ANTHROPIC_API_KEY",
        "max_tokens": 2048,
        "timeout": 120,
    },
    "agent": {"max_turns": 8, "max_results": 5, "max_per_doc": 3, "tool_result_chars": 12000},
    "ingest": {"passage_words": 250, "overlap_words": 40,
               "keep_markdown": False,    # True : conserve le markdown complet et des métadonnées (ClickHouse)
               "keep_figures": False,     # True (avec keep_markdown) : repère et conserve les figures des articles
               "figures_min_px": 120, "figures_max_per_doc": 40},
    "dates": {"mode": "llm", "provider": "anthropic", "model": "claude-haiku-5-5", "base_url": "",
              "head_chars": 1500, "tail_chars": 500},   # mode : "llm" ou "none"
    "themes": {"mode": "none", "provider": "anthropic", "model": "claude-haiku-5-5", "base_url": "",
               "max_themes": 3, "excerpt_chars": 2500,        # mode : "llm" ou "none"
               "vocabulary": ["cybersécurité", "data", "delivery", "CI/CD", "IA agentique", "architecture",
                              "java", "web services", "intégration", "IA générative", "cloud",
                              "gouvernance et conformité"]},
    "backend": {"kind": "memory"},       # "memory" ou "clickhouse"
    "clickhouse": {
        "host": "localhost", "port": 8123, "username": "default",
        "password_env": "CLICKHOUSE_PASSWORD", "database": "agentic", "secure": False,
        "retention_months": 12,
    },
    "tracing": {"kind": "jsonl"},        # "none", "jsonl" ou "clickhouse"
    "server": {"host": "127.0.0.1", "port": 8080, "token_env": "AGENTIC_API_TOKEN"},
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    path = path or os.environ.get("AGENTIC_CONFIG") or ("config.toml" if Path("config.toml").exists() else None)
    if path:
        with open(path, "rb") as f:
            _merge(cfg, tomllib.load(f))
    env = os.environ
    if env.get("AGENTIC_DATA_DIR"):
        cfg["data_dir"] = env["AGENTIC_DATA_DIR"]
    if env.get("AGENTIC_BACKEND"):
        cfg["backend"]["kind"] = env["AGENTIC_BACKEND"]
    if env.get("AGENTIC_TRACING"):
        cfg["tracing"]["kind"] = env["AGENTIC_TRACING"]
    if env.get("AGENTIC_PROVIDER"):
        cfg["llm"]["provider"] = env["AGENTIC_PROVIDER"]
    if env.get("AGENTIC_MODEL"):
        cfg["llm"]["model"] = env["AGENTIC_MODEL"]
    if env.get("AGENTIC_DATES_MODE"):
        cfg["dates"]["mode"] = env["AGENTIC_DATES_MODE"]
    if env.get("AGENTIC_THEMES_MODE"):
        cfg["themes"]["mode"] = env["AGENTIC_THEMES_MODE"]
    if env.get("AGENTIC_DATES_BASE_URL"):
        cfg["dates"]["base_url"] = env["AGENTIC_DATES_BASE_URL"]
    if env.get("AGENTIC_LLM_BASE_URL"):
        cfg["llm"]["base_url"] = env["AGENTIC_LLM_BASE_URL"]
    ch = cfg["clickhouse"]
    if env.get("CLICKHOUSE_HOST"):
        ch["host"] = env["CLICKHOUSE_HOST"]
    if env.get("CLICKHOUSE_PORT"):
        ch["port"] = int(env["CLICKHOUSE_PORT"])
    if env.get("CLICKHOUSE_USER"):
        ch["username"] = env["CLICKHOUSE_USER"]
    if env.get("CLICKHOUSE_DB"):
        ch["database"] = env["CLICKHOUSE_DB"]
    if env.get("AGENTIC_SERVER_HOST"):
        cfg["server"]["host"] = env["AGENTIC_SERVER_HOST"]
    if env.get("AGENTIC_SERVER_PORT"):
        cfg["server"]["port"] = int(env["AGENTIC_SERVER_PORT"])
    return cfg
