"""Ligne de commande : python -m agentic_search <commande> (ou `agentic-search <commande>`)."""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__
from .backend import SearchBackend, write_neutral
from .config import load_config
from .ingest import ingest_corpus
from .tools import Toolbox
from .tracing import JsonlTracer, NullTracer, Tracer


def open_backend(cfg: dict) -> SearchBackend:
    kind = cfg["backend"]["kind"]
    if kind == "memory":
        from .memory_backend import MemoryBackend
        return MemoryBackend(cfg["data_dir"])
    if kind == "clickhouse":
        from .clickhouse_backend import ClickHouseBackend
        return ClickHouseBackend(cfg)
    raise SystemExit(f"backend inconnu : {kind}")


def open_tracer(cfg: dict, backend: SearchBackend) -> Tracer:
    kind = cfg["tracing"]["kind"]
    if kind == "none":
        return NullTracer()
    if kind == "jsonl":
        return JsonlTracer(cfg["data_dir"])
    if kind == "clickhouse":
        from .clickhouse_backend import ClickHouseBackend, ClickHouseTracer
        if not isinstance(backend, ClickHouseBackend):
            backend = ClickHouseBackend(cfg)
        return ClickHouseTracer(backend)
    raise SystemExit(f"traçage inconnu : {kind}")


def make_toolbox(cfg: dict, backend: SearchBackend) -> Toolbox:
    a = cfg["agent"]
    return Toolbox(backend, a["max_results"], a["max_per_doc"], a["tool_result_chars"])


def make_date_resolver(cfg: dict, disabled: bool = False):
    if disabled or cfg["dates"]["mode"] != "llm":
        return None
    from .dates import DateResolver
    from .llm import LLMError, make_dates_llm
    try:
        llm = make_dates_llm(cfg)
    except LLMError as e:
        print(f"  attention : dates non vérifiées par le modèle ({e})")
        return None
    d = cfg["dates"]
    return DateResolver(llm, Path(cfg["data_dir"]) / "dates_cache.json", d["head_chars"], d["tail_chars"])


def cmd_ingest(args, cfg):
    resolver = make_date_resolver(cfg, args.no_dates)
    docs, passages, warnings = ingest_corpus(args.corpus, args.urls, cfg["ingest"]["passage_words"],
                                             cfg["ingest"]["overlap_words"], date_resolver=resolver)
    write_neutral(cfg["data_dir"], docs, passages)
    if resolver:
        resolver.save()
        print(resolver.summary())
    print(f"{len(docs)} documents, {len(passages)} passages -> {cfg['data_dir']}/")
    for w in warnings:
        print("  attention :", w)
    undated = [d for d in docs if not d.get("date")]
    if undated:
        print(f"  {len(undated)} document(s) sans date (agentic-search set-date <id> AAAA-MM-JJ) :")
        for d in undated[:10]:
            print(f"    {d['id']}  {d['title'][:70]}")
    if cfg["backend"]["kind"] == "clickhouse":
        backend = open_backend(cfg)
        backend.load(passages, docs)
        print("chargé dans ClickHouse")


def cmd_stats(args, cfg):
    print(json.dumps(open_backend(cfg).stats(), ensure_ascii=False, indent=1))


def cmd_search(args, cfg):
    b = open_backend(cfg)
    for h in b.search(args.query, limit=args.limit, doc_type=args.type, date_min=args.date_from, date_max=args.date_to):
        print(f"[{h['id']}] {h['score']:>6}  {h['date'] or '----------'}  {h['title'][:60]}\n    {h['extrait'][:220]}")


def cmd_docs(args, cfg):
    for d in open_backend(cfg).list_documents(args.contains, args.type, args.date_from, args.date_to, args.order, args.limit):
        origin = f" ({d['date_origine']})" if d.get("date_origine") else ""
        print(f"{d['id']}  {d['date'] or '----------'}  {d['type']:5}  {d['title'][:60]}{origin}")


def cmd_set_date(args, cfg):
    b = open_backend(cfg)
    print("date mise à jour" if b.set_date(args.doc_id, args.date) else "document inconnu")


def cmd_ask(args, cfg):
    from .agent import run_agent
    from .llm import make_llm
    b = open_backend(cfg)
    llm = make_llm(cfg, args.provider, args.model)
    on_step = (lambda n, a, k: print(f"  · {n} {a} -> {k} résultat(s)", file=sys.stderr)) if args.verbose else None
    res = run_agent(args.question, llm, make_toolbox(cfg, b), open_tracer(cfg, b), cfg["agent"]["max_turns"], on_step)
    print(res.answer)
    print(f"\n[{res.status} · {res.turns} tour(s) · {res.usage['input']} jetons en entrée, "
          f"{res.usage['output']} en sortie · {llm.provider}/{llm.model} · run {res.run_id}]", file=sys.stderr)


def cmd_runs(args, cfg):
    b = open_backend(cfg)
    for r in open_tracer(cfg, b).recent(args.n):
        print(f"{r['ts']}  {r['status']:14} {r.get('turns')} tours  {r['question'][:70]}")


def cmd_serve(args, cfg):
    from .llm import make_llm
    from .server import make_server
    b = open_backend(cfg)
    srv = make_server(cfg, b, open_tracer(cfg, b), make_toolbox(cfg, b), lambda p, m: make_llm(cfg, p, m))
    host, port = srv.server_address[:2]
    logging.info("service sur http://%s:%s (backend %s)", host, port, b.name)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_ch_init(args, cfg):
    from .clickhouse_backend import ClickHouseBackend
    b = ClickHouseBackend(cfg)
    print(f"schéma prêt dans la base {b.db} (index de texte : {'oui' if b.text_index else 'non'})")


def cmd_ch_selftest(args, cfg):
    from .clickhouse_backend import selftest
    from .memory_backend import MemoryBackend
    ref = MemoryBackend(cfg["data_dir"])
    if not ref.passages:
        raise SystemExit("aucun corpus : lancez d'abord `ingest`")
    report = selftest(cfg, ref)
    print("\n".join(report))
    sys.exit(0 if report[-1].endswith("identique") else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="agentic-search", description=__doc__)
    p.add_argument("--config", help="fichier TOML (sinon AGENTIC_CONFIG ou ./config.toml)")
    p.add_argument("--version", action="version", version=__version__)
    p.add_argument("-q", "--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="extraire, découper et charger un corpus")
    s.add_argument("corpus"); s.add_argument("--urls", help="fichier d'URL (défaut : <corpus>/urls.txt)")
    s.add_argument("--no-dates", action="store_true", help="ne pas utiliser le modèle pour les dates")
    s.set_defaults(fn=cmd_ingest)
    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    s = sub.add_parser("search", help="recherche directe (sans modèle)")
    s.add_argument("query"); s.add_argument("--limit", type=int, default=5); s.add_argument("--type")
    s.add_argument("--from", dest="date_from"); s.add_argument("--to", dest="date_to")
    s.set_defaults(fn=cmd_search)
    s = sub.add_parser("docs", help="lister les documents par date")
    s.add_argument("--contains"); s.add_argument("--type"); s.add_argument("--limit", type=int, default=30)
    s.add_argument("--from", dest="date_from"); s.add_argument("--to", dest="date_to")
    s.add_argument("--order", default="date_desc", choices=["date_desc", "date_asc", "titre"])
    s.set_defaults(fn=cmd_docs)
    s = sub.add_parser("set-date", help="corriger la date d'un document")
    s.add_argument("doc_id"); s.add_argument("date"); s.set_defaults(fn=cmd_set_date)
    s = sub.add_parser("ask", help="poser une question à l'agent")
    s.add_argument("question"); s.add_argument("--provider"); s.add_argument("--model")
    s.add_argument("-v", "--verbose", action="store_true"); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("runs", help="dernières exécutions tracées")
    s.add_argument("-n", type=int, default=10); s.set_defaults(fn=cmd_runs)
    sub.add_parser("serve", help="service HTTP").set_defaults(fn=cmd_serve)
    sub.add_parser("ch-init", help="créer le schéma ClickHouse").set_defaults(fn=cmd_ch_init)
    sub.add_parser("ch-selftest", help="comparer ClickHouse et le backend mémoire").set_defaults(fn=cmd_ch_selftest)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")
    args.fn(args, load_config(args.config))


if __name__ == "__main__":
    main()
