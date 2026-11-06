# Application « ClickHouse »

Mêmes fonctions que l'application « base », avec un serveur **ClickHouse sur site** pour :

- les **passages** et un index de postings : le classement BM25 est calculé en SQL ;
- les **documents** classés et filtrés par date (ReplacingMergeTree sur l'identifiant : une correction de
  date réinsère une version plus récente) ;
- les **traces** des exécutions de l'agent (`agent_runs`, `agent_events`, conservation de 12 mois par défaut).

- Configuration : [`config.toml`](config.toml)
- Image : [`Dockerfile`](Dockerfile) (contexte = racine du dépôt) et [`docker-compose.yml`](docker-compose.yml)
- Variables : [`.env.example`](.env.example)

## Validation obligatoire avant usage

Le SQL a été écrit et testé avec un **faux client**, jamais contre un vrai serveur ClickHouse. Après
l'ingestion, lancez :

```bash
agentic-search --config apps/clickhouse/config.toml ch-selftest
```

La commande charge le corpus dans une base temporaire et vérifie que les résultats BM25 de ClickHouse sont
identiques à ceux du backend mémoire. Si elle signale des écarts, ne l'utilisez pas en production et
ouvrez un ticket avec la sortie.

## Sans conteneur (depuis la racine du dépôt)

```bash
pip install '.[all]'
export ANTHROPIC_API_KEY=... CLICKHOUSE_PASSWORD=...
agentic-search --config apps/clickhouse/config.toml ch-init
agentic-search --config apps/clickhouse/config.toml ingest ./corpus
agentic-search --config apps/clickhouse/config.toml ch-selftest
agentic-search --config apps/clickhouse/config.toml ask "votre question" -v
```

Prérequis : ClickHouse ≥ 26.2 (l'index de texte y est en disponibilité générale ; une version LTS est
recommandée). Sur un serveur plus ancien, le service fonctionne sans index de texte.

Détail du schéma, des requêtes d'analyse des traces et des limites : [README racine](../../README.md).
