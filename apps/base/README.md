# Application « base »

Recherche agentique lexicale sur un corpus local, **sans base de données** : index BM25 en mémoire
(reconstruit au démarrage depuis `data/`), documents classés par date, traces des exécutions en fichiers
JSONL (`data/traces/`). C'est la version à lancer en premier pour valider l'agent et la qualité de la
recherche.

- Configuration : [`config.toml`](config.toml)
- Image : [`Dockerfile`](Dockerfile) (contexte = racine du dépôt) et [`docker-compose.yml`](docker-compose.yml)
- Variables : [`.env.example`](.env.example)

## Sans conteneur (depuis la racine du dépôt)

```bash
pip install '.[web]'
export ANTHROPIC_API_KEY=...
agentic-search --config apps/base/config.toml ingest ./corpus
agentic-search --config apps/base/config.toml ask "votre question" -v
agentic-search --config apps/base/config.toml serve
```

## Avec Docker

Voir l'en-tête de `docker-compose.yml`. Le build n'a pas pu être testé dans l'environnement où le dépôt a
été préparé : à vérifier à la première construction.

Détail des commandes, de la configuration, du service HTTP et des limites : [README racine](../../README.md).
