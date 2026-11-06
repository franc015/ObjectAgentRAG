# ObjectAgentRAG (paquet `agentic-search`)

Service de **recherche textuelle agentique** : un agent (modèle de langage appelé par API, ou modèle
local) interroge un corpus d'articles de presse, de pages web, de documents et de livres au moyen
d'outils de recherche **lexicale** (mots-clés, classement BM25, sans sémantique pour l'instant).

## Organisation du dépôt

Un cœur commun et **deux applications** qui le déploient :

```
agentic_search/      cœur : ingestion, dates, moteurs (mémoire, ClickHouse), outils, agent, service HTTP
tests/               53 tests (unittest)
apps/base/           application sans base de données : config, Dockerfile, docker-compose, README
apps/clickhouse/     application avec ClickHouse sur site : config, Dockerfile, docker-compose, README
docs/choix.md        décisions, faits vérifiés, points non validés
```

Choix et points à valider en premier : [docs/choix.md](docs/choix.md).

Deux variantes du même service, choisies par la configuration :

| | Version de base | Version ClickHouse |
|---|---|---|
| Index | BM25 en mémoire (fichiers JSON) | BM25 calculé en SQL (table de postings) |
| Documents par date | oui | oui, triés et filtrés dans la base |
| Traces des exécutions | fichiers JSONL | tables `agent_runs` / `agent_events` |
| Dépendances serveur | aucune | ClickHouse ≥ 26.2 (sur site) |
| Application | [`apps/base`](apps/base) | [`apps/clickhouse`](apps/clickhouse) |
| Configuration | `apps/base/config.toml` | `apps/clickhouse/config.toml` |

Formats pris en charge : PDF (texte), Word `.docx`, EPUB (sans DRM), Markdown / texte, HTML, et liens web
(`urls.txt`).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install .                  # version de base
pip install '.[clickhouse]'    # + client ClickHouse
pip install '.[all]'           # + extraction web améliorée (trafilatura)
```

Roue : `pip wheel . --no-deps -w dist` (ou `python -m build`), puis
`pip install 'dist/agentic_search-0.1.0-py3-none-any.whl[all]'`.

Python ≥ 3.11. Les versions des dépendances ne sont pas figées : après une première installation validée,
exécutez `pip freeze > requirements.lock` pour des déploiements reproductibles.

## Démarrage rapide (version de base)

```bash
export ANTHROPIC_API_KEY=...            # jamais dans un fichier versionné
agentic-search --config apps/base/config.toml ingest ./corpus   # extrait, découpe en passages, écrit data/
agentic-search stats
agentic-search search "assurance habitation franchise" --from 2024
agentic-search docs --order date_desc --limit 20
agentic-search ask "Comment la franchise a-t-elle évolué ?" -v
agentic-search runs                     # dernières exécutions tracées
```

Le dossier du corpus peut contenir un `urls.txt` (une URL par ligne, `#` pour les commentaires) pour les
pages web. Respectez les conditions d'utilisation des sites ; les pages derrière un mur payant ou
bloquant les robots doivent être enregistrées en PDF et placées dans le dossier.

### Dates : détection par un modèle léger

Les métadonnées d'un PDF ou d'un Word indiquent souvent la date de création du fichier, pas celle de
publication. À l'ingestion, un **modèle léger** (`claude-haiku-5-5` par défaut, section `[dates]`) lit le
début et la fin du texte (mentions « publié le », signatures, dépôt légal) et propose une date :

- appelé seulement pour les cas douteux : PDF, Word, texte, ou document sans date. Les dates issues de la
  page web (métadonnées ou URL) et celles des EPUB ne sont pas retouchées ;
- réponse validée strictement (format, pas de date future, pas avant 1800) ; une confiance « basse » ou
  une réponse invalide laisse la date existante en place ;
- la date retenue porte l'origine `modele` (autres origines : `meta_web`, `url`, `metadonnee_fichier`,
  `manuel`), visible avec `agentic-search docs` ;
- résultats mis en cache dans `data/dates_cache.json` (une ré-ingestion ne repaie pas les appels) ;
- si le modèle est indisponible, l'ingestion continue avec les dates existantes (arrêt des appels après
  3 échecs consécutifs). `--no-dates` ou `mode = "none"` désactive la fonction ; avec un modèle local,
  réglez `provider`, `model` et `base_url` dans `[dates]`.

Le coût est faible (un petit extrait par document) mais non nul ; le résumé de l'ingestion indique les
jetons consommés. Le modèle peut se tromper : l'origine `modele` permet de filtrer ou de revoir ces dates.
Les documents restant sans date sont listés ; corrigez-les à la main :

```bash
agentic-search set-date <id_document> 2024-03-12      # AAAA, AAAA-MM ou AAAA-MM-JJ
```

## Variante ClickHouse

```bash
export CLICKHOUSE_PASSWORD=...
agentic-search --config apps/clickhouse/config.toml ch-init       # crée la base et les tables
agentic-search --config apps/clickhouse/config.toml ingest ./corpus
agentic-search --config apps/clickhouse/config.toml ch-selftest   # À FAIRE AVANT USAGE
```

`ch-selftest` charge votre corpus dans une base temporaire et vérifie que les résultats BM25 de
ClickHouse sont identiques à ceux de la version de base. **Les requêtes SQL ont été écrites et testées
avec un faux client, pas contre un serveur réel** : ce contrôle est la validation à effectuer chez vous.

Schéma : `documents` (ReplacingMergeTree sur l'identifiant : une correction de date réinsère une version
plus récente ; les requêtes utilisent `FINAL`), `passages` (avec index de texte sur le texte normalisé),
`postings` (terme, passage, fréquence, longueur : le BM25 est calculé sans jointure sur `passages`),
`agent_runs`, `agent_events` (conservation configurable, 12 mois par défaut). Si le serveur est antérieur
à la 26.2, la table est créée sans index de texte (le service fonctionne, plus lentement pour les
expressions exactes).

Exemples d'analyse des traces :

```sql
-- coût et durée par modèle
SELECT modele, count() AS n, avg(jetons_in) AS in_moy, avg(jetons_out) AS out_moy, avg(duree_ms) AS ms
FROM agentic.agent_runs GROUP BY modele;

-- recherches sans résultat
SELECT arguments FROM agentic.agent_events WHERE outil = 'search' AND n_resultats = 0 ORDER BY ts DESC LIMIT 20;

-- documents les plus cités
SELECT arrayJoin(passages_cites) AS passage, count() AS n FROM agentic.agent_runs GROUP BY passage ORDER BY n DESC;
```

## Modèles

`[llm]` dans la configuration (ou `AGENTIC_PROVIDER`, `AGENTIC_MODEL`, `AGENTIC_LLM_BASE_URL`) :

- `provider = "anthropic"` : API Messages d'Anthropic (`claude-sonnet-5-5` par défaut ; essayez
  `claude-haiku-5-5` pour comparer le coût et la qualité sur vos questions).
- `provider = "openai_compat"` : tout serveur exposant `/chat/completions` avec appels d'outils
  (Ollama, vLLM, llama.cpp…). La fiabilité de l'appel d'outils varie beaucoup d'un modèle local à l'autre :
  comparez-les sur un même jeu de questions, en vous aidant des traces.

Les appels réseau sont faits avec la bibliothèque standard (aucun SDK). Le format des requêtes est validé
par les tests contre un faux serveur, pas contre les API réelles.

## Service HTTP

```bash
export AGENTIC_API_TOKEN=un-jeton-long-et-aleatoire
agentic-search serve          # 127.0.0.1:8080 par défaut
curl -H "Authorization: Bearer $AGENTIC_API_TOKEN" -d '{"question":"..."}' localhost:8080/ask
```

`GET /health` (sans authentification), `GET /stats`, `GET /search?q=...&type=pdf&from=2024&to=2025`,
`POST /ask` (`{"question": "...", "model": "..."}` ; réponse avec sources, jetons et `run_id`).
Si `AGENTIC_API_TOKEN` n'est pas défini, le service n'exige aucun jeton : ne l'exposez alors que sur
`127.0.0.1`. Placez-le derrière un proxy inverse avec TLS pour tout autre usage.

## Conteneur

Chaque application a son image, construite depuis la **racine du dépôt** :

```bash
docker build -f apps/base/Dockerfile -t objectagentrag-base .
docker build -f apps/clickhouse/Dockerfile -t objectagentrag-clickhouse .
```

Les `docker-compose.yml` (et leur `.env.example`) sont dans chaque dossier d'application ; voir leur
README. Le tag de l'image ClickHouse est à vérifier sur Docker Hub avant usage. Ces fichiers n'ont pas pu
être construits ni lancés dans l'environnement où le dépôt a été préparé.

## Sécurité

- Le contenu des documents est traité comme de la donnée : le prompt système demande au modèle d'ignorer
  toute consigne présente dans un passage (risque d'injection de prompt via des pages web). C'est une
  atténuation, pas une garantie ; l'agent n'a d'ailleurs que des outils en lecture seule.
- Les requêtes ClickHouse sont paramétrées (`{nom:Type}`) ; seuls les noms de base et de table, validés,
  sont insérés dans le SQL.
- Les clés (API du modèle, ClickHouse, jeton du service) viennent de variables d'environnement.
- Les traces contiennent les questions et les réponses : traitez-les comme des données du corpus.

## Limites connues

- Recherche **lexicale** : pas de synonymes ni de sens ; l'agent compense en reformulant. Radicalisation
  du français minimale (pluriels). La sémantique (vecteurs) est la prochaine étape naturelle.
- PDF scannés : pas d'OCR (un avertissement est affiché). EPUB avec DRM : refusés. Colonnes multiples
  dans un PDF : l'ordre de lecture peut être imparfait ; vérifiez quelques documents après ingestion.
- L'ingestion remplace tout le contenu à chaque exécution (pas d'ingestion incrémentale).
- Le service HTTP est minimal (bibliothèque standard) ; pour une charge importante, placez un serveur
  d'application devant ou adaptez `server.py`.

## Tests

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests
```

53 tests : ingestion des cinq formats, détection des dates par modèle, recherche et filtres de date, outils, boucle agentique, formats
d'appel des modèles (faux serveur HTTP), service HTTP, plomberie ClickHouse (faux client).
