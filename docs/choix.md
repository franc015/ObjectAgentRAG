# Choix et décisions

Ce document garde la trace des choix faits pour les deux applications. Il distingue ce qui a été décidé,
ce qui a été vérifié dans des sources, et ce qui reste à valider.

## Décisions

- **Objectif** : tester une recherche textuelle agentique, **lexicale, sans sémantique pour le moment**, avec
  un modèle de langage appelé par API derrière.
- **Corpus** : articles de presse et web (plusieurs centaines ; PDF, Word, liens web) et une dizaine de
  livres (EPUB).
- **Deux applications** dans un seul dépôt, sur un cœur commun (`agentic_search/`) :
  - `apps/base` : sans base de données (index BM25 en mémoire, traces en fichiers JSONL) ;
  - `apps/clickhouse` : ClickHouse sur site pour les passages, le tri des articles par date et les traces.
- **Traçage** : pas d'infrastructure OpenTelemetry existante, donc insertion directe des traces (fichiers
  ou ClickHouse) derrière une interface `Tracer`, remplaçable plus tard par OpenTelemetry.
- **Dates** : détectées par un **modèle léger** (`claude-haiku-5-5` par défaut) pour les documents dont la
  date de fichier est peu fiable ; origine de chaque date conservée (`date_origine`).
- **Modèle de l'agent** : `claude-sonnet-5-5` par défaut, comparable à Haiku ou à un modèle local
  (`openai_compat`) sur un même jeu de questions grâce aux traces.
- **Packaging** : paquet Python (roue), image de conteneur par application, `docker-compose`.

## Faits vérifiés dans des sources (au 6 novembre 2026)

- L'**index de texte de ClickHouse** est en disponibilité générale à partir de la version 26.2 ; ce n'est
  pas un moteur de pertinence (ni TF-IDF ni BM25) : le classement BM25 est donc calculé en SQL à partir
  d'une table de postings. 26.3 est une version LTS (publiée le 26 mars 2026).
  Sources : documentation ClickHouse « Text indexes » ; annonce de disponibilité générale (10 mars 2026).
- Les **conventions OpenTelemetry pour les agents GenAI** ont le statut « Development » dans la
  documentation officielle : leurs noms peuvent encore changer.
- La dernière révision stable du **protocole MCP** est datée du 2026-07-28 (non utilisée dans ce dépôt ;
  une exposition de l'agent en serveur MCP est une extension possible).

## Choix de conception notables

- Dans ClickHouse, la table `documents` est triée par identifiant (et non par date) : avec
  ReplacingMergeTree, une clé de tri contenant la date aurait laissé des doublons après correction d'une
  date. Le tri par date se fait à la requête.
- Les postings contiennent la longueur du passage, pour calculer le BM25 sans jointure sur la table des
  passages.
- La normalisation française (accents, pluriels, mots vides) est faite en Python, une seule fois, par le
  même module pour l'indexation et la requête : les deux applications donnent des résultats comparables.

## Variante ClickHouse : contenu, métadonnées et thèmes

- Pour comparer le retrieval (et non la transformation en markdown), les deux applications produisent les
  mêmes passages : mêmes réglages `keep_markdown` / `keep_figures`, markdown écrit sur disque dans
  `data/markdown/<id>.md` par les deux. Le plus sûr est d'ingérer une seule fois, puis de charger le même
  `data/` dans les deux (`agentic-search load` pour ClickHouse ; le backend mémoire relit `data/`).
  Les métadonnées JSON et les thèmes ne sont stockés que par ClickHouse ; ils ne changent pas les passages.
- Le markdown complet est dans une table séparée `documents_contenu` (ZSTD) : une correction de date ou de
  thème ne réécrit pas le texte. Il permet `reindex` (reconstruire passages et postings sans relire les
  fichiers) et l'outil `get_document` (lecture d'un article par tranches).
- Les métadonnées variables (fichier, empreinte, URL, site, langue) sont dans une colonne `meta` de type
  JSON (production depuis la 25.3, repli en String avant) ; les champs de tri et de filtre restent en colonnes typées.
- Les thèmes sont attribués par un modèle léger, uniquement dans un vocabulaire contrôlé modifiable, et
  corrigeables à la main (`set-themes`). Le filtre est un « au moins un des thèmes ».
- Les figures (images) sont repérées dans les Word, PDF (images raster), EPUB et pages web. Les fichiers
  restent sur disque (`data/figures/<document>/<empreinte>.<ext>`), le markdown les référence par une ligne
  `![légende](chemin)` et ClickHouse stocke leurs métadonnées (`documents_figures`). La légende est indexée
  (« Figure : … »), jamais le chemin. Pour le web, seule l'adresse est conservée (aucun téléchargement).
  Non repérés : schémas vectoriels des PDF, formes et SmartArt Word, texte à l'intérieur des images (OCR).
- Les corrections passent par de nouvelles versions de ligne (ReplacingMergeTree) ; les mises à jour
  légères (`UPDATE`) sont évitées tant que la documentation les indique en bêta.

## Ingestion incrémentale

- La transformation des fichiers (extraction, markdown, découpage) se fait à l'ingestion, jamais à la
  recherche. `ingest` ne relit que les fichiers nouveaux ou modifiés (empreinte SHA-256 dans
  `data/manifest.json`) ; les autres sont repris tels quels, avec leurs passages et leur markdown. Les fichiers
  supprimés sont retirés, ainsi que leurs images. Un lien déjà ingéré n'est pas retéléchargé.
- Un changement de réglages (taille des passages, figures, modèles de dates ou de thèmes, vocabulaire)
  provoque un retraitement complet, de même que `--full`. Les caches de dates et de thèmes évitent de
  rappeler les modèles.
- Limite : les corrections faites dans ClickHouse (`set-date`, `set-themes`) ne sont pas écrites dans
  `data/` ; un `ingest` ou un `load` les écrase. Celles de la version de base (`set-date`) sont écrites dans
  `data/documents.json` et conservées pour les fichiers inchangés.
- Le chargement dans ClickHouse reste complet à chaque `ingest` (rapide, sans relire les sources).

## Non validé (à faire en premier)

1. Le **SQL ClickHouse** n'a jamais été exécuté contre un vrai serveur : lancer `ch-selftest` (il vérifie
   aussi l'aller-retour des thèmes, des métadonnées JSON et du markdown, le filtre par thème, la lecture par
   tranches et la réindexation).
2. Les **appels aux API de modèles** sont testés contre un faux serveur, pas contre les API réelles.
3. Les **Dockerfile** et **docker-compose** n'ont pas été construits (pas de démon Docker disponible lors
   de la préparation) ; le tag `clickhouse/clickhouse-server:26.3` est à vérifier sur Docker Hub.
4. L'**extraction de pages web** n'a pas été testée sur de vrais sites (pas d'accès réseau lors de la
   préparation).

## Pistes ensuite

- Ajouter la recherche sémantique (vecteurs) et mesurer le gain sur un jeu de questions de référence.
- Comparer Sonnet, Haiku et un modèle local sur ces questions (coût, qualité, appels d'outils).
- Remplacer le `Tracer` direct par OpenTelemetry si une infrastructure d'observabilité apparaît.
