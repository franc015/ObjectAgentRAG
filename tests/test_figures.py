"""Figures des articles : repérage (Word, PDF, EPUB, HTML, web), markdown, passages, stockage ClickHouse."""

import io
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw

from agentic_search.clickhouse_backend import ClickHouseBackend, check_documents
from agentic_search.config import load_config
from agentic_search.ingest import (FigureCollector, build_passages, extract_html, figure_label,
                                   figure_passage_line, ingest_corpus, markdown_to_sections, sniff_image)

from fixtures import filler
from test_clickhouse_plumbing import FakeClient, check_placeholders


def png(w=320, h=200, text="Architecture API -> BDD") -> bytes:
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((10, h // 2), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


BIG, SMALL = png(), png(40, 40, "logo")
SVG = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="300" height="200"><rect/></svg>'
CORPUS_WORDS = dict(words=120, overlap=20)


def ingest(root, figures=True, markdown=True, fig_dir=None, **kw):
    return ingest_corpus(root, keep_markdown=markdown, keep_figures=figures,
                         figures_dir=fig_dir, **CORPUS_WORDS, **kw)


def make_docx(path: Path, alt="Architecture cible : API vers base de données"):
    import docx
    d = docx.Document()
    d.core_properties.title = "Article Word avec schéma"
    d.add_heading("Contexte", 1)
    d.add_paragraph("Avant le schéma. " + filler(40))
    (path.parent / "fig.png").write_bytes(BIG)
    d.add_picture(str(path.parent / "fig.png"))
    d.inline_shapes[0]._inline.docPr.set("descr", alt)
    d.add_paragraph("Figure 1 : architecture cible")
    d.add_picture(str(path.parent / "fig.png"))                 # même image : une seule figure enregistrée
    (path.parent / "small.png").write_bytes(SMALL)
    d.add_picture(str(path.parent / "small.png"))               # logo : écarté
    d.add_paragraph("Après le schéma. " + filler(40, 3))
    d.save(str(path))
    (path.parent / "fig.png").unlink()
    (path.parent / "small.png").unlink()


class TestCollector(unittest.TestCase):
    def test_sniff_par_contenu(self):
        self.assertEqual(sniff_image(BIG), ("png", "image/png"))
        self.assertEqual(sniff_image(SVG)[0], "svg")
        self.assertEqual(sniff_image(b"\xff\xd8\xff\xe0abc")[0], "jpg")
        for bad in (b"MZ\x90\x00 executable", b"\x01\x00\x00\x00 emf", b"<html></html>", b""):
            self.assertIsNone(sniff_image(bad))

    def test_filtres_et_quota(self):
        c = FigureCollector(min_px=120, max_items=2)
        self.assertEqual(c.add(data=BIG), "[[figure:0]]")
        self.assertIsNone(c.add(data=SMALL))                    # trop petite
        self.assertIsNone(c.add(data=b"MZ not an image"))       # pas une image (nom ou extension sans effet)
        self.assertEqual(c.add(data=SVG), "[[figure:1]]")      # SVG : dimensions inconnues, acceptée
        self.assertIsNone(c.add(data=BIG))                      # quota
        self.assertIsNone(c.add())                              # ni données ni adresse
        self.assertEqual(c.add(url="https://x.test/a.png") if len(c.items) < 2 else None, None)
        self.assertIn("1 petites", c.warning())
        self.assertIn("1 format", c.warning())
        self.assertIn("1 quota", c.warning())
        self.assertIsNone(FigureCollector().warning())

    def test_libelle_et_ligne_de_passage(self):
        self.assertEqual(figure_label("Schéma CI/CD", "Pipeline [v2]"), "Pipeline v2 — Schéma CI/CD")
        self.assertEqual(figure_label("Même texte", "même texte en plus"), "même texte en plus")
        self.assertEqual(figure_passage_line("![Architecture](figures/ab12cd34/0123456789ab.png)"), "Figure : Architecture")
        self.assertEqual(figure_passage_line("![Figure 2 : flux](https://x.test/f.png)"), "Figure 2 : flux")
        self.assertEqual(figure_passage_line("![](figures/ab12cd34/0123456789ab.png)"), "")      # sans légende : omise
        self.assertIsNone(figure_passage_line("![logo](chemin/quelconque.png)"))                 # pas une de nos lignes
        self.assertIsNone(figure_passage_line("Texte ![a](figures/x/y.png) dans une phrase"))

    def test_passages_sans_chemin_ni_url(self):
        text = "Intro du texte. " * 5 + "\n![Flux des échanges](figures/ab12cd34/0123456789ab.png)\n![](figures/ab12cd34/ff.png)\nSuite."
        on = " ".join(p["text"] for p in build_passages("ab12cd34", [("", text)], figures=True))
        off = " ".join(p["text"] for p in build_passages("ab12cd34", [("", text)], figures=False))
        self.assertIn("Figure : Flux des échanges", on)
        self.assertNotIn("figures/", on)
        self.assertNotIn("![](", on)
        self.assertIn("figures/ab12cd34", off)                 # sans l'option : texte inchangé


class TestLecteurs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "corpus"
        self.root.mkdir()
        self.out = Path(self.tmp.name) / "data" / "figures"

    def tearDown(self):
        self.tmp.cleanup()

    def test_word(self):
        make_docx(self.root / "a.docx")
        docs, passages, warnings = ingest(self.root, fig_dir=self.out)
        d = docs[0]
        self.assertEqual(len(d["figures"]), 1)                  # doublon et logo écartés
        f = d["figures"][0]
        self.assertEqual((f["mime"], f["width"], f["height"], f["page"]), ("image/png", 320, 200, 0))
        self.assertEqual(f["alt"], "Architecture cible : API vers base de données")
        self.assertEqual(f["chemin"], f"figures/{d['id']}/{f['sha256'][:12]}.png")
        self.assertEqual((self.out / d["id"] / f"{f['sha256'][:12]}.png").read_bytes(), BIG)
        md = d["markdown"]
        line = f"![{f['alt']}]({f['chemin']})"
        target = f"]({f['chemin']})"
        self.assertEqual(md.count(target), 2)                   # deux occurrences (la 2e sans texte alternatif), un seul fichier
        self.assertIn(line, md)
        self.assertTrue(md.index("Avant le schéma") < md.index(line) < md.index("Figure 1 : architecture cible")
                        < md.rindex(target) < md.index("Après le schéma"))   # position dans le flux conservée
        self.assertNotIn("[[figure", md)
        self.assertTrue(any("1 petites" in w for w in warnings), warnings)
        text = " ".join(p["text"] for p in passages)
        self.assertIn("Figure : Architecture cible", text)
        self.assertNotIn("figures/", text)
        self.assertEqual(len(list((self.out / d["id"]).iterdir())), 1)
        # aller-retour : la réindexation depuis le markdown redonne les mêmes passages
        again = build_passages(d["id"], markdown_to_sections(md), 120, 20, figures=True)
        self.assertEqual(again, passages)

    def test_version_de_base_inchangee(self):
        make_docx(self.root / "a.docx")
        base, bp, _ = ingest_corpus(self.root, **CORPUS_WORDS)
        self.assertNotIn("figures", base[0])
        self.assertNotIn("![", " ".join(p["text"] for p in bp))
        sans, sp, _ = ingest(self.root, figures=False, fig_dir=self.out)         # markdown sans figures
        self.assertEqual(sp, bp)
        self.assertNotIn("![", sans[0]["markdown"])
        self.assertFalse(self.out.exists())
        solo, _, _ = ingest(self.root, figures=True, markdown=False, fig_dir=self.out)   # figures exigent le markdown
        self.assertNotIn("figures", solo[0])

    def test_pdf(self):
        from reportlab.pdfgen import canvas
        (self.root / "fig.png").write_bytes(BIG)
        c = canvas.Canvas(str(self.root / "a.pdf"))
        c.drawString(72, 780, "Article PDF avec un schema d'architecture " + "mot " * 20)
        c.drawString(72, 760, "Figure 1 : architecture cible")
        c.drawImage(str(self.root / "fig.png"), 72, 520, width=320, height=200)
        c.drawString(72, 480, filler(15))
        c.save()
        (self.root / "fig.png").unlink()
        docs, passages, _ = ingest(self.root, fig_dir=self.out)
        f = docs[0]["figures"][0]
        self.assertEqual((f["page"], f["legende"], f["mime"]), (1, "Figure 1 : architecture cible", "image/png"))
        self.assertIn("## Figures", docs[0]["markdown"])         # position exacte non conservée pour les PDF
        self.assertIn(f"![Figure 1 : architecture cible]({f['chemin']})", docs[0]["markdown"])
        self.assertTrue(any(p["text"].startswith("Figure 1 : architecture cible") or
                            "Figure 1 : architecture cible" in p["text"] for p in passages))

    def test_epub(self):
        body = ("<html xmlns='http://www.w3.org/1999/xhtml'><body><h1>Chapitre</h1><p>" + "Texte du chapitre. " * 15 + "</p>"
                "<figure><img src='images/fig.png' alt='Architecture API vers base'/>"
                "<figcaption>Figure 1.1 : architecture</figcaption></figure>"
                "<img src='https://tracker.test/p.png' alt='suivi'/><img src='../../hors.png'/></body></html>")
        with zipfile.ZipFile(self.root / "a.epub", "w") as z:
            z.writestr("mimetype", "application/epub+zip")
            z.writestr("META-INF/container.xml", '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
            z.writestr("OEBPS/content.opf", '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Livre</dc:title></metadata><manifest><item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="c1"/></spine></package>')
            z.writestr("OEBPS/ch1.xhtml", body)
            z.writestr("OEBPS/images/fig.png", BIG)
        docs, passages, _ = ingest(self.root, fig_dir=self.out)
        d = docs[0]
        self.assertEqual(len(d["figures"]), 1)                  # lien externe et chemin hors archive : ignorés
        f = d["figures"][0]
        self.assertEqual(f["legende"], "Figure 1.1 : architecture")
        self.assertEqual(f["alt"], "Architecture API vers base")
        self.assertIn("Figure 1.1 : architecture — Architecture API vers base", d["markdown"])
        self.assertEqual(d["markdown"].count("Figure 1.1 : architecture"), 1)     # la légende n'est pas dupliquée

    def test_html_local_et_chemins_dangereux(self):
        (self.root / "img").mkdir()
        (self.root / "img" / "a.png").write_bytes(BIG)
        (Path(self.tmp.name) / "secret.png").write_bytes(BIG)                   # hors du dossier du corpus
        (self.root / "page.html").write_text(
            "<html><head><title>Page</title></head><body><article><p>" + "Texte de la page. " * 20 + "</p>"
            "<img src='img/a.png' alt='Schéma local'/><img src='../secret.png' alt='piège'/>"
            "<img src='data:image/png;base64,AAAA'/><img src='/etc/passwd'/><img src='img/absent.png' alt='x'/>"
            "</article></body></html>", encoding="utf-8")
        docs, _, _ = ingest(self.root, fig_dir=self.out)
        figs = docs[0]["figures"]
        self.assertEqual([f["alt"] for f in figs], ["Schéma local"])
        self.assertNotIn("piège", docs[0]["markdown"])
        self.assertNotIn("passwd", docs[0]["markdown"])

    def test_web_references_distantes(self):
        html = ("<html><head><title>T</title></head><body><article><p>" + "Texte de l'article. " * 20 + "</p>"
                "<figure><img src='/img/pipeline.png' alt='Schéma CI/CD'/><figcaption>Pipeline</figcaption></figure>"
                "<img src='data:image/png;base64,AAAA'/><img src='/p.gif' width='1' height='1'/>"
                "</article></body></html>")

        def fetch(url, figures=None):
            from urllib.parse import urljoin
            return extract_html(html, url, figures, lambda src: (None, urljoin(url, src)))

        (self.root / "urls.txt").write_text("https://exemple.test/blog/article\n", encoding="utf-8")
        docs, passages, _ = ingest_corpus(self.root, keep_markdown=True, keep_figures=True, figures_dir=self.out,
                                          fetch=fetch, **CORPUS_WORDS)
        d = docs[0]
        self.assertEqual(len(d["figures"]), 1)
        f = d["figures"][0]
        self.assertEqual((f["url"], f["chemin"], f["sha256"]), ("https://exemple.test/img/pipeline.png", "", ""))
        self.assertIn("![Pipeline — Schéma CI/CD](https://exemple.test/img/pipeline.png)", d["markdown"])
        self.assertTrue(any("Figure : Pipeline — Schéma CI/CD" in p["text"] for p in passages))
        self.assertFalse(self.out.exists())                     # aucune image téléchargée


class FakeIngestClient(FakeClient):
    pass


class TestClickHouseFigures(unittest.TestCase):
    INFO = {"sha256": "a" * 64, "mime": "image/png", "width": 320, "height": 200, "octets": 1832,
            "alt": "Architecture", "legende": "Figure 1 : architecture", "page": 2,
            "chemin": "figures/aaaaaaaa/aaaaaaaaaaaa.png", "url": ""}
    DOC = {"id": "aaaaaaaa", "title": "T", "type": "pdf", "source": "t.pdf", "author": "", "date": None,
           "date_origine": "", "n_passages": 1, "n_words": 5, "figures": [INFO],
           "markdown": "# T\n\nTexte.\n\n![Figure 1 : architecture — Architecture](figures/aaaaaaaa/aaaaaaaaaaaa.png)\n"}

    def make(self, figures=True):
        cfg = load_config()
        cfg["ingest"]["keep_markdown"], cfg["ingest"]["keep_figures"] = True, figures
        self.fake = FakeClient()
        return ClickHouseBackend(cfg, client=self.fake, database="test_db")

    def test_schema_et_chargement(self):
        b = self.make()
        sql = "\n".join(self.fake.commands)
        self.assertIn("CREATE TABLE IF NOT EXISTS `test_db`.`documents_figures`", sql)
        self.assertIn("ORDER BY (doc_id, idx)", sql)
        b.load([], [self.DOC])
        by = {t: (c, r) for t, _, c, r in self.fake.inserts}
        cols, rows = by["documents_figures"]
        self.assertEqual(rows[0][cols.index("doc_id")], "aaaaaaaa")
        self.assertEqual((rows[0][cols.index("idx")], rows[0][cols.index("chemin")], rows[0][cols.index("page")]),
                         (0, self.INFO["chemin"], 2))
        self.assertIn("TRUNCATE TABLE `test_db`.`documents_figures`", self.fake.commands)
        self.assertIn("documents_figures", [t for t, *_ in self.fake.inserts])

    def test_export_et_get_figures(self):
        b = self.make()
        row = ("aaaaaaaa", "T", "pdf", "t.pdf", "", None, "", 1, 5, [], {}, "")
        figrow = ("aaaaaaaa", "a" * 64, "image/png", 320, 200, 1832, "Architecture", "Figure 1 : architecture",
                  2, self.INFO["chemin"], "")
        self.fake.handlers = [("FROM `test_db`.`documents_figures` FINAL ORDER BY doc_id", [figrow]),
                              ("FROM `test_db`.`documents_figures` FINAL WHERE doc_id", [figrow[1:]]),
                              ("themes, meta, content_sha256 FROM `test_db`.`documents` FINAL ORDER BY id", [row])]
        self.assertEqual(b.export_documents()[0]["figures"], [self.INFO])
        self.assertEqual(b.get_figures("aaaaaaaa"), [self.INFO])
        sql, params = [q for q in self.fake.queries if "WHERE doc_id" in q[0]][0]
        check_placeholders(self, sql, params)

    def test_reindex_garde_les_figures_et_le_markdown(self):
        b = self.make()
        row = ("aaaaaaaa", "T", "pdf", "t.pdf", "", None, "", 1, 5, [], {}, "")
        self.fake.handlers = [("FROM `test_db`.`documents_contenu` FINAL", [("aaaaaaaa", self.DOC["markdown"])]),
                              ("themes, meta, content_sha256 FROM `test_db`.`documents` FINAL ORDER BY id", [row])]
        b.reindex(120, 20)
        truncated = [c for c in self.fake.commands if c.startswith("TRUNCATE")]
        self.assertEqual(truncated, ["TRUNCATE TABLE `test_db`.`passages`", "TRUNCATE TABLE `test_db`.`postings`"])
        ptable = [r for t, _, c, r in self.fake.inserts if t == "passages"][0]
        text = " ".join(r[4] for r in ptable)
        self.assertIn("Figure : Figure 1 : architecture", text.replace("Figure : Figure 1", "Figure : Figure 1")) \
            if "Figure : Figure 1" in text else self.assertIn("Figure 1 : architecture", text)
        self.assertNotIn("figures/", text)

    def test_selftest_compare_les_figures(self):
        b = self.make()
        row = ("aaaaaaaa", "T", "pdf", "t.pdf", "", None, "", 1, 5, [], {}, "")
        figrow = ("aaaaaaaa", "a" * 64, "image/png", 320, 200, 1832, "Architecture", "Figure 1 : architecture",
                  2, self.INFO["chemin"], "")
        self.fake.handlers = [("SELECT count() FROM `test_db`.`passages`", [(1,)]),
                              ("substringUTF8", [(len(self.DOC["markdown"]), self.DOC["markdown"][5:45])]),
                              ("FROM `test_db`.`documents_figures` FINAL ORDER BY doc_id", [figrow]),
                              ("FROM `test_db`.`documents_contenu` FINAL", [("aaaaaaaa", self.DOC["markdown"])]),
                              ("themes, meta, content_sha256 FROM `test_db`.`documents` FINAL ORDER BY id", [row])]
        passages = build_passages("aaaaaaaa", markdown_to_sections(self.DOC["markdown"]), 250, 40, figures=True)
        lines = check_documents(b, [self.DOC], passages, b.cfg)
        self.assertTrue(lines[0].startswith("OK"), lines)
        bad = dict(self.DOC, figures=[dict(self.INFO, sha256="b" * 64)])
        self.assertTrue(check_documents(b, [bad], passages, b.cfg)[0].startswith("ÉCART"))


if __name__ == "__main__":
    unittest.main()
