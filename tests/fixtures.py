"""Construit un petit corpus synthétique (docx, pdf, epub, md, html) pour les tests."""

import zipfile
from datetime import datetime
from pathlib import Path

PHRASES = [
    "Le marché européen évolue lentement sous la pression des régulateurs.",
    "Les assureurs adaptent leurs offres aux nouveaux comportements des clients.",
    "Cette transformation organisationnelle suppose de revoir les systèmes d'information.",
    "Les équipes techniques et les équipes métier travaillent désormais ensemble.",
    "Chaque trimestre, les indicateurs de performance sont présentés au comité de direction.",
    "Le rapport souligne la difficulté de standardiser les processus entre les filiales.",
]


def filler(n_words: int, seed: int = 0) -> str:
    out, i = [], seed
    while sum(len(s.split()) for s in out) < n_words:
        out.append(PHRASES[i % len(PHRASES)])
        i += 1
    return " ".join(out)


def build_corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    # Word : article daté de mars 2024
    import docx
    d = docx.Document()
    d.core_properties.title = "Sinistres habitation : la franchise augmente"
    d.core_properties.author = "Rédaction Assurance"
    d.core_properties.created = datetime(2024, 3, 12, 9, 0)
    d.add_heading("Franchise et sinistres", level=1)
    d.add_paragraph("Les assureurs relèvent la franchise des contrats d'assurance habitation après "
                    "la multiplication des sinistres liés aux intempéries. " + filler(120))
    d.add_heading("Réactions des associations", level=2)
    d.add_paragraph("Les associations de consommateurs dénoncent une hausse jugée excessive. " + filler(80, 2))
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Garantie", "Franchise"
    t.cell(1, 0).text, t.cell(1, 1).text = "Dégât des eaux", "250 euros"
    d.save(root / "article_habitation.docx")

    # PDF : rapport sur les inondations, plusieurs pages avec en-tête répété
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(root / "rapport_inondations.pdf"), pagesize=A4)
    c.setTitle("Rapport sur les inondations en Europe")
    c.setAuthor("Observatoire des risques")
    for page in range(1, 5):
        c.setFont("Helvetica", 9)
        c.drawString(60, 800, "Observatoire des risques - Rapport annuel")
        c.drawString(300, 30, f"Page {page}")
        c.setFont("Helvetica", 11)
        y = 760
        body = ("Les inondations ont touché plusieurs régions et provoqué des dégâts considérables. "
                "Le coût des catastrophes naturelles pour les assureurs a fortement augmenté. ") + filler(70, page)
        words = body.split()
        line = ""
        for w in words:
            if len(line) + len(w) > 85:
                c.drawString(60, y, line)
                y -= 15
                line = ""
            line += w + " "
        c.drawString(60, y, line)
        c.showPage()
    c.save()

    # EPUB : un « livre » de trois chapitres
    epub = root / "livre_transformation.epub"
    chapters = {
        "ch1.xhtml": ("Chapitre 1 : Les fondations", "L'histoire de l'assurance commence avec le commerce maritime. " + filler(400)),
        "ch2.xhtml": ("Chapitre 2 : La mutation numérique", "Le paléontologue de la donnée explore les archives des compagnies. " + filler(400, 3)),
        "ch3.xhtml": ("Chapitre 3 : Organisation", "L'organisation doit se réinventer autour de ses clients. " + filler(300, 1)),
    }
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        manifest = '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>'
        spine = ""
        for i, (fn, (title, text)) in enumerate(chapters.items()):
            manifest += f'<item id="c{i}" href="{fn}" media-type="application/xhtml+xml"/>'
            spine += f'<itemref idref="c{i}"/>'
            z.writestr(f"OEBPS/{fn}", f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>{title}</title></head>'
                                      f'<body><h1>{title}</h1><p>{text}</p></body></html>')
        z.writestr("OEBPS/nav.xhtml", "<html><body><nav><ol><li>Sommaire</li></ol></nav></body></html>")
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>La transformation des assureurs</dc:title>'
                   '<dc:creator>A. Auteur</dc:creator><dc:date>2019-05-01</dc:date><dc:language>fr</dc:language></metadata>'
                   f'<manifest>{manifest}</manifest><spine>{spine}</spine></package>')

    # Markdown
    (root / "note.md").write_text("# Note interne\n\nRéunion sur le projet de refonte du système d'information.\n\n"
                                  "## Décisions\n\nLe comité retient une architecture orientée événements. " + filler(60),
                                  encoding="utf-8")

    # HTML local avec métadonnées de date
    (root / "page_climat.html").write_text(
        '<html><head><title>Climat : les assureurs s\'inquiètent</title>'
        '<meta property="og:title" content="Climat : les assureurs s\'inquiètent">'
        '<meta property="article:published_time" content="2025-01-20T08:00:00+01:00"></head>'
        '<body><nav>Menu Accueil Contact</nav><article><h1>Climat : les assureurs s\'inquiètent</h1>'
        f'<p>La multiplication des tempêtes fragilise le modèle économique de l\'assurance. {filler(90)}</p></article>'
        '<footer>Mentions légales</footer></body></html>', encoding="utf-8")

    (root / "urls.txt").write_text("# articles en ligne\nhttps://exemple.test/2023/06/14/reassurance\n", encoding="utf-8")
    return root


def fake_fetch(url: str):
    from agentic_search.ingest import extract_html
    html = ('<html><head><title>Réassurance : les prix repartent à la hausse</title></head><body><article>'
            f'<p>Les réassureurs augmentent leurs tarifs. {filler(80)}</p></article></body></html>')
    return extract_html(html, url)
