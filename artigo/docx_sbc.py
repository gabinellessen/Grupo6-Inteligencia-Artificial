"""
Projeto: Detecção de fraudes em transações financeiras com aprendizado supervisionado
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - [e-mail a preencher]
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: artigo/docx_sbc.py
Síntese:
    Pequena camada sobre o python-docx para escrever o artigo no template oficial da
    SBC (sbc_template.docx, convertido do .doc distribuído no modelosparapublicaodeartigos.zip):
    estilos Title/Author/Address/Email/Abstract/Heading/Caption/Reference, recuo de
    1,27 cm a partir do segundo parágrafo de cada seção, figuras, tabelas e marcação
    simples de **negrito** e *itálico* no texto.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import re
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

MARCAS = re.compile(r"(\*\*.+?\*\*|\*.+?\*)")


def br(x, casas=3):
    return f"{x:.{casas}f}".replace(".", ",")


def milhar(n):
    return f"{int(n):,}".replace(",", ".")


class Artigo:
    def __init__(self, template):
        self.doc = docx.Document(template)
        corpo = self.doc.element.body
        for el in list(corpo):
            if el.tag != qn("w:sectPr"):
                corpo.remove(el)
        s = self.doc.sections[-1]
        s.page_width, s.page_height = Cm(21.0), Cm(29.7)
        s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Cm(3.5), Cm(2.5), Cm(3.0), Cm(3.0)
        self.n_fig = 0
        self.n_tab = 0
        self._primeiro = True

    # ---------------------------------------------------------------- texto
    def _runs(self, p, texto, tamanho=None):
        for parte in MARCAS.split(texto):
            if not parte:
                continue
            if parte.startswith("**"):
                r = p.add_run(parte[2:-2]); r.bold = True
            elif parte.startswith("*"):
                r = p.add_run(parte[1:-1]); r.italic = True
            else:
                r = p.add_run(parte)
            if tamanho:
                r.font.size = Pt(tamanho)
        return p

    def par(self, texto, estilo="Normal", recuo=None):
        p = self.doc.add_paragraph(style=estilo)
        self._runs(p, texto)
        if estilo == "Normal":
            if recuo is None:
                recuo = not self._primeiro
            if recuo:
                p.paragraph_format.first_line_indent = Cm(1.27)
            self._primeiro = False
        return p

    def lista(self, itens):
        """Itens em parágrafos com marcador, recuados como no template."""
        for it in itens:
            p = self.doc.add_paragraph(style="Normal")
            p.paragraph_format.left_indent = Cm(1.27)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.paragraph_format.space_before = Pt(3)
            self._runs(p, "• " + it)
        self._primeiro = False

    def titulo(self, texto):
        p = self.doc.add_paragraph(texto, style="Title")
        p.paragraph_format.first_line_indent = Cm(0)

    def autores(self, texto):
        self.doc.add_paragraph(texto, style="Author")

    def endereco(self, linhas):
        p = self.doc.add_paragraph(style="Address")
        for i, l in enumerate(linhas):
            r = p.add_run(l)
            if i < len(linhas) - 1:
                r.add_break()

    def email(self, texto):
        self.doc.add_paragraph(texto, style="Email")

    def resumo(self, rotulo, texto):
        p = self.doc.add_paragraph(style="Abstract")
        p.paragraph_format.right_indent = Cm(0.8)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(rotulo + ". "); r.bold = True
        self._runs(p, texto)

    def h1(self, texto):
        self.doc.add_paragraph(texto, style="Heading 1")
        self._primeiro = True

    def h2(self, texto):
        self.doc.add_paragraph(texto, style="Heading 2")
        self._primeiro = True

    # ---------------------------------------------------------------- figuras e tabelas
    def figura(self, arquivo, legenda, largura_cm=15.0):
        self.n_fig += 1
        p = self.doc.add_paragraph(style="Figure")
        p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(6)
        p.add_run().add_picture(str(arquivo), width=Cm(largura_cm))
        c = self.doc.add_paragraph(style="Caption")
        c.paragraph_format.right_indent = Cm(0.8)
        if len(legenda) > 90:
            c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.add_run(f"Figura {self.n_fig}. {legenda}")
        self._primeiro = False
        return self.n_fig

    def proxima_figura(self):
        return self.n_fig + 1

    def proxima_tabela(self):
        return self.n_tab + 1

    def tabela(self, cabecalho, linhas, legenda, larguras_cm=None, fonte=9, alinhar=None, nota=None):
        """Tabela com legenda acima (estilo Caption) e bordas simples, sem cores."""
        self.n_tab += 1
        c = self.doc.add_paragraph(style="Caption")
        c.paragraph_format.right_indent = Cm(0.8)
        c.paragraph_format.space_before = Pt(6)
        c.paragraph_format.keep_with_next = True
        if len(legenda) > 90:
            c.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        c.add_run(f"Tabela {self.n_tab}. {legenda}")
        t = self.doc.add_table(rows=1 + len(linhas), cols=len(cabecalho))
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        self._bordas(t)
        for j, h in enumerate(cabecalho):
            self._celula(t.cell(0, j), h, fonte, negrito=True, alinhar="c")
        for i, linha in enumerate(linhas, start=1):
            for j, v in enumerate(linha):
                a = (alinhar[j] if alinhar else ("l" if j == 0 else "c"))
                self._celula(t.cell(i, j), str(v), fonte, alinhar=a)
        if larguras_cm:
            for j, w in enumerate(larguras_cm):
                for cel in t.columns[j].cells:
                    cel.width = Cm(w)
        # repetir cabeçalho e não quebrar linhas entre páginas
        tr = t.rows[0]._tr
        trpr = tr.get_or_add_trPr()
        el = OxmlElement("w:tblHeader"); el.set(qn("w:val"), "true"); trpr.append(el)
        if nota:
            p = self.doc.add_paragraph(style="Normal")
            p.paragraph_format.space_before = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            self._runs(p, nota, tamanho=9)
        else:
            p = self.doc.add_paragraph(style="Normal")
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)

        self._primeiro = False
        return self.n_tab

    def _celula(self, cel, texto, fonte, negrito=False, alinhar="c"):
        cel.text = ""
        p = cel.paragraphs[0]
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.first_line_indent = Cm(0)
        p.alignment = {"c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT,
                       "r": WD_ALIGN_PARAGRAPH.RIGHT}[alinhar]
        self._runs(p, texto, tamanho=fonte)
        if negrito:
            for r in p.runs:
                r.bold = True

    @staticmethod
    def _bordas(t):
        tblpr = t._tbl.tblPr
        b = OxmlElement("w:tblBorders")
        for lado in ("top", "left", "bottom", "right", "insideH", "insideV"):
            e = OxmlElement(f"w:{lado}")
            e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "808080")
            b.append(e)
        tblpr.append(b)

    # ---------------------------------------------------------------- referências
    def referencias(self, refs):
        self.doc.add_paragraph("Referências", style="Heading 1")
        for r in sorted(refs, key=lambda s: s.lower()):
            p = self.doc.add_paragraph(style="Reference")
            p.paragraph_format.space_before = Pt(6)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT  # URLs longas não espalham a linha
            self._runs(p, r)

    def salvar(self, caminho):
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        self.doc.save(str(caminho))
