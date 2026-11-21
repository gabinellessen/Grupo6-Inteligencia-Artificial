"""
Projeto: Agente Identificador de Fraudes Bancárias
Disciplina: Inteligência Artificial - 7º semestre - Ciência da Computação
Universidade Presbiteriana Mackenzie - Faculdade de Computação e Informática (FCI)
Professor: Prof. Dr. Ivan Carlos Alcântara de Oliveira

Integrantes:
    Cheuk Ki Yu                  - RA 10419664 - cheuk.yu@mackenzista.com.br
    Gabriela Nellessen de Sousa  - RA 10441930 - gabrielanellessen@gmail.com
    Milton Almeida Leoncio       - RA 10416764 - milton.almeida@live.com

Arquivo: src/figura_arquitetura.py
Síntese:
    Desenha a figura da arquitetura da solução (fluxo de dados, treinamento,
    avaliação e protótipo) usada no artigo e no README.

Histórico de alterações:
    2026-11-20 - Milton Almeida Leoncio - Versão inicial (N2).
"""
import comum as c


def desenhar():
    plt = c.preparar_matplotlib()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.95); ax.axis("off")

    def caixa(x, y, w, h, titulo, texto, cor):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor=cor, edgecolor="#6b6a64", linewidth=0.9))
        ax.text(x + w / 2, y + h - 0.2, titulo, ha="center", va="top", fontsize=9.5, weight="bold", color=c.TINTA)
        ax.text(x + w / 2, y + h - 0.52, texto, ha="center", va="top", fontsize=7.6, color=c.TINTA_2, linespacing=1.35)
        return (x, y, w, h)

    def seta(a, b, dx_a=1, dx_b=0, vertical=False):
        if vertical:
            p1 = (a[0] + a[2] / 2, a[1]); p2 = (b[0] + b[2] / 2, b[1] + b[3])
        else:
            p1 = (a[0] + a[2], a[1] + a[3] / 2); p2 = (b[0], b[1] + b[3] / 2)
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, color="#6b6a64", linewidth=1.1))

    azul, laranja, verde, cinza, rosa = "#e6effa", "#fdeee4", "#e3f3ec", "#f1f0ec", "#f8e6ef"
    b1 = caixa(0.1, 2.1, 2.2, 1.75, "1. Dados", "Base A - IBM\n24,4 mi transações\nBase B - Sparkov\n339,6 mil transações\n(sintéticas, públicas)", cinza)
    b2 = caixa(2.6, 2.1, 2.3, 1.75, "2. Preparação", "conversão de tipos\nremoção de nº do cartão,\nCVV, nome e endereço\njunção perfil/cartão", azul)
    b3 = caixa(5.2, 2.1, 2.4, 1.75, "3. Atributos", "temporais e geográficos\ncomportamentais por\ncartão/cliente (24h,\nmédia, novidade)\nsó com o passado", azul)
    b4 = caixa(7.9, 2.1, 1.9, 1.75, "4. Divisão\ntemporal", "\ntreino (passado)\nvalidação\nteste (futuro)", azul)
    b5 = caixa(10.1, 2.1, 1.8, 1.75, "5. Modelos", "regras, Reg. Log.,\nRF, XGBoost,\nLightGBM, MLP,\nSVM", laranja)
    b6 = caixa(7.9, 0.1, 4.0, 1.5, "6. Ajuste na validação", "balanceamento (pesos, subamostragem, SMOTE)\nOptuna (TPE) maximizando a AUPRC\nlimiar de decisão = maior F1\nescolha do modelo final", laranja)
    b7 = caixa(3.9, 0.1, 3.7, 1.5, "7. Avaliação no teste", "precisão, recall, F1, AUPRC, ROC-AUC\nIC 95% por bootstrap pareado\nmudança de conceito e retreino mensal\nviés por grupo e explicações (XAI)", verde)
    b8 = caixa(0.1, 0.1, 3.5, 1.5, "8. Protótipo", "escore de risco + limiar\nfila de alertas com motivos\npainel Streamlit / simulação\ndecisão final do analista", rosa)
    for a, b in [(b1, b2), (b2, b3), (b3, b4), (b4, b5)]:
        seta(a, b)
    seta(b5, b6, vertical=True)
    ax.add_patch(FancyArrowPatch((7.9, 0.85), (7.6, 0.85), arrowstyle="-|>", mutation_scale=12, color="#6b6a64", linewidth=1.1))
    ax.add_patch(FancyArrowPatch((3.9, 0.85), (3.6, 0.85), arrowstyle="-|>", mutation_scale=12, color="#6b6a64", linewidth=1.1))
    fig.tight_layout()
    fig.savefig(c.RESULTADOS / "fig_arquitetura.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    desenhar()
