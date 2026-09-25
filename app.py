from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Segmentação de risco por conta",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "df_contas_final.csv"
COLUNAS_ESPERADAS = {
    "account_id",
    "frequencia",
    "ticket_medio_winsor",
    "taxa_outlier",
    "cluster",
    "nome_cluster",
}


@st.cache_data
def carregar_dados(caminho: Path) -> pd.DataFrame:
    dados = pd.read_csv(caminho)
    faltantes = COLUNAS_ESPERADAS.difference(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltantes)}")
    return dados


df_contas_final = carregar_dados(CSV_PATH)

# Regra aprovada no notebook: cluster 1 OU taxa de outliers de pelo menos 10%.
mascara_prioridade = (
    df_contas_final["cluster"].eq(1)
    | df_contas_final["taxa_outlier"].ge(0.10)
)
contas_prioritarias = (
    df_contas_final.loc[mascara_prioridade]
    .sort_values(
        ["taxa_outlier", "ticket_medio_winsor", "frequencia", "account_id"],
        ascending=[False, False, False, True],
    )
)

total_contas = len(df_contas_final)
total_transacoes = int(df_contas_final["frequencia"].sum())
total_flags = int(round(
    (df_contas_final["frequencia"] * df_contas_final["taxa_outlier"]).sum()
))

st.title("Segmentação de risco por conta")
st.caption(
    "Painel acadêmico para priorização de revisão. "
    "As flags indicam atipicidade estatística, não fraude."
)

kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
kpi_1.metric("Contas", f"{total_contas}")
kpi_2.metric("Transações", f"{total_transacoes:,}".replace(",", "."))
kpi_3.metric("Flags de outlier", f"{total_flags} ({total_flags / total_transacoes:.2%})")
kpi_4.metric("Contas para revisão", f"{len(contas_prioritarias)} ({len(contas_prioritarias) / total_contas:.0%})")

st.subheader("Ticket médio e taxa de outliers")
fig_scatter = px.scatter(
    df_contas_final,
    x="ticket_medio_winsor",
    y="taxa_outlier",
    color="nome_cluster",
    hover_name="account_id",
    hover_data={
        "frequencia": True,
        "ticket_medio_winsor": ":.2f",
        "taxa_outlier": ":.2%",
        "cluster": True,
        "nome_cluster": False,
    },
    labels={
        "ticket_medio_winsor": "Ticket médio winsorizado",
        "taxa_outlier": "Taxa de outliers",
        "nome_cluster": "Cluster",
    },
    color_discrete_sequence=["#355C7D", "#C06C84", "#6C9A8B"],
)
fig_scatter.add_hline(
    y=0.10,
    line_dash="dash",
    line_color="#7A7A7A",
    annotation_text="Limite de 10%",
    annotation_position="top left",
)
fig_scatter.add_trace(
    go.Scatter(
        x=contas_prioritarias["ticket_medio_winsor"],
        y=contas_prioritarias["taxa_outlier"],
        mode="markers+text",
        text=contas_prioritarias["account_id"],
        textposition="top center",
        marker={
            "size": 17,
            "color": "rgba(0,0,0,0)",
            "line": {"color": "#111111", "width": 2},
        },
        name="Selecionadas para revisão",
        hoverinfo="skip",
    )
)
fig_scatter.update_yaxes(tickformat=".0%", rangemode="tozero")
fig_scatter.update_xaxes(rangemode="tozero")
fig_scatter.update_layout(
    height=500,
    legend_title_text="Cluster",
    margin={"l": 20, "r": 20, "t": 20, "b": 20},
)
st.plotly_chart(fig_scatter, width="stretch")

col_perfis, col_prioridades = st.columns([1, 1.35], gap="large")

with col_perfis:
    st.subheader("Perfil dos clusters")
    perfil_clusters = (
        df_contas_final.groupby(["cluster", "nome_cluster"], as_index=False)
        .agg(
            contas=("account_id", "size"),
            frequencia_media=("frequencia", "mean"),
            ticket_medio_winsor=("ticket_medio_winsor", "mean"),
            taxa_outlier_media=("taxa_outlier", "mean"),
        )
    )
    perfil_exibicao = perfil_clusters[
        [
            "cluster",
            "contas",
            "frequencia_media",
            "ticket_medio_winsor",
            "taxa_outlier_media",
        ]
    ].copy()
    perfil_exibicao["frequencia_media"] = perfil_exibicao["frequencia_media"].round(2)
    perfil_exibicao["ticket_medio_winsor"] = perfil_exibicao["ticket_medio_winsor"].round(2)
    perfil_exibicao["taxa_outlier_media"] = perfil_exibicao["taxa_outlier_media"].map("{:.2%}".format)
    st.dataframe(
        perfil_exibicao,
        hide_index=True,
        width="stretch",
        column_config={
            "cluster": "Cluster",
            "contas": "Contas",
            "frequencia_media": "Freq. média",
            "ticket_medio_winsor": "Ticket médio",
            "taxa_outlier_media": "Outliers",
        },
    )

with col_prioridades:
    st.subheader("Contas prioritárias")
    st.caption("Critério aprovado: cluster 1 OU taxa de outliers ≥ 10%.")
    tabela_prioridades = contas_prioritarias[
        [
            "account_id",
            "cluster",
            "nome_cluster",
            "frequencia",
            "ticket_medio_winsor",
            "taxa_outlier",
        ]
    ]
    st.dataframe(
        tabela_prioridades,
        hide_index=True,
        width="stretch",
        column_config={
            "account_id": "Conta",
            "cluster": "Cluster",
            "nome_cluster": "Perfil",
            "frequencia": "Transações",
            "ticket_medio_winsor": st.column_config.NumberColumn(
                "Ticket médio", format="%.2f"
            ),
            "taxa_outlier": st.column_config.NumberColumn(
                "Taxa de outliers", format="percent"
            ),
        },
    )

with st.expander("Metodologia e fonte"):
    st.markdown(
        """
        - **Fonte:** `df_contas_final.csv`, exportado do notebook da atividade.
        - **Features:** frequência, ticket médio winsorizado e taxa de outliers.
        - **Modelo:** StandardScaler e K-Means com K = 3.
        - **Priorização:** cluster 1 OU taxa de outliers ≥ 10%.
        - As flags foram calculadas antes da winsorização e não comprovam fraude.
        """
    )
