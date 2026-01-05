import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modulos.data_loader import SFODataLoader
from modulos.risk_engine import RiskEngine

# --- 1. CONFIGURAÇÃO PAGINA E CSS ---
st.set_page_config(
    page_title="Gestão de Risco SFO",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None
)

# Estilo CSS Minimalista e Corporativo
st.markdown("""
    <style>
        /* Remover padding excessivo do topo */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Estilização das Métricas */
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 10px;
            border-radius: 5px;
            color: #212529;
        }
        
        label[data-testid="stMetricLabel"] {
            color: #6c757d;
            font-size: 0.9rem;
        }
        
        div[data-testid="stMetricValue"] {
            color: #0d3b66; /* Azul Marinho */
            font-weight: 600;
        }

        /* Ajuste de fontes gerais */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #212529;
        }
        
        /* Títulos */
        h1, h2, h3 {
            color: #0d3b66; /* Azul Marinho Corporativo */
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. CACHE E INICIALIZAÇÃO ---
@st.cache_resource
def carregar_sistema_v2():
    print("DEBUG: Iniciando carregar_sistema_v2 com Quant_Fundos.xlsm")
    # O arquivo está na raiz, conforme verificado
    loader = SFODataLoader('Quant_Fundos.xlsm')
    engine = RiskEngine()
    return loader, engine

try:
    loader, engine = carregar_sistema_v2()
except Exception as e:
    st.error(f"Erro crítico ao carregar sistema: {e}")
    st.image("https://http.cat/500", width=400) # Fallback visual
    st.stop()

# Exibir data mais recente dos dados
data_mais_recente = loader.obter_data_mais_recente()
if data_mais_recente:
    st.info(f"**Dados atualizados até:** {data_mais_recente.strftime('%d/%m/%Y')}")

# --- 3. BARRA LATERAL (CONTROLES GLOBAIS) ---
with st.sidebar:
    st.header("Parâmetros")
    
    # Seleção de Ativos
    fundos_disp = loader.listar_ativos('Fundo')
    benchs_disp = loader.listar_ativos('Benchmark')
    
    fundo_selecionado = st.selectbox("Ativo Principal", options=fundos_disp)
    
    benchs_possiveis = [b for b in benchs_disp if b != fundo_selecionado]
    default_benchs = [b for b in ["IBOV", "CDI", "PTAXV"] if b in benchs_possiveis]
    
    benchmarks_selecionados = st.multiselect("Comparativo (Benchmarks)", options=benchs_possiveis, default=default_benchs)
    
    st.markdown("#### Seleção de Pares (Peers)")
    # Filtros para os fundos de comparação
    with st.expander("Filtros Avançados de Busca"):
        # 1. Classe
        classes_disp = ["Todas"] + loader.listar_classes(tipo='Fundo')
        filtro_classe = st.selectbox("Classe:", options=classes_disp, index=0, key="filt_classe_comp")
        
        # 2. Subclasse (Dinâmico base classe)
        classe_search = filtro_classe if filtro_classe != "Todas" else None
        subclasses_disp = ["Todas"] + loader.listar_subclasses(tipo='Fundo', classe=classe_search)
        filtro_subclasse = st.selectbox("Subclasse:", options=subclasses_disp, index=0, key="filt_sub_comp")
        
        # 3. Grupo (Dinâmico)
        sub_search = filtro_subclasse if filtro_subclasse != "Todas" else None
        grupos_disp = ["Todos"] + loader.listar_grupos(tipo='Fundo', classe=classe_search, subclasse=sub_search)
        filtro_grupo = st.selectbox("Grupo:", options=grupos_disp, index=0, key="filt_grupo_comp")

    # Aplicar filtros
    grupo_search = filtro_grupo if filtro_grupo != "Todos" else None
    
    fundos_filtrados = loader.listar_ativos(
        'Fundo', 
        classe=classe_search, 
        subclasse=sub_search, 
        grupo=grupo_search
    )
    
    # Remover o fundo principal da lista de comparação e garantir que só mostramos o que foi filtrado
    fundos_possiveis = [f for f in fundos_filtrados if f != fundo_selecionado]
    fundos_comparacao = st.multiselect("Comparar com outros Fundos", options=fundos_possiveis, default=[])
    
    st.divider()
    
    # --- NOVO: SELETOR DE PERÍODO (Igual app2.py) ---
    st.markdown("### Período de Análise")
    periodo_opcao = st.radio(
        "Janela de Visualização:",
        options=["3 Meses", "6 Meses", "12 Meses", "24 Meses", "36 Meses", "Desde o Início"],
        index=2 # Default 12m
    )
    
    # Configuração Técnica (Fixo)
    janela_rolling = 63
    
    st.divider()
    atualizar = st.button("Atualizar Dados", type="primary")

# --- 4. CORPO PRINCIPAL ---

st.title("Relatório de Risco e Performance")

if not atualizar and 'dados_carregados' not in st.session_state:
    st.info("Configure os parâmetros para gerar o relatório.")
    st.stop()

st.session_state['dados_carregados'] = True
    
# Processamento de Dados
ativos_totais = [fundo_selecionado] + benchmarks_selecionados + fundos_comparacao
df_raw = loader.buscar_dados(ativos_totais)

if df_raw.empty:
    st.error("Não há dados disponíveis para os ativos selecionados no período comum.")
    st.stop()

# Corte Inicial (Data Valida do Fundo)
first_idx = df_raw[fundo_selecionado].first_valid_index()
if not first_idx:
    st.warning("O fundo selecionado não possui dados válidos.")
    st.stop()

# CORRIGIDO: Não preencher com zeros - zeros artificiais distorcem métricas de risco
# Cada cálculo tratará NaNs apropriadamente
df_dados_full = df_raw.loc[first_idx:].copy()

# Slicing pelo Período Selecionado
mapa_dias = {
    "3 Meses": 63, "6 Meses": 126, "12 Meses": 252, 
    "24 Meses": 504, "36 Meses": 756, "Desde o Início": len(df_dados_full)
}
dias_corte = min(mapa_dias[periodo_opcao], len(df_dados_full))
df_dados = df_dados_full.iloc[-dias_corte:].copy()

# Séries Individuais (VIEW)
s_fundo = df_dados[fundo_selecionado]
# Benchmark primário
nome_bench_pri = benchmarks_selecionados[0] if benchmarks_selecionados else None
s_bench_pri = df_dados[nome_bench_pri] if nome_bench_pri else None

# Cálculos do Engine
# --- Taxa Livre de Risco (CDI do Período) ---
try:
    df_cdi = loader.buscar_dados(['CDI'])
    s_cdi = df_cdi['CDI'].loc[s_fundo.index].dropna() # Alinha com o fundo
    if not s_cdi.empty:
        # Retorno Composto Anualizado do CDI no período
        ret_cdi_per = (1 + s_cdi).prod() - 1
        rf_anualizado = (1 + ret_cdi_per) ** (252 / len(s_cdi)) - 1
    else:
        rf_anualizado = 0.0
except:
    rf_anualizado = 0.0

# Retornos e Métricas no Período
metricas = engine.calcular_metricas_risco_retorno(s_fundo, rf=rf_anualizado)
retornos_janela = engine.calcular_retornos_periodo(s_fundo)

# CORRIGIDO: Calcular rolling stats no DATASET COMPLETO primeiro
# Isso garante que temos dados suficientes para todas as janelas (1m, 3m, 6m, 12m)
# mesmo quando o usuário seleciona um período curto para visualização
s_fundo_full = df_dados_full[fundo_selecionado].dropna()
s_bench_pri_full = df_dados_full[nome_bench_pri].dropna() if nome_bench_pri else None

# Calcular stats no histórico completo
stats_series_full = engine.calcular_rolling_stats(s_fundo_full, s_bench_pri_full)

# Cortar para o período selecionado APENAS para visualização
stats_series = stats_series_full.loc[s_fundo.index] if not stats_series_full.empty else pd.DataFrame()

# Drawdown calculado no período de visualização (comportamento esperado)
drawdown_series, max_dd = engine.calcular_drawdown(s_fundo)

# --- 5. VISUALIZAÇÃO ---

# Paleta de Cores Global (Solicitada pelo Usuário)
paleta_cores = [
    "#1F77B4", # Azul Royal
    "#FF7F0E", # Laranja Vivo
    "#2CA02C", # Verde Esmeralda
    "#D62728", # Vermelho Carmim
    "#9467BD", # Roxo Ametista
    "#8C564B", # Marrom Café
    "#E377C2"  # Rosa Choque
]

# Cabeçalho do Fundo
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.markdown(f"### {fundo_selecionado}")
    st.markdown(f"**Início:** {first_idx.strftime('%d/%m/%Y')}")

with col_header_2:
    # Retorno acumulado do período visualizado (s_fundo já está cortado)
    # Tratar caso de série vazia
    if not s_fundo.empty:
        ret_acumulado_periodo = (1 + s_fundo).prod() - 1
        label_periodo = periodo_opcao.replace("Meses", "m").replace("Desde o Início", "Início")
        st.metric(f"Retorno Acumulado ({label_periodo})", f"{ret_acumulado_periodo:.2%}")

st.markdown("---")

# Abas de Conteúdo
tab_perf, tab_risk, tab_style = st.tabs(["Performance", "Risco & Drawdown", "Análise de Fatores"])

# === ABA 1: PERFORMANCE ===
with tab_perf:
    st.write("") # Espaçamento
    
    # KPIs Principais
    kp1, kp2, kp3, kp4 = st.columns(4)
    kp1.metric("Retorno 12 Meses", f"{retornos_janela.get('12m', 0.0):.2%}" if pd.notna(retornos_janela.get('12m')) else "-")
    kp2.metric("Volatilidade Anual", f"{metricas['Volatilidade_Ano']:.2%}")
    kp3.metric("Sharpe Ratio", f"{metricas['Sharpe']:.2f}")
    kp4.metric("Calmar Ratio", f"{metricas['Calmar']:.2f}")
    
    st.markdown("#### Evolução Patrimonial (Base 100)")
    
    # Gráfico de Linha Base 100
    # Para o gráfico de performance, forward fill é apropriado (carrega último valor conhecido)
    df_dados_filled = df_dados.fillna(method='ffill')
    
    df_norm = (1 + df_dados_filled).cumprod() * 100
    df_norm = df_norm / df_norm.iloc[0] * 100
    
    # Cores personalizadas
    cores = {}
    
    # Atribuir cores
    cores[fundo_selecionado] = paleta_cores[0] # Azul Royal - Destaque
    
    # Gerar cores para comparativos (Benchmarks + Fundos)
    # Paleta estendida para suportar mais itens se necessário
    paleta_secundaria = paleta_cores[1:] + px.colors.qualitative.Prism
    
    # Lista combinada para iterar e atribuir cores
    ativos_comparativos = benchmarks_selecionados + fundos_comparacao
    
    for i, b in enumerate(ativos_comparativos):
        # Proteção para index out of bounds se tiver muiiitos ativos
        cor_idx = i % len(paleta_secundaria)
        cores[b] = paleta_secundaria[cor_idx]
        
    fig_perf = px.line(df_norm, x=df_norm.index, y=df_norm.columns, color_discrete_map=cores)
    
    # Configurar espessura das linhas: Fundo principal mais grosso
    fig_perf.update_traces(line=dict(width=1.5))
    fig_perf.update_traces(selector=dict(name=fundo_selecionado), line=dict(width=3.5))

    # Customizar tooltip para melhor legibilidade
    fig_perf.update_traces(
        hovertemplate="<b>%{fullData.name}</b><br>" +
                      "Valor: %{y:.2f}<br>" +
                      "<extra></extra>"
    )
    
    fig_perf.update_layout(
        template="plotly_white",
        xaxis_title="",
        yaxis_title="Base 100",
        legend=dict(title=None, orientation="h", y=1.02, yanchor="bottom", x=0.5, xanchor="center"),
        hovermode="x unified",
        margin=dict(t=50),
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_family="Segoe UI, Roboto, sans-serif"
        )
    )
    
    # Formatar eixo X para mostrar data de forma legível
    fig_perf.update_xaxes(
        hoverformat="%d/%m/%Y"
    )
    
    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.divider()
    
    # --- GRÁFICO DE JANELAS MÓVEIS (NOVO) ---
    st.markdown("#### Análise de Janelas Móveis")
    
    c_roll_opts, c_roll_chart = st.columns([1, 3])
    
    with c_roll_opts:
        tipo_metrica_movel = st.radio(
            "Métrica:",
            options=["Retorno Efetivo", "% CDI", "Volatilidade Anualizada"],
            key="metrica_movel_radio"
        )
        
        # Filtra "Desde o Início" das opções globais pois não faz sentido para janela fixa
        opcoes_janela_movel = [opt for opt in mapa_dias.keys() if opt != "Desde o Início"]
        
        janela_movel_sel = st.selectbox(
            "Janela de Observação:",
            options=opcoes_janela_movel,
            index=2, # Default 12 Meses
            key="sel_janela_movel"
        )
        
    days_rolling = mapa_dias[janela_movel_sel]
    
    # Preparacao das Series full para calculo correto da janela
    ativos_janela = [fundo_selecionado] + benchmarks_selecionados + fundos_comparacao
    
    df_plot_roll = pd.DataFrame()
    lbl_y = ""
    tk_fmt = ".2%"
    
    # Cache do CDI se necessario
    s_cdi_calc = None
    if tipo_metrica_movel == "% CDI":
        if 'CDI' in df_dados_full.columns:
             s_cdi_calc = df_dados_full['CDI']
        else:
             df_cdi_tmp = loader.buscar_dados(['CDI'])
             if not df_cdi_tmp.empty and 'CDI' in df_cdi_tmp.columns:
                s_cdi_calc = df_cdi_tmp['CDI']
    
    for ativo in ativos_janela:
        if ativo not in df_dados_full.columns:
            continue
            
        s_calc = df_dados_full[ativo].dropna()
        if s_calc.empty:
            continue
            
        series_res = None
        
        if tipo_metrica_movel == "Retorno Efetivo":
            # Retorno Composto na Janela: (1+r)^T - 1
            series_res = (1 + s_calc).rolling(window=days_rolling).apply(np.prod, raw=True) - 1
            lbl_y = f"Retorno Efetivo ({janela_movel_sel})"
            
        elif tipo_metrica_movel == "Volatilidade Anualizada":
            # Std * sqrt(252)
            series_res = s_calc.rolling(window=days_rolling).std() * (252**0.5)
            lbl_y = f"Volatilidade ({janela_movel_sel})"
            
        elif tipo_metrica_movel == "% CDI":
            if s_cdi_calc is not None:
                # Alinhar dados
                df_concat = pd.concat([s_calc, s_cdi_calc], axis=1).dropna()
                sf = df_concat.iloc[:, 0]
                sc = df_concat.iloc[:, 1]
                
                ret_f = (1 + sf).rolling(window=days_rolling).apply(np.prod, raw=True) - 1
                ret_c = (1 + sc).rolling(window=days_rolling).apply(np.prod, raw=True) - 1
                
                # Evitar divisao por zero
                series_res = (ret_f / ret_c) 
                lbl_y = f"% CDI ({janela_movel_sel})"
        
        if series_res is not None:
            df_plot_roll[ativo] = series_res

    with c_roll_chart:
        if not df_plot_roll.empty:
            # Filtrar para exibir apenas o período selecionado na visualização global
            idx_view = s_fundo.index
            # Reindex para alinhar com o zoom da página
            df_viz = df_plot_roll.reindex(idx_view)
            
            # --- KPI's da Janela ---
            kp_j1, kp_j2 = st.columns(2)
            
            if fundo_selecionado in df_viz.columns:
                media_periodo = df_viz[fundo_selecionado].mean()
                kp_j1.metric(f"Média no Período", f"{media_periodo:.2%}")
                
                # Cálculo de % do tempo acima do benchmark primário
                bench_ref = benchmarks_selecionados[0] if benchmarks_selecionados else None
                
                if bench_ref and bench_ref in df_viz.columns:
                    # Compara linha a linha na visualização atual
                    s_f = df_viz[fundo_selecionado]
                    s_b = df_viz[bench_ref]
                    
                    # Calcula proporção
                    acima_bench = (s_f > s_b).mean()
                    kp_j2.metric(f"% do Tempo Acima de {bench_ref}", f"{acima_bench:.0%}")
                else:
                    kp_j2.metric("% do Tempo Acima do Bench", "-")
            
            # Plotar series compostas
            fig_roll = px.line(df_viz, x=df_viz.index, y=df_viz.columns, color_discrete_map=cores)
            
            # Adicionar linha de referência 100% se for % CDI
            if tipo_metrica_movel == "% CDI":
                 fig_roll.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="100% CDI")
            
            # Configurar espessura das linhas
            fig_roll.update_traces(line=dict(width=1.5))
            if fundo_selecionado in df_viz.columns:
                 fig_roll.update_traces(selector=dict(name=fundo_selecionado), line=dict(width=3.5))

            fig_roll.update_layout(
                template="plotly_white",
                title=dict(text="Janela Móvel", font=dict(size=14, color="#0d3b66")),
                xaxis_title="",
                yaxis_title=lbl_y,
                yaxis_tickformat=tk_fmt,
                margin=dict(t=30, b=10, l=10, r=10),
                height=350,
                hovermode="x unified",
                showlegend=True, # Agora mostra legenda com multiplos ativos
                legend=dict(title=None, orientation="h", y=1.1, yanchor="bottom", x=0.5, xanchor="center")
            )
            
            # Hover template unificado
            fig_roll.update_traces(
                hovertemplate="<b>%{fullData.name}</b><br>Valor: %{y:.2%}<extra></extra>"
            )
            
            st.plotly_chart(fig_roll, use_container_width=True)
        else:
            st.info("Dados insuficientes para calcular esta métrica na janela selecionada.")

    st.markdown("#### Matriz de Correlação")
    # Usar apenas pares com dados válidos (pairwise deletion)
    corr_matrix = df_dados.corr(method='pearson', min_periods=30)
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale="Blues")
    fig_corr.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_corr, use_container_width=True)

# === ABA 2: RISCO ===
with tab_risk:
    st.write("")
    
    col_r1, col_r2 = st.columns(2)
    
    # Gráfico de Volatilidade
    with col_r1:
        st.markdown(f"#### Volatilidade Móvel ({janela_rolling} dias)")
        
        cols_vol = [c for c in stats_series.columns if 'Vol_' in c]
        if cols_vol and not stats_series[cols_vol].dropna().empty:
            fig_vol = px.line(stats_series[cols_vol], color_discrete_sequence=paleta_cores)
            fig_vol.update_layout(
                template="plotly_white", 
                showlegend=True, 
                legend_title=None, 
                xaxis_title="", 
                yaxis_title="Volatilidade Anualizada",
                yaxis_tickformat=".2%"
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Dados insuficientes para cálculo de volatilidade móvel.")

    # Gráfico de Beta
    with col_r2:
        st.markdown(f"#### Beta Móvel")
        
        if benchmarks_selecionados:
            # Tenta definir Ibovespa como padrão
            idx_def = 0
            if "Ibovespa" in benchmarks_selecionados:
                idx_def = benchmarks_selecionados.index("Ibovespa")
                
            bench_beta_sel = st.selectbox(
                "Selecione o Benchmark para o Beta:",
                options=benchmarks_selecionados,
                index=idx_def
            )
            
            # CORRIGIDO: Recalcular stats com benchmark selecionado usando dataset completo
            s_bench_beta_full = df_dados_full[bench_beta_sel].dropna()
            stats_beta_full = engine.calcular_rolling_stats(s_fundo_full, s_bench_beta_full)
            
            # Cortar para período de visualização
            stats_beta = stats_beta_full.loc[s_fundo.index] if not stats_beta_full.empty else pd.DataFrame()
            
            # Filtrar apenas as janelas de interesse (6m e 12m)
            cols_beta = [c for c in stats_beta.columns if c in ['Beta_6m', 'Beta_12m']]
            
            if cols_beta and not stats_beta[cols_beta].dropna(how='all').empty:
                # Usa paleta_cores para diferenciar os períodos
                fig_beta = px.line(stats_beta[cols_beta], color_discrete_sequence=paleta_cores)
                fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutro")
                fig_beta.update_layout(
                    template="plotly_white", 
                    showlegend=True, 
                    legend_title="Janela", 
                    xaxis_title="", 
                    yaxis_title="Beta"
                )
                st.plotly_chart(fig_beta, use_container_width=True)
            else:
                st.info("Dados insuficientes para cálculo de Beta neste período.")
        else:
            st.info("Selecione um benchmark válido para visualizar o Beta.")

    st.markdown("#### Underwater Plot (Drawdown)")
    
    # Seletor de comparação para Drawdown
    opts_dd = benchmarks_selecionados + fundos_comparacao
    dd_comps = st.multiselect("Comparar com:", options=opts_dd, key="dd_comps_multiselect")
    
    # Area chart para Drawdown
    fig_dd = go.Figure()
    
    # Fundo Principal (Area Preenchida)
    fig_dd.add_trace(go.Scatter(
        x=drawdown_series.index, 
        y=drawdown_series, 
        fill='tozeroy', 
        mode='lines',
        line=dict(color=paleta_cores[3], width=2),
        name=fundo_selecionado
    ))
    
    # Adicionar comparativos (Linhas)
    for i, comp in enumerate(dd_comps):
        if comp in df_dados.columns:
             s_comp = df_dados[comp].dropna()
             if not s_comp.empty:
                 # Calcula drawdown usando a engine
                 dd_comp, _ = engine.calcular_drawdown(s_comp)
                 
                 # Escolher cor: Usando paleta "Bold" ou "Dark24" para cores mais escuras e distintas
                 paleta_escura = px.colors.qualitative.Bold
                 cor_comp = paleta_escura[i % len(paleta_escura)]
                 
                 fig_dd.add_trace(go.Scatter(
                    x=dd_comp.index, 
                    y=dd_comp, 
                    mode='lines',
                    line=dict(color=cor_comp, width=2), # Linha sólida (padrão) e um pouco mais grossa
                    name=comp
                ))
    fig_dd.update_layout(
        template="plotly_white",
        yaxis_tickformat=".2%",
        xaxis_title="",
        yaxis_title="Queda do Topo",
        margin=dict(t=20)
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.metric("Drawdown Máximo do Período", f"{max_dd:.2%}")


# === ABA 3: Fatores e Estilo ===
with tab_style:
    st.write("")
    st.markdown("#### Análise de Estilo Baseada em Retornos (RBSA)")
    st.caption("Regressão multivariada dos retornos do fundo contra os benchmarks selecionados para identificar sensibilidade.")
    
    if len(benchmarks_selecionados) >= 1:
        # Preparar dados para regressão
        df_indicies = df_dados[benchmarks_selecionados]
        
        coefs, r2, fitted = engine.analise_estilo_rbsa(s_fundo, df_indicies)
        
        if coefs:
            col_s1, col_s2 = st.columns([1, 2])
            
            with col_s1:
                st.markdown("##### Estatísticas do Modelo")
                st.metric("R² (Coef. de Determinação)", f"{r2:.2%}")
                st.markdown("""
                **Interpretação:**
                O R² indica quanto da variação do fundo é "explicada" pelos movimentos dos índices selecionados.
                * R² > 80%: Fundo segue muito os índices (Beta alto/Passivo).
                * R² < 50%: Fundo tem muito "alpha" ou risco idiossincrático.
                """)
            
            with col_s2:
                st.markdown("##### Exposição aos Fatores")
                # Converter dict para DF
                df_coefs = pd.DataFrame(list(coefs.items()), columns=['Fator', 'Sensibilidade'])
                
                # Remover constante/Alpha e manter o resto (incluindo Residual_Cash)
                df_coefs = df_coefs[~df_coefs['Fator'].isin(['const', 'Alpha'])]
                
                fig_bar = px.bar(df_coefs, x='Fator', y='Sensibilidade', text_auto='.2f', color='Fator')
                
                # Definir cores manuais para garantir que Cash seja cinza
                color_map_bar = {k: paleta_cores[0] for k in df_coefs['Fator'].unique()}
                cols_b = [c for c in benchmarks_selecionados] # Cores rotativas para benchmarks
                for i, b in enumerate(cols_b):
                    if b in color_map_bar:
                         color_map_bar[b] = paleta_cores[i % len(paleta_cores)]
                
                color_map_bar['Residual_Cash'] = '#D3D3D3'
                
                fig_bar.update_traces(textfont_size=12, textangle=0, cliponaxis=False)
                fig_bar.update_layout(
                    template="plotly_white", 
                    yaxis_title="Peso (%)", 
                    showlegend=False,
                    xaxis_title=None
                )
                fig_bar.update_traces(marker_color=[color_map_bar.get(x, paleta_cores[0]) for x in df_coefs['Fator']])
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            st.divider()
            
            # --- NOVO: Gráfico de Evolução Histórica (Rolling RBSA) ---
            st.markdown("#### Evolução Histórica do Estilo")
            
            col_sel_rbsa, _ = st.columns([1, 2])
            with col_sel_rbsa:
                janela_rbsa_sel = st.selectbox(
                    "Janela de Análise (Rolling):",
                    options=["3 Meses", "6 Meses", "12 Meses", "24 Meses"],
                    index=1 # Padrão 6 Meses
                )
            
            # Mapear seleção para dias
            mapa_janela_rbsa = {
                "3 Meses": 63,
                "6 Meses": 126,
                "12 Meses": 252,
                "24 Meses": 504
            }
            window_days = mapa_janela_rbsa[janela_rbsa_sel]
            
            st.caption(f"Como a exposição aos fatores mudou ao longo do tempo (Janela Móvel de {janela_rbsa_sel}).")
            
            # Preparar dados para o rolling (usar dados full para ter histórico)
            s_fundo_rbsa = df_dados_full[fundo_selecionado].dropna()
            df_benchs_rbsa = df_dados_full[benchmarks_selecionados].dropna()
            
            # Calcular Rolling RBSA
            # Window dinâmico, Step 21 (1m) para performance
            df_rolling_style = engine.calcular_rolling_rbsa(s_fundo_rbsa, df_benchs_rbsa, window=window_days, step=21)
            
            if not df_rolling_style.empty:
                # Filtrar colunas de peso (excluir Alpha e R2)
                cols_weights = [c for c in df_rolling_style.columns if c not in ['Alpha', 'R2']]
                
                # Mover 'Residual_Cash' para o final da lista para ficar no topo da pilha (ou base, depende da pref.)
                # Geralmente, resíduo no topo faz sentido visualmente como "o que sobra"
                if 'Residual_Cash' in cols_weights:
                    cols_weights.remove('Residual_Cash')
                    cols_weights.append('Residual_Cash')
                
                # Paleta extensiva + cor para caixa
                cores_map = {k: v for k, v in zip(cols_weights, paleta_cores)}
                # Definir CINZA para Residual_Cash explicitamente se não tiver sido mapeado ou sobrescrever
                cores_map['Residual_Cash'] = "#D3D3D3" # Cinza Claro
                
                # Plotar Área Empilhada com mapa de cores explicito
                fig_roll = px.area(
                    df_rolling_style, 
                    x=df_rolling_style.index, 
                    y=cols_weights,
                    color_discrete_map=cores_map
                )
                
                fig_roll.update_layout(
                    template="plotly_white",
                    xaxis_title="",
                    yaxis_title="Exposição (Peso)",
                    yaxis_tickformat=".0%", 
                    legend=dict(title=None, orientation="h", y=1.02, x=0.5, xanchor="center"),
                    margin=dict(t=30)
                )
                st.plotly_chart(fig_roll, use_container_width=True)
                
                # Opcional: Mostrar R² histórico em linha secundária ou separado
                with st.expander("Ver Evolução do R² (Qualidade do Modelo)"):
                     fig_r2 = px.line(df_rolling_style, x=df_rolling_style.index, y='R2')
                     fig_r2.update_layout(template="plotly_white", yaxis_title="R²", yaxis_tickformat=".1%", xaxis_title="")
                     st.plotly_chart(fig_r2, use_container_width=True)
            else:
                st.info("Histórico insuficiente para calcular evolução do estilo (Requer > 6 meses de dados).")

            st.divider()
            st.markdown("#### Geração de Alpha: Fundo vs Sintético")
            st.caption("O 'Sintético' é o retorno esperado dado o risco (Betas) assumido. Se o Fundo (Linha Azul) está acima do Sintético (Linha Cinza), houve geração de Alpha.")
            
            # Alinhar retornos do fundo com os retornos da regressão (fitted)
            # fitted tem apenas as datas que entraram no modelo (interseção s_fundo e benchmarks sem NaNs)
            fund_aligned = s_fundo.loc[fitted.index]
            
            # Cálculo Base 100
            wealth_fund = (1 + fund_aligned).cumprod() * 100
            wealth_synth = (1 + fitted).cumprod() * 100
            
            df_alpha_plot = pd.DataFrame({
                'Fundo Real': wealth_fund,
                'Sintético (Replicado)': wealth_synth
            })
            
            # Plot
            fig_alpha = px.line(df_alpha_plot, color_discrete_sequence=[paleta_cores[0], paleta_cores[5]])
            fig_alpha.update_layout(
                template="plotly_white", 
                yaxis_title="Base 100", 
                xaxis_title="", 
                legend=dict(title=None, orientation="h", y=1.05, x=0.5, xanchor="center")
            )
            st.plotly_chart(fig_alpha, use_container_width=True)
                
        else:
            st.error("Erro ao calcular regressão. Verifique se os dados são suficientes.")
    else:
        st.warning("Selecione pelo menos um benchmark para realizar a análise de estilo.")
