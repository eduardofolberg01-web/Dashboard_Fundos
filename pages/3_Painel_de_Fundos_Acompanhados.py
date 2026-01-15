import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Ajuste de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modulos.data_loader import SFODataLoader
from modulos.dashboard_metrics import DashboardMetrics

st.set_page_config(page_title="Painel Geral", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #0d3b66; }
</style>
""", unsafe_allow_html=True)

st.title("Painel de Fundos Acompanhados")

# --- Carregar Dados ---
@st.cache_resource
def load_data():
    loader = SFODataLoader('Quant_Fundos.xlsm')
    return loader

try:
    loader = load_data()
    metrics_engine = DashboardMetrics()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

# Exibir data mais recente dos dados
data_mais_recente = loader.obter_data_mais_recente()
if data_mais_recente:
    st.info(f"**Dados atualizados até:** {data_mais_recente.strftime('%d/%m/%Y')}")

# --- Seleção de Ativos ---
with st.sidebar:
    st.header("Filtros")
    
    # Inicializar session_state para filtros
    if 'painel_classe_selecionada' not in st.session_state:
        st.session_state.painel_classe_selecionada = "Todas"
    if 'painel_subclasse_selecionada' not in st.session_state:
        st.session_state.painel_subclasse_selecionada = "Todas"
    if 'painel_grupo_selecionado' not in st.session_state:
        st.session_state.painel_grupo_selecionado = "Todos"
    
    # Filtros de Classe e Subclasse
    classes_disponiveis = loader.listar_classes('Fundo')
    
    # Classe
    opcoes_classe = ["Todas"] + classes_disponiveis
    try:
        index_classe = opcoes_classe.index(st.session_state.painel_classe_selecionada)
    except ValueError:
        index_classe = 0
        st.session_state.painel_classe_selecionada = "Todas"
    
    classe_selecionada = st.selectbox(
        "Filtrar por Classe",
        options=opcoes_classe,
        index=index_classe,
        key='selectbox_classe_painel'
    )
    st.session_state.painel_classe_selecionada = classe_selecionada
    
    # Subclasse depende da classe selecionada
    if classe_selecionada != "Todas":
        subclasses_disponiveis = loader.listar_subclasses('Fundo', classe_selecionada)
    else:
        subclasses_disponiveis = loader.listar_subclasses('Fundo')
    
    opcoes_subclasse = ["Todas"] + subclasses_disponiveis
    try:
        index_subclasse = opcoes_subclasse.index(st.session_state.painel_subclasse_selecionada)
    except ValueError:
        index_subclasse = 0
        st.session_state.painel_subclasse_selecionada = "Todas"
    
    subclasse_selecionada = st.selectbox(
        "Filtrar por Subclasse",
        options=opcoes_subclasse,
        index=index_subclasse,
        key='selectbox_subclasse_painel'
    )
    st.session_state.painel_subclasse_selecionada = subclasse_selecionada
    
    # Grupo depende de classe e subclasse
    filtro_classe_temp = None if classe_selecionada == "Todas" else classe_selecionada
    filtro_subclasse_temp = None if subclasse_selecionada == "Todas" else subclasse_selecionada
    
    grupos_disponiveis = loader.listar_grupos('Fundo', classe=filtro_classe_temp, subclasse=filtro_subclasse_temp)
    
    opcoes_grupo = ["Todos"] + grupos_disponiveis
    try:
        index_grupo = opcoes_grupo.index(st.session_state.painel_grupo_selecionado)
    except ValueError:
        index_grupo = 0
        st.session_state.painel_grupo_selecionado = "Todos"
    
    grupo_selecionado = st.selectbox(
        "Filtrar por Grupo",
        options=opcoes_grupo,
        index=index_grupo,
        key='selectbox_grupo_painel'
    )
    st.session_state.painel_grupo_selecionado = grupo_selecionado
    
    # Aplicar filtros para listar fundos
    filtro_classe = None if classe_selecionada == "Todas" else classe_selecionada
    filtro_subclasse = None if subclasse_selecionada == "Todas" else subclasse_selecionada
    filtro_grupo = None if grupo_selecionado == "Todos" else grupo_selecionado
    
    todos_fundos = loader.listar_ativos('Fundo', classe=filtro_classe, subclasse=filtro_subclasse, grupo=filtro_grupo)
    
    st.divider()
    
    fundos_selecionados = st.multiselect(
        "Selecionar Fundos (Vazio = Todos)", 
        options=todos_fundos,
        default=[] 
    )
    
    if not fundos_selecionados:
        fundos_processar = todos_fundos
    else:
        fundos_processar = fundos_selecionados

    st.info(f"Processando {len(fundos_processar)} ativos...")

# --- Processamento ---
# Carregar dados de benchmarks necessários (CDI, Ibovespa, IMAB)
benchs_necessarios = ['CDI', 'Ibovespa', 'IMAB', 'PTAX', 'Dolar'] 
# Tenta carregar o que achar
df_benchs = loader.buscar_dados(benchs_necessarios)

# Carregar dados dos fundos
df_fundos = loader.buscar_dados(fundos_processar)

# Preparar Lista de Resultados
rows = []

periods_ret = {
    '12m': 12,
    '24m': 24,
    '36m': 36,
    '48m': 48,
    '60m': 60
}

# Processar cada fundo
for fundo in fundos_processar:
    if fundo not in df_fundos.columns:
        continue
        
    s_fundo = df_fundos[fundo].dropna()
    if s_fundo.empty:
        continue
        
    s_cdi = df_benchs['CDI'].reindex(s_fundo.index).dropna() if 'CDI' in df_benchs else pd.Series()
    s_ibov = df_benchs['Ibovespa'].reindex(s_fundo.index).dropna() if 'Ibovespa' in df_benchs else pd.Series()
    s_imab = df_benchs['IMAB'].reindex(s_fundo.index).dropna() if 'IMAB' in df_benchs else pd.Series()
    
    
    row = {'Fundo': fundo}
    
    # 0. Retorno Mensal (mês mais recente) e % CDI Mensal
    ret_mensal = metrics_engine.calcular_retorno_mensal(s_fundo)
    row['Retorno Mensal'] = ret_mensal
    
    if not s_cdi.empty:
        ret_cdi_mensal = metrics_engine.calcular_retorno_mensal(s_cdi)
        row['%CDI Mensal'] = metrics_engine.calcular_percentual_cdi(ret_mensal, ret_cdi_mensal)
    else:
        row['%CDI Mensal'] = np.nan
    
    # 1. Retorno YTD e % CDI YTD
    ret_ytd = metrics_engine.calcular_ytd(s_fundo)
    row['Retorno YTD'] = ret_ytd
    
    if not s_cdi.empty:
        # Calcular CDI YTD
        ret_cdi_ytd = metrics_engine.calcular_ytd(s_cdi)
        row['%CDI YTD'] = metrics_engine.calcular_percentual_cdi(ret_ytd, ret_cdi_ytd)
    else:
        row['%CDI YTD'] = np.nan
        
    # 2. Retornos Janelas e % CDI
    for label, meses in periods_ret.items():
        ret_per = metrics_engine.calcular_retorno_periodo_custom(s_fundo, meses)
        row[f'Retorno {label}'] = ret_per
        
        if not s_cdi.empty:
            ret_cdi_per = metrics_engine.calcular_retorno_periodo_custom(s_cdi, meses)
            row[f'%CDI {label}'] = metrics_engine.calcular_percentual_cdi(ret_per, ret_cdi_per)
        else:
             row[f'%CDI {label}'] = np.nan
             
    # 3. Volatilidade (12m, 24m, Inception)
    row['Vol 12m'] = metrics_engine.calcular_volatilidade_periodo(s_fundo, 12)
    row['Vol 24m'] = metrics_engine.calcular_volatilidade_periodo(s_fundo, 24)
    row['Vol Início'] = metrics_engine.calcular_volatilidade_periodo(s_fundo, None)
    
    # 4. Drawdown Max e Recuperação
    dd_max, rec_days = metrics_engine.calcular_drawdown_recuperacao(s_fundo)
    row['Max Drawdown'] = dd_max
    row['Recuperação (Dias)'] = rec_days
    
    # 5. Beta IBOV (12m, 24m, 36m)
    if not s_ibov.empty:
        row['Beta IBOV 12m'] = metrics_engine.calcular_beta_periodo(s_fundo, s_ibov, 12)
        row['Beta IBOV 24m'] = metrics_engine.calcular_beta_periodo(s_fundo, s_ibov, 24)
        row['Beta IBOV 36m'] = metrics_engine.calcular_beta_periodo(s_fundo, s_ibov, 36)
    else:
        row['Beta IBOV 12m'] = np.nan
        row['Beta IBOV 24m'] = np.nan
        row['Beta IBOV 36m'] = np.nan

    # 6. Beta IMAB (Removido por solicitação)
    # Mantendo apenas IBOV
    pass
        
    # 7. VaR 95% e Data 1a Cota
    row['VaR 95% (Diário)'] = metrics_engine.calcular_var_95(s_fundo)
    
    # Data de Início: Garantir a primeira data com valor efetivo
    # Filtra zeros que possam existir antes do inicio real e pega a primeira data cronológica
    first_valid_date = s_fundo[s_fundo != 0].first_valid_index()
    if first_valid_date is None:
        first_valid_date = s_fundo.index[0]
        
    row['Início'] = first_valid_date.strftime('%d/%m/%Y')
    
    rows.append(row)

# Criar DataFrame Final
df_painel = pd.DataFrame(rows)

# --- Exibição ---
if not df_painel.empty:
    # Formatação das colunas
    
    # Configuração de colunas para o St.DataFrame
    column_config = {
        "Fundo": st.column_config.TextColumn("Fundo", pinned=True),
        "Início": st.column_config.TextColumn("Início"),
        "Recuperação (Dias)": st.column_config.NumberColumn("Recup. DD (Dias)", format="%d"),
    }
    
    # Colunas de Percentual (Retorno, Vol, VaR, DD)
    cols_pct = [c for c in df_painel.columns if any(x in c for x in ['Retorno', 'Vol', 'VaR', 'Drawdown'])]
    for c in cols_pct:
        column_config[c] = st.column_config.NumberColumn(c, format="%.1f%%")
        
    # Colunas de % do CDI (1 casa)
    cols_cdi = [c for c in df_painel.columns if '%CDI' in c]
    for c in cols_cdi:
        column_config[c] = st.column_config.NumberColumn(c, format="%.1f%%")
        
    # Colunas de Beta (Decimal 1 casa)
    cols_beta = [c for c in df_painel.columns if 'Beta' in c]
    for c in cols_beta:
        column_config[c] = st.column_config.NumberColumn(c, format="%.1f")

    # Mover 'Fundo' para primeira coluna, seguido de Início e métricas mensais
    cols = ['Fundo', 'Início', 'Retorno Mensal', '%CDI Mensal', 'Retorno YTD', '%CDI YTD']
    # Adicionar as outras na ordem
    rest_cols = [c for c in df_painel.columns if c not in cols]
    # Tentar organizar logicamente
    # 1. Retornos (excluindo Mensal e YTD que já estão em cols)
    r_cols = [c for c in rest_cols if 'Retorno ' in c]
    # 2. %CDI (excluindo Mensal e YTD que já estão em cols)
    cdi_cols = [c for c in rest_cols if '%CDI ' in c]
    # 3. Vol
    v_cols = [c for c in rest_cols if 'Vol ' in c]
    # 4. Risco
    risk_cols = ['Max Drawdown', 'Recuperação (Dias)', 'VaR 95% (Diário)']
    # 5. Betas
    b_cols = [c for c in rest_cols if 'Beta ' in c]
    
    final_order = cols + r_cols + cdi_cols + v_cols + risk_cols + b_cols
    
    # Garantir que todas as colunas existem (algumas podem ter falhado se rest_cols tiver algo diferente)
    final_order = [c for c in final_order if c in df_painel.columns]
    
    df_show = df_painel[final_order]
    
    # Aplicar estilo condicional para colunas %CDI
    def color_cdi_threshold(val):
        if pd.isna(val):
            return None
        # Verde se >= 100% (1.0), Vermelho se < 100%
        # Usando cores do tema (Verde Esmeralda e Vermelho Carmim)
        color = '#2ca02c' if val >= 100.0 else '#d62728'
        return f'color: {color}; font-weight: bold'

    cols_cdi_style = [c for c in df_show.columns if '%CDI' in c]
    
    # Criar Styler (usando map ou applymap conforme versão)
    # Tenta usar map (Pandas novo), fallback para applymap se der erro é implicito se usarmos nomes padrão
    # Mas aqui vou usar map pois estamos em ambiente dev recente
    styler = df_show.style.map(color_cdi_threshold, subset=cols_cdi_style)
    
    st.dataframe(
        styler, 
        column_config=column_config,
        height=600,
        use_container_width=True,
        hide_index=True
    )
    
    # Botão Download
    csv = df_show.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Baixar Tabela CSV",
        csv,
        "painel_geral_fundos.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.warning("Nenhum dado encontrado para os fundos selecionados.")
