import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Ajuste de Path para encontrar os módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modulos.data_loader import SFODataLoader
from modulos.optimizer import PortfolioOptimizer

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Laboratório de Otimização", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS Minimalista e Corporativo (Consistente com app principal)
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

        /* Expander limpo */
        div[data-testid="stExpander"] {
            background-color: #ffffff;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            color: #212529;
        }

        /* Ajuste de fontes gerais */
        html, body, [class*="css"] {
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            color: #212529;
        }
        
        /* Títulos */
        h1, h2, h3 {
            color: #0d3b66;
            font-weight: 700;
        }
        
        div.stButton > button {
            width: 100%;
            border-radius: 4px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Laboratório de Otimização de Portfólio")
st.caption("Simulação de Fronteira Eficiente e Alocação de Ativos (Markowitz Mean-Variance)")

# --- 2. BARRA LATERAL (PARÂMETROS) ---
with st.sidebar:
    st.header("1. Seleção de Ativos")
    
    try:
        # Carrega path relativo à raiz
        file_path = os.path.join(root_dir, 'Quant_Fundos.xlsm')
        loader = SFODataLoader(file_path)
    except Exception as e:
        st.error(f"Erro ao carregar base de dados: {e}")
        st.stop()

# Exibir data mais recente dos dados (fora da sidebar para ficar no topo)
data_mais_recente = loader.obter_data_mais_recente()
if data_mais_recente:
    st.info(f"**Dados atualizados até:** {data_mais_recente.strftime('%d/%m/%Y')}")

with st.sidebar:
    
    # Inicializar session_state para filtros
    if 'lab_filtro_classe' not in st.session_state:
        st.session_state.lab_filtro_classe = "Todas"
    if 'lab_filtro_subclasse' not in st.session_state:
        st.session_state.lab_filtro_subclasse = "Todas"
    if 'lab_filtro_grupo' not in st.session_state:
        st.session_state.lab_filtro_grupo = "Todos"
    if 'lab_janela' not in st.session_state:
        st.session_state.lab_janela = "24 Meses"
    
    # Filtros de Ativos
    with st.expander("Filtros de Ativos"):
        c_p1, c_p2 = st.columns(2)
        
        # Filtros
        classes_disp = ["Todas"] + loader.listar_classes() # Todas as classes (Fundos e Benchs)
        
        try:
            index_classe = classes_disp.index(st.session_state.lab_filtro_classe)
        except ValueError:
            index_classe = 0
            st.session_state.lab_filtro_classe = "Todas"
        
        filtro_classe = st.selectbox(
            "Classe:", 
            options=classes_disp, 
            index=index_classe,
            key='selectbox_classe_lab'
        )
        st.session_state.lab_filtro_classe = filtro_classe
        
        classe_search = filtro_classe if filtro_classe != "Todas" else None
        subclasses_disp = ["Todas"] + loader.listar_subclasses(classe=classe_search)
        
        try:
            index_subclasse = subclasses_disp.index(st.session_state.lab_filtro_subclasse)
        except ValueError:
            index_subclasse = 0
            st.session_state.lab_filtro_subclasse = "Todas"
        
        filtro_subclasse = st.selectbox(
            "Subclasse:", 
            options=subclasses_disp, 
            index=index_subclasse,
            key='selectbox_subclasse_lab'
        )
        st.session_state.lab_filtro_subclasse = filtro_subclasse
        
        sub_search = filtro_subclasse if filtro_subclasse != "Todas" else None
        grupos_disp = ["Todos"] + loader.listar_grupos(classe=classe_search, subclasse=sub_search)
        
        try:
            index_grupo = grupos_disp.index(st.session_state.lab_filtro_grupo)
        except ValueError:
            index_grupo = 0
            st.session_state.lab_filtro_grupo = "Todos"
        
        filtro_grupo = st.selectbox(
            "Grupo:", 
            options=grupos_disp, 
            index=index_grupo,
            key='selectbox_grupo_lab'
        )
        st.session_state.lab_filtro_grupo = filtro_grupo
        
        grupo_search = filtro_grupo if filtro_grupo != "Todos" else None

    # Listar ativos filtrados (Fundos + Benchmarks)
    ativos_disp = loader.listar_ativos(classe=classe_search, subclasse=sub_search, grupo=grupo_search)
    
    # Se lista vazia, fallback para evitar erro
    if not ativos_disp:
        st.warning("Nenhum ativo encontrado com os filtros selecionados.")
    
    # Lógica de seleção padrão:
    # Se algum filtro estiver ativo (não "Todas"/"Todos"), selecionar TODOS os ativos filtrados por padrão.
    # Se estiver vendo TUDO, limitar a 4 para não travar a interface.
    filtros_ativos = (classe_search is not None) or (sub_search is not None) or (grupo_search is not None)
    
    default_selection = ativos_disp if filtros_ativos else (ativos_disp[:4] if len(ativos_disp) > 3 else ativos_disp)
    
    ativos_sel = st.multiselect(
        "Carteira para Otimizar", 
        options=ativos_disp, 
        default=default_selection
    )
    
    opcoes_janela = ["6 Meses", "12 Meses", "24 Meses", "36 Meses", "Desde o Início"]
    try:
        index_janela = opcoes_janela.index(st.session_state.lab_janela)
    except ValueError:
        index_janela = 2
        st.session_state.lab_janela = "24 Meses"
    
    janela = st.selectbox(
        "Histórico para Calibração", 
        opcoes_janela,
        index=index_janela,
        key='selectbox_janela_lab'
    )
    st.session_state.lab_janela = janela
    
    # --- Risk Free (CDI) ---
    # Busca dados do CDI para usar como taxa livre de risco
    try:
        df_cdi = loader.buscar_dados(['CDI'])
        # Ajusta para o período selecionado (aproximado)
        window_map_cdi = {
            "6 Meses": 126,
            "12 Meses": 252,
            "24 Meses": 504,
            "36 Meses": 756
        }
        
        if janela in window_map_cdi:
            days_cdi = window_map_cdi[janela]
        else:
            days_cdi = len(df_cdi)
            
        df_cdi_slice = df_cdi.tail(days_cdi)['CDI'].dropna()
        
        # Retorno Total do CDI no período
        ret_total_cdi = (1 + df_cdi_slice).prod() - 1
        
        # Anualiza para a taxa usada no Sharpe
        # Se o período for < 1 ano, projeta. Se > 1 ano, anualiza a média geométrica.
        n_dias_uteis = len(df_cdi_slice)
        if n_dias_uteis > 0:
            rf_rate_calc = (1 + ret_total_cdi) ** (252 / n_dias_uteis) - 1
        else:
            rf_rate_calc = 0.10 # Fallback 10%
            
    except Exception as e:
        st.warning(f"Não foi possível carregar CDI ({e}). Usando 10% padrão.")
        rf_rate_calc = 0.10

    st.metric("Taxa Livre de Risco (CDI Anualizado)", f"{rf_rate_calc:.2%}", help="Calculado com base no retorno histórico do CDI no período selecionado.")
    rf_rate = rf_rate_calc
    
    st.divider()
    
    st.header("2. Restrições")
    with st.expander("Limites de Alocação (Mínimo e Máximo)"):
        st.caption("Defina o peso MÍNIMO e MÁXIMO para cada ativo (0-100%).")
        
        # --- Definição em Massa ---
        st.markdown("##### Definições em Massa")
        col_mass_1, col_mass_2, col_mass_3 = st.columns([1, 1, 1])
        with col_mass_1:
            global_min = st.number_input("Mínimo Global (%)", min_value=0.0, max_value=100.0, value=0.0, step=5.0)
        with col_mass_2:
            global_max = st.number_input("Máximo Global (%)", min_value=0.0, max_value=100.0, value=100.0, step=5.0)
        with col_mass_3:
            st.write("") # Espaçamento
            st.write("") 
            if st.button("Aplicar a Todos"):
                for ativo in ativos_sel:
                    st.session_state[f"min_{ativo}"] = global_min
                    st.session_state[f"max_{ativo}"] = global_max
                st.rerun() # Recarrega para atualizar os inputs individuais
        
        st.divider()
        st.markdown("##### Definições Individuais")
        
        min_weights = {}
        max_weights = {}
        
        for ativo in ativos_sel:
            c_min, c_max = st.columns(2)
            with c_min:
                val_min = st.number_input(
                    f"Min % {ativo}", 
                    min_value=0.0, max_value=100.0, value=0.0, step=5.0, 
                    key=f"min_{ativo}"
                )
                min_weights[ativo] = val_min / 100.0
            with c_max:
                val_max = st.number_input(
                    f"Max % {ativo}", 
                    min_value=0.0, max_value=100.0, value=100.0, step=5.0, 
                    key=f"max_{ativo}"
                )
                max_weights[ativo] = val_max / 100.0

    # Validações
    soma_minima = sum(min_weights.values())
    if soma_minima > 1.0:
        st.error(f"Erro: A soma dos mínimos ({soma_minima:.0%}) excede 100%.")
        st.stop()
        
    for ativo in ativos_sel:
        if min_weights[ativo] > max_weights[ativo]:
            st.error(f"Erro em {ativo}: Mínimo ({min_weights[ativo]:.0%}) maior que Máximo ({max_weights[ativo]:.0%}).")
            st.stop()
        
    st.divider()
    
    st.header("3. Objetivo")
    definir_target = st.checkbox("Definir Volatilidade Máxima (Alvo)?", value=False)
    
    target_vol = None
    if definir_target:
        target_vol = st.slider(
            "Volatilidade Alvo (%)", 
            min_value=0.5, max_value=40.0, value=10.0, step=0.5, 
            format="%.1f%%"
        ) / 100.0
    
    rodar = st.button("Executar Otimização", type="primary")

# --- 3. PROCESSAMENTO ---
# --- 3. PROCESSAMENTO & STATE MANAGEMENT ---
if rodar:
    st.session_state['run_optimization'] = True
    # Limpa resultados anteriores se houver
    if 'opt_results' in st.session_state:
        del st.session_state['opt_results']

if not st.session_state.get('run_optimization', False):
    st.info("Configure os ativos e restrições na barra lateral e clique em 'Executar Otimização' para iniciar.")
    st.stop()

# Recupera dados APENAS se não estiverem cacheados ou se 'rodar' foi clicado
if rodar or 'opt_results' not in st.session_state:

    if len(ativos_sel) < 2:
        st.error("Selecione pelo menos 2 ativos para compor um portfólio.")
        st.stop()
    
    # A. Dados
    df_raw = loader.buscar_dados(ativos_sel)
    if df_raw.empty:
        st.error("Dados insuficientes.")
        st.stop()
    
    # ... Processamento de Dados ...
    
    # DIAGNÓSTICO: Verificar quem está limitando o histórico
    first_valid_indices = df_raw.apply(lambda col: col.first_valid_index())
    # O início comum é a MAIOR data de início dentre os ativos (gargalo)
    if not first_valid_indices.dropna().empty:
        max_start_date = first_valid_indices.max()
        ativo_limitante = first_valid_indices.idxmax()
        
        # Se a data limite for muito recente (ex: menos de 95% da janela solicitada ou simplesmente informativa)
        # Vamos sempre avisar se for "recente" no contexto do ano
        
        # Formatação para o usuário
        st.info(f"ℹ️ **Informação sobre o Período**: O histórico comum dos ativos selecionados começa em **{max_start_date.strftime('%d/%m/%Y')}** (limitado pelo ativo **{ativo_limitante}**). A otimização utiliza apenas o período em que todos os ativos possuem dados (interseção).")

    df_ret = df_raw.dropna().copy()
    if df_ret.abs().mean().mean() > 0.1:
        st.warning("⚠️ Detectado dados em percentual inteiro (ex: 1.0). Convertendo para decimal (ex: 0.01).")
        df_ret = df_ret / 100.0
        
    # B. Corte Temporal
    window_map_ret = {
        "6 Meses": 126,
        "12 Meses": 252,
        "24 Meses": 504,
        "36 Meses": 756
    }
    
    if janela in window_map_ret:
        dias = window_map_ret[janela]
    else:
        dias = len(df_ret)
        
    df_ret = df_ret.tail(dias)
    
    # C. Bounds e Optimizer
    bounds = tuple((min_weights[a], max_weights[a]) for a in ativos_sel)
    
    try:
        opt = PortfolioOptimizer(df_ret, risk_free_rate=rf_rate)
    except Exception as e:
        st.error(f"Erro na inicialização do módulo de otimização: {e}")
        st.stop()
        
    # Executa Otimização
    results_cache = {
        'df_ret': df_ret,
        'bounds': bounds
    }
    
    with st.spinner("Otimizando Portfólios..."):
        if definir_target:
            # === MODO 1: ALVO FIXO ===
            min_vol_result = opt.otimizar_min_vol(bounds)
            if min_vol_result:
                piso_vol = min_vol_result['Volatility']
                if target_vol < piso_vol:
                     st.warning(f"Volatilidade solicitada ({target_vol:.1%}) < Mínimo ({piso_vol:.1%}). Usando Min Vol.")
                     result = min_vol_result
                else:
                    result = opt.otimizar_vol_alvo(target_vol, bounds)
            else:
                result = None
            
            results_cache['mode'] = 'target'
            results_cache['result'] = result
            
        else:
            # === MODO 2: FRONTEIRA ===
            portfolios = opt.gerar_fronteira_pontos(n_points=10, bounds=bounds)
            max_sharpe_port = opt.otimizar_max_sharpe(bounds)
            
            results_cache['mode'] = 'frontier'
            results_cache['portfolios'] = portfolios
            results_cache['max_sharpe'] = max_sharpe_port
            
    # Salva no Session State
    st.session_state['opt_results'] = results_cache

# --- 4. EXIBIÇÃO DOS RESULTADOS (DADOS DO CACHE) ---
if 'opt_results' in st.session_state:
    cache = st.session_state['opt_results']
    df_ret_cached = cache['df_ret']
    
    if janela == "Desde o Início":
         st.info(f"📅 Data Inicial dos Dados: {df_ret_cached.index[0].strftime('%d/%m/%Y')}")

    st.divider()
    
    if cache['mode'] == 'target':
        result = cache['result']
        if result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Retorno Esperado (a.a.)", f"{result['Return']:.2%}")
            c2.metric("Volatilidade (a.a.)", f"{result['Volatility']:.2%}")
            c3.metric("Sharpe Ratio", f"{result['Sharpe']:.2f}")
            
            col_pie, col_line = st.columns([1, 2])
            with col_pie:
                df_w = pd.DataFrame(list(result['Weights'].items()), columns=['Ativo', 'Peso'])
                df_plot = df_w[df_w['Peso'] > 0.001].copy()
                fig = px.pie(df_plot, values='Peso', names='Ativo', title="Alocação Sugerida", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
                
            with col_line:
                st.markdown("#### Backtest Histórico (In-Sample)")
                weights_series = pd.Series(result['Weights'])
                retorno_diario_port = (df_ret_cached[weights_series.index] * weights_series).sum(axis=1)
                cum_ret = (1 + retorno_diario_port).cumprod() * 100
                cum_ret = cum_ret / cum_ret.iloc[0] * 100
                fig_line = px.line(cum_ret, title="Evolução Patrimonial (Base 100)")
                fig_line.update_traces(line_color='#8B9DC3')
                fig_line.update_layout(template="plotly_white", yaxis_title="Base 100", xaxis_title="")
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.error("O otimizador não convergiu.")
            
    elif cache['mode'] == 'frontier':
        portfolios = cache['portfolios']
        # max_sharpe_port = cache['max_sharpe']
        
        if portfolios:
            sharpe_vals = [p['Sharpe'] for p in portfolios]
            idx_max_sharpe_fronteira = sharpe_vals.index(max(sharpe_vals))
            indices_show = list(range(len(portfolios)))
            
            resumo_data = []
            df_backtest = pd.DataFrame(index=df_ret_cached.index)
            
            for idx in indices_show:
                if idx >= len(portfolios): continue
                p = portfolios[idx]
                
                if idx == 0: label = "Min Vol"
                elif idx == idx_max_sharpe_fronteira: label = "Max Sharpe"
                else: label = f"Portfolio {idx + 1}"
                
                row = {
                    "Perfil": label,
                    "Retorno (a.a)": f"{p['Return']:.2%}",
                    "Volatilidade": f"{p['Volatility']:.2%}",
                    "Sharpe": f"{p['Sharpe']:.2f}",
                }
                for ativo, peso in p['Weights'].items():
                    row[ativo] = f"{peso:.1%}"
                resumo_data.append(row)
                
                w_s = pd.Series(p['Weights'])
                ret_daily = (df_ret_cached[w_s.index] * w_s).sum(axis=1)
                cum_series = (1 + ret_daily).cumprod() * 100
                cum_series = cum_series / cum_series.iloc[0] * 100
                df_backtest[label] = cum_series
                
            st.dataframe(pd.DataFrame(resumo_data), use_container_width=True, hide_index=True)
            
            col_graf, col_weights = st.columns([2, 1])
            with col_graf:
                x_vals = [p['Volatility'] * 100 for p in portfolios]
                y_vals = [p['Return'] * 100 for p in portfolios]
                fig_front = go.Figure()
                fig_front.add_trace(go.Scatter(
                    x=x_vals, y=y_vals, mode='lines+markers', 
                    line=dict(color='#8B9DC3', width=2),
                    marker=dict(size=8, color=y_vals, colorscale='Viridis', showscale=True, colorbar=dict(title="Retorno (%)", ticksuffix="%")),
                    name='Fronteira',
                    hovertemplate='<b>Volatilidade:</b> %{x:.2f}%<br><b>Retorno:</b> %{y:.2f}%<extra></extra>'
                ))
                fig_front.update_layout(title="Risco vs Retorno", xaxis_title="Volatilidade (%)", yaxis_title="Retorno (%)", template="plotly_white", height=450)
                st.plotly_chart(fig_front, use_container_width=True)
                
            with col_weights:
                data_stack = []
                for idx in indices_show:
                    if idx >= len(portfolios): continue
                    p = portfolios[idx]
                    if idx == 0: label = "Min Vol"
                    elif idx == idx_max_sharpe_fronteira: label = "Max Sharpe"
                    else: label = f"Portfolio {idx + 1}"
                    for ativo, peso in p['Weights'].items():
                        data_stack.append({'Cenário': label, 'Ativo': ativo, 'Peso': peso})
                
                fig_stack = px.bar(pd.DataFrame(data_stack), x='Cenário', y='Peso', color='Ativo', title="Alocação")
                fig_stack.update_layout(template="plotly_white", height=450)
                st.plotly_chart(fig_stack, use_container_width=True)
            
            # --- Backtest Gráfico com Filtro ---
            st.markdown("#### Evolução Histórica dos Cenários (Base 100)")
            
            sel_scenarios = st.multiselect(
                "Filtrar Cenários:",
                options=df_backtest.columns,
                default=df_backtest.columns
            )
            
            if sel_scenarios:
                fig_evol = px.line(df_backtest[sel_scenarios], x=df_backtest.index, y=sel_scenarios)
                fig_evol.update_layout(template="plotly_white", xaxis_title="", yaxis_title="Base 100", legend_title=None)
                st.plotly_chart(fig_evol, use_container_width=True)
            else:
                st.warning("Selecione pelo menos um cenário.")
        else:
            st.error("Não foi possível gerar a fronteira.")