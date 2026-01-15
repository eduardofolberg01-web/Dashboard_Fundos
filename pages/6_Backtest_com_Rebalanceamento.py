import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Ajuste de Path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modulos.data_loader import SFODataLoader
from modulos.portfolio_backtest import PortfolioBacktester

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================

st.set_page_config(
    page_title="Backtest com Rebalanceamento",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        div[data-testid="stMetric"] {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 10px;
            border-radius: 5px;
        }
        
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

st.title("Backtest de Portfólio com Rebalanceamento")
st.caption("Simulação histórica com múltiplas estratégias de otimização e drift realista")

# ==========================================
# CARREGAMENTO DE DADOS
# ==========================================

try:
    file_path = os.path.join(root_dir, 'Quant_Fundos.xlsm')
    loader = SFODataLoader(file_path)
except Exception as e:
    st.error(f"Erro ao carregar base de dados: {e}")
    st.stop()

data_mais_recente = loader.obter_data_mais_recente()
if data_mais_recente:
    st.info(f"**Dados atualizados até:** {data_mais_recente.strftime('%d/%m/%Y')}")

# ==========================================
# SIDEBAR - PARÂMETROS
# ==========================================

with st.sidebar:
    st.header("Configurações do Backtest")
    
    # --- Seleção de Ativos ---
    st.subheader("1. Seleção de Ativos")
    
    # Inicializar session_state para filtros
    if 'bt_filtro_classe' not in st.session_state:
        st.session_state.bt_filtro_classe = "Todas"
    if 'bt_filtro_subclasse' not in st.session_state:
        st.session_state.bt_filtro_subclasse = "Todas"
    if 'bt_filtro_grupo' not in st.session_state:
        st.session_state.bt_filtro_grupo = "Todos"
    
    with st.expander("Filtros de Ativos"):
        classes_disp = ["Todas"] + loader.listar_classes()
        
        try:
            index_classe = classes_disp.index(st.session_state.bt_filtro_classe)
        except ValueError:
            index_classe = 0
        
        filtro_classe = st.selectbox(
            "Classe:", 
            options=classes_disp, 
            index=index_classe,
            key='selectbox_classe_bt'
        )
        st.session_state.bt_filtro_classe = filtro_classe
        
        classe_search = filtro_classe if filtro_classe != "Todas" else None
        subclasses_disp = ["Todas"] + loader.listar_subclasses(classe=classe_search)
        
        try:
            index_subclasse = subclasses_disp.index(st.session_state.bt_filtro_subclasse)
        except ValueError:
            index_subclasse = 0
        
        filtro_subclasse = st.selectbox(
            "Subclasse:", 
            options=subclasses_disp, 
            index=index_subclasse,
            key='selectbox_subclasse_bt'
        )
        st.session_state.bt_filtro_subclasse = filtro_subclasse
        
        sub_search = filtro_subclasse if filtro_subclasse != "Todas" else None
        grupos_disp = ["Todos"] + loader.listar_grupos(classe=classe_search, subclasse=sub_search)
        
        try:
            index_grupo = grupos_disp.index(st.session_state.bt_filtro_grupo)
        except ValueError:
            index_grupo = 0
        
        filtro_grupo = st.selectbox(
            "Grupo:", 
            options=grupos_disp, 
            index=index_grupo,
            key='selectbox_grupo_bt'
        )
        st.session_state.bt_filtro_grupo = filtro_grupo
        
        grupo_search = filtro_grupo if filtro_grupo != "Todos" else None
    
    ativos_disp = loader.listar_ativos(classe=classe_search, subclasse=sub_search, grupo=grupo_search)
    
    # Remover CDI das opções de ativos (CDI é usado apenas como taxa livre de risco e benchmark)
    ativos_disp = [a for a in ativos_disp if a != 'CDI']
    
    if not ativos_disp:
        st.warning("Nenhum ativo encontrado com os filtros selecionados.")
    
    # Inicializar session_state para ativos selecionados
    if 'bt_ativos_sel' not in st.session_state:
        filtros_ativos = (classe_search is not None) or (sub_search is not None) or (grupo_search is not None)
        default_selection = ativos_disp if filtros_ativos else (ativos_disp[:4] if len(ativos_disp) > 3 else ativos_disp)
        st.session_state.bt_ativos_sel = default_selection
    
    # Garantir que os ativos salvos ainda estão disponíveis
    ativos_validos = [a for a in st.session_state.bt_ativos_sel if a in ativos_disp]
    if not ativos_validos and ativos_disp:
        # Se nenhum ativo salvo está disponível, usar default
        filtros_ativos = (classe_search is not None) or (sub_search is not None) or (grupo_search is not None)
        ativos_validos = ativos_disp if filtros_ativos else (ativos_disp[:4] if len(ativos_disp) > 3 else ativos_disp)
    
    ativos_sel = st.multiselect(
        "Ativos para Backtest:", 
        options=ativos_disp, 
        default=ativos_validos,
        key='multiselect_ativos_bt'
    )
    
    # Atualizar session_state
    st.session_state.bt_ativos_sel = ativos_sel
    
    st.divider()
    
    # --- Estratégia de Otimização ---
    st.subheader("2. Estratégia de Otimização")
    
    strategy_options = {
        'max_sharpe': 'Máximo Sharpe Ratio',
        'min_variance': 'Mínima Variância',
        'max_return_fixed_risk': 'Máximo Retorno (Risco Fixo)',
        'risk_parity': 'Paridade de Risco',
        'equal_weight': 'Pesos Iguais (Benchmark)'
    }
    
    # Inicializar session_state para estratégia
    if 'bt_strategy' not in st.session_state:
        st.session_state.bt_strategy = 'max_sharpe'
    
    try:
        index_strategy = list(strategy_options.keys()).index(st.session_state.bt_strategy)
    except ValueError:
        index_strategy = 0
    
    strategy = st.selectbox(
        "Selecione a Estratégia:",
        options=list(strategy_options.keys()),
        format_func=lambda x: strategy_options[x],
        index=index_strategy,
        key='selectbox_strategy_bt'
    )
    
    # Atualizar session_state
    st.session_state.bt_strategy = strategy
    
    # Descrições das estratégias
    strategy_descriptions = {
        'max_sharpe': "Maximiza o Sharpe Ratio (retorno ajustado ao risco). Ideal para buscar a melhor relação risco-retorno.",
        'min_variance': "Minimiza a volatilidade do portfólio. Ideal para investidores conservadores.",
        'max_return_fixed_risk': "Maximiza o retorno sujeito a uma volatilidade máxima. Permite controlar o risco.",
        'risk_parity': "Equaliza a contribuição de risco de cada ativo. Ativos mais voláteis recebem menos peso.",
        'equal_weight': "Distribui igualmente entre os ativos (1/N). Serve como benchmark simples."
    }
    
    st.info(strategy_descriptions[strategy])
    
    # Volatilidade alvo (apenas para max_return_fixed_risk)
    target_vol = None
    if strategy == 'max_return_fixed_risk':
        # Inicializar session_state para target_vol
        if 'bt_target_vol' not in st.session_state:
            st.session_state.bt_target_vol = 15.0
        
        target_vol_pct = st.slider(
            "Volatilidade Alvo (% a.a.):",
            min_value=5.0, max_value=40.0, value=st.session_state.bt_target_vol, step=1.0,
            key='slider_target_vol_bt'
        )
        
        # Atualizar session_state
        st.session_state.bt_target_vol = target_vol_pct
        target_vol = target_vol_pct / 100.0
    
    st.divider()
    
    # --- Parâmetros de Rebalanceamento ---
    st.subheader("3. Parâmetros de Rebalanceamento")
    
    # Inicializar session_state para lookback_days
    if 'bt_lookback_days' not in st.session_state:
        st.session_state.bt_lookback_days = 252
    
    try:
        index_lookback = [126, 252, 504, 756].index(st.session_state.bt_lookback_days)
    except ValueError:
        index_lookback = 1
    
    lookback_days = st.selectbox(
        "Janela de Calibração:",
        options=[126, 252, 504, 756],
        format_func=lambda x: f"{x} dias úteis (~{x//21} meses)",
        index=index_lookback,
        key='selectbox_lookback_bt'
    )
    
    # Atualizar session_state
    st.session_state.bt_lookback_days = lookback_days
    
    # Inicializar session_state para rebalance_freq
    if 'bt_rebalance_freq' not in st.session_state:
        st.session_state.bt_rebalance_freq = 'Trimestral'
    
    try:
        index_rebal = ['Mensal', 'Trimestral', 'Semestral', 'Anual'].index(st.session_state.bt_rebalance_freq)
    except ValueError:
        index_rebal = 1
    
    rebalance_freq = st.selectbox(
        "Frequência de Rebalanceamento:",
        options=['Mensal', 'Trimestral', 'Semestral', 'Anual'],
        index=index_rebal,
        key='selectbox_rebalance_bt'
    )
    
    # Atualizar session_state
    st.session_state.bt_rebalance_freq = rebalance_freq
    
    st.divider()
    
    # --- Benchmarks ---
    st.subheader("4. Benchmarks de Comparação")
    
    st.info("O portfólio Equal-Weight (pesos iguais) será sempre calculado como benchmark automático.")
    
    # Listar todos os ativos disponíveis (fundos + benchmarks)
    # INCLUINDO os ativos já selecionados para o portfólio
    todos_ativos = loader.listar_ativos()
    
    # Inicializar session_state para benchmarks
    if 'bt_benchmarks' not in st.session_state:
        st.session_state.bt_benchmarks = []
    
    # Garantir que os benchmarks salvos ainda estão disponíveis
    benchmarks_validos = [b for b in st.session_state.bt_benchmarks if b in todos_ativos]
    
    benchmarks_adicionais = st.multiselect(
        "Benchmarks/Fundos Adicionais (Opcional):",
        options=todos_ativos,
        default=benchmarks_validos,
        help="Selecione um ou mais ativos para comparação (benchmarks, fundos, etc.). Pode incluir os mesmos ativos do portfólio para comparar estratégia vs. desempenho individual.",
        key='multiselect_benchmarks_bt'
    )
    
    # Atualizar session_state
    st.session_state.bt_benchmarks = benchmarks_adicionais
    
    st.divider()
    
    # --- Restrições de Peso ---
    st.subheader("5. Restrições de Peso")
    
    with st.expander("Limites de Alocação"):
        st.caption("Defina o peso MÍNIMO e MÁXIMO para cada ativo (0-100%).")
        
        # Inicializar valores no session_state
        for ativo in ativos_sel:
            if f"bt_min_{ativo}" not in st.session_state:
                st.session_state[f"bt_min_{ativo}"] = 0.0
            if f"bt_max_{ativo}" not in st.session_state:
                st.session_state[f"bt_max_{ativo}"] = 100.0
        
        # Definição em massa
        st.markdown("##### Definições em Massa")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            global_min = st.number_input("Mínimo Global (%)", 0.0, 100.0, 0.0, 5.0)
        with col2:
            global_max = st.number_input("Máximo Global (%)", 0.0, 100.0, 100.0, 5.0)
        with col3:
            st.write("")
            st.write("")
            if st.button("Aplicar a Todos"):
                for ativo in ativos_sel:
                    st.session_state[f"bt_min_{ativo}"] = global_min
                    st.session_state[f"bt_max_{ativo}"] = global_max
                st.rerun()
        
        st.divider()
        st.markdown("##### Definições Individuais")
        
        min_weights = {}
        max_weights = {}
        
        for ativo in ativos_sel:
            c_min, c_max = st.columns(2)
            with c_min:
                val_min = st.number_input(
                    f"Min % {ativo}", 
                    min_value=0.0, max_value=100.0, step=5.0, 
                    key=f"bt_min_{ativo}"
                )
                min_weights[ativo] = val_min / 100.0
            with c_max:
                val_max = st.number_input(
                    f"Max % {ativo}", 
                    min_value=0.0, max_value=100.0, step=5.0, 
                    key=f"bt_max_{ativo}"
                )
                max_weights[ativo] = val_max / 100.0
    
    # Validações
    soma_minima = sum(min_weights.values())
    if soma_minima > 1.0:
        st.error(f"Erro: A soma dos mínimos ({soma_minima:.0%}) excede 100%.")
        st.stop()
    
    for ativo in ativos_sel:
        if min_weights[ativo] > max_weights[ativo]:
            st.error(f"Erro em {ativo}: Mínimo maior que Máximo.")
            st.stop()
    
    st.divider()
    
    rodar = st.button("Executar Backtest", type="primary")

# ==========================================
# PROCESSAMENTO
# ==========================================

if rodar:
    st.session_state['run_backtest'] = True

if not st.session_state.get('run_backtest', False):
    st.info("Configure os parâmetros na barra lateral e clique em '🚀 Executar Backtest' para iniciar.")
    st.stop()

if rodar or 'backtest_results' not in st.session_state:
    
    if len(ativos_sel) < 2:
        st.error("Selecione pelo menos 2 ativos para o backtest.")
        st.stop()
    
    with st.spinner("Carregando dados..."):
        # Buscar dados dos ativos, CDI e benchmarks adicionais (se houver)
        # Usar set para evitar duplicação se um ativo estiver no portfólio E nos benchmarks
        ativos_para_buscar = list(set(ativos_sel + ['CDI'] + (benchmarks_adicionais if benchmarks_adicionais else [])))
        
        df_raw = loader.buscar_dados(ativos_para_buscar)
        
        if df_raw.empty:
            st.error("Dados insuficientes.")
            st.stop()
        
        # Separar CDI e benchmarks
        df_cdi = df_raw['CDI'].copy()
        df_returns = df_raw[ativos_sel].copy()
        
        df_benchmarks_adicionais = {}
        if benchmarks_adicionais:
            for bench in benchmarks_adicionais:
                df_benchmarks_adicionais[bench] = df_raw[bench].copy()
        
        # Remove NaN
        if benchmarks_adicionais:
            df_combined = pd.concat([df_returns, df_cdi] + list(df_benchmarks_adicionais.values()), axis=1).dropna()
            for bench in benchmarks_adicionais:
                df_benchmarks_adicionais[bench] = df_combined[bench]
        else:
            df_combined = pd.concat([df_returns, df_cdi], axis=1).dropna()
        
        df_returns = df_combined[ativos_sel]
        df_cdi = df_combined['CDI']
        
        if df_returns.empty:
            st.error("Não há período comum com dados para todos os ativos.")
            st.stop()
        
        # Ajustar período de análise baseado nos benchmarks
        # Se algum benchmark tiver histórico mais curto, limitar o período
        data_inicio_analise = df_returns.index[0]
        
        if benchmarks_adicionais:
            # Encontrar a data mais recente de início entre todos os benchmarks
            datas_inicio_benchmarks = []
            for bench_name, bench_data in df_benchmarks_adicionais.items():
                primeira_data = bench_data.first_valid_index()
                if primeira_data:
                    datas_inicio_benchmarks.append(primeira_data)
            
            if datas_inicio_benchmarks:
                data_mais_recente = max(datas_inicio_benchmarks)
                
                # Se algum benchmark começar depois do portfólio, ajustar
                if data_mais_recente > data_inicio_analise:
                    st.warning(
                        f"Ajustando período de análise: Benchmark mais recente inicia em {data_mais_recente.strftime('%d/%m/%Y')}. "
                        f"Portfólio otimizado começará na mesma data para comparação justa."
                    )
                    data_inicio_analise = data_mais_recente
                    
                    # Cortar dados para começar na data ajustada
                    df_returns = df_returns.loc[data_inicio_analise:]
                    df_cdi = df_cdi.loc[data_inicio_analise:]
                    for bench_name in df_benchmarks_adicionais.keys():
                        df_benchmarks_adicionais[bench_name] = df_benchmarks_adicionais[bench_name].loc[data_inicio_analise:]
        
        # Detecta escala
        if df_returns.abs().mean().mean() > 0.1:
            st.warning("Detectado dados em percentual. Convertendo para decimal.")
            df_returns = df_returns / 100.0
        
        # Verifica escala do CDI
        cdi_mean_abs = df_cdi.abs().mean()
        if cdi_mean_abs > 0.1:
            df_cdi = df_cdi / 100.0
        
        # Verifica escala dos benchmarks adicionais
        for bench_name, bench_data in df_benchmarks_adicionais.items():
            bench_mean_abs = bench_data.abs().mean()
            if bench_mean_abs > 0.1:
                df_benchmarks_adicionais[bench_name] = bench_data / 100.0
    
    # Criar bounds
    bounds = tuple((min_weights[a], max_weights[a]) for a in ativos_sel)
    
    # Executar backtest da estratégia principal
    with st.spinner(f"Executando backtest com estratégia {strategy_options[strategy]}..."):
        backtester = PortfolioBacktester(df_returns, df_cdi, ativos_sel)
        
        results = backtester.run(
            lookback_days=lookback_days,
            rebalance_freq=rebalance_freq,
            strategy=strategy,
            target_vol=target_vol,
            bounds=bounds
        )
        
        if results is None:
            st.error("Histórico insuficiente para os parâmetros escolhidos.")
            st.stop()
    
    # Executar backtest Equal-Weight (benchmark automático)
    with st.spinner("Calculando benchmark Equal-Weight..."):
        backtester_ew = PortfolioBacktester(df_returns, df_cdi, ativos_sel)
        
        results_ew = backtester_ew.run(
            lookback_days=lookback_days,
            rebalance_freq=rebalance_freq,
            strategy='equal_weight',
            target_vol=None,
            bounds=bounds
        )
    
    # Processar benchmarks adicionais (se houver)
    results_benchmarks = []
    if df_benchmarks_adicionais:
        start_date = results['start_date']
        
        for bench_name, bench_data in df_benchmarks_adicionais.items():
            # Alinhar com o período da estratégia principal
            df_bench_aligned = bench_data.loc[start_date:]
            
            # Calcular retorno acumulado
            cum_bench = (1 + df_bench_aligned).cumprod()
            
            results_benchmarks.append({
                'cum_ret': cum_bench,
                'returns': df_bench_aligned,
                'name': bench_name
            })
    
    # Salvar resultados
    st.session_state['backtest_results'] = results
    st.session_state['backtest_results_ew'] = results_ew
    st.session_state['backtest_results_benchmarks'] = results_benchmarks
    st.session_state['backtest_params'] = {

            'strategy': strategy,
            'strategy_name': strategy_options[strategy],
            'lookback_days': lookback_days,
            'rebalance_freq': rebalance_freq,
            'ativos': ativos_sel
        }

# ==========================================
# EXIBIÇÃO DOS RESULTADOS
# ==========================================

if 'backtest_results' in st.session_state:
    results = st.session_state['backtest_results']
    params = st.session_state['backtest_params']
    
    st.success(f"Backtest concluído! Estratégia: {params['strategy_name']}")
    
    # Informação sobre o período
    start_date = results['start_date']
    end_date = results['cum_ret_portfolio'].index[-1]
    
    st.info(
        f"Período de Simulação: {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')} "
        f"({len(results['cum_ret_portfolio'])} dias úteis) | "
        f"Rebalanceamentos: {len(results['rebalance_dates'])}"
    )
    
    # --- MÉTRICAS PRINCIPAIS ---
    st.header("Métricas de Performance")
    
    metrics = results['metrics']
    rolling = results['rolling']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric(
        "Retorno Total",
        f"{metrics['total_return']:.2%}",
        delta=f"{(metrics['total_return'] - metrics['cdi_total_return']):.2%} vs CDI"
    )
    
    col2.metric(
        "Retorno Anualizado",
        f"{metrics['ann_return']:.2%}",
        delta=f"{(metrics['ann_return'] - metrics['cdi_ann_return']):.2%} vs CDI"
    )
    
    col3.metric(
        "Volatilidade (a.a.)",
        f"{metrics['ann_vol']:.2%}"
    )
    
    col4.metric(
        "Sharpe Ratio",
        f"{metrics['sharpe']:.2f}",
        help="(Retorno - CDI) / Volatilidade"
    )
    
    col5.metric(
        "Drawdown Máximo",
        f"{metrics['max_drawdown']:.2%}"
    )
    
    # Métricas adicionais
    col6, col7, col8 = st.columns(3)
    
    col6.metric(
        "CDI Total",
        f"{metrics['cdi_total_return']:.2%}"
    )
    
    col7.metric(
        "% CDI (Rolling 252d)",
        f"{rolling['avg_pct_cdi']:.1f}%",
        help="Média do retorno rolling de 252 dias como % do CDI"
    )
    
    col8.metric(
        "Rebalanceamentos",
        f"{len(results['rebalance_dates'])}"
    )
    
    st.divider()
    
    # --- TABELA COMPARATIVA ---
    st.header("Comparação de Estratégias")
    
    # Recuperar benchmarks
    results_ew = st.session_state.get('backtest_results_ew')
    results_benchmarks = st.session_state.get('backtest_results_benchmarks', [])
    
    # Criar tabela comparativa
    comp_data = []
    
    # Estratégia principal
    comp_data.append({
        'Estratégia': params['strategy_name'],
        'Retorno Total': f"{metrics['total_return']:.2%}",
        'Retorno Anualizado': f"{metrics['ann_return']:.2%}",
        'Volatilidade': f"{metrics['ann_vol']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe']:.2f}",
        'Drawdown Máximo': f"{metrics['max_drawdown']:.2%}"
    })
    
    # Equal-Weight
    if results_ew:
        metrics_ew = results_ew['metrics']
        comp_data.append({
            'Estratégia': 'Equal-Weight',
            'Retorno Total': f"{metrics_ew['total_return']:.2%}",
            'Retorno Anualizado': f"{metrics_ew['ann_return']:.2%}",
            'Volatilidade': f"{metrics_ew['ann_vol']:.2%}",
            'Sharpe Ratio': f"{metrics_ew['sharpe']:.2f}",
            'Drawdown Máximo': f"{metrics_ew['max_drawdown']:.2%}"
        })
    
    # Benchmarks adicionais
    if results_benchmarks:
        start_date = results['start_date']
        cdi_aligned = df_cdi.loc[start_date:]
        cdi_ann_ret = cdi_aligned.mean() * 252
        
        for bench_result in results_benchmarks:
            bench_returns = bench_result['returns']
            bench_total_ret = bench_result['cum_ret'].iloc[-1] - 1
            bench_ann_ret = bench_returns.mean() * 252
            bench_ann_vol = bench_returns.std() * np.sqrt(252)
            
            bench_sharpe = (bench_ann_ret - cdi_ann_ret) / bench_ann_vol if bench_ann_vol > 0 else 0
            
            # Drawdown
            cum_bench = bench_result['cum_ret'] / bench_result['cum_ret'].iloc[0]
            running_max_bench = cum_bench.cummax()
            drawdown_bench = (cum_bench - running_max_bench) / running_max_bench
            max_dd_bench = drawdown_bench.min()
            
            comp_data.append({
                'Estratégia': bench_result['name'],
                'Retorno Total': f"{bench_total_ret:.2%}",
                'Retorno Anualizado': f"{bench_ann_ret:.2%}",
                'Volatilidade': f"{bench_ann_vol:.2%}",
                'Sharpe Ratio': f"{bench_sharpe:.2f}",
                'Drawdown Máximo': f"{max_dd_bench:.2%}"
            })
    
    # CDI
    comp_data.append({
        'Estratégia': 'CDI',
        'Retorno Total': f"{metrics['cdi_total_return']:.2%}",
        'Retorno Anualizado': f"{metrics['cdi_ann_return']:.2%}",
        'Volatilidade': '-',
        'Sharpe Ratio': '-',
        'Drawdown Máximo': '-'
    })
    
    df_comp = pd.DataFrame(comp_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # --- GRÁFICOS ---
    st.header("Análise Gráfica")
    
    # Recuperar benchmarks
    results_ew = st.session_state.get('backtest_results_ew')
    results_benchmarks = st.session_state.get('backtest_results_benchmarks', [])
    
    # Gráfico 1: Performance Acumulada
    fig1 = go.Figure()
    
    # Estratégia principal
    fig1.add_trace(go.Scatter(
        x=results['cum_ret_portfolio'].index,
        y=results['cum_ret_portfolio'].values,
        name=f"Portfólio ({params['strategy_name']})",
        line=dict(color='#2E86AB', width=3)
    ))
    
    # CDI
    fig1.add_trace(go.Scatter(
        x=results['cum_cdi'].index,
        y=results['cum_cdi'].values,
        name='CDI',
        line=dict(color='#06A77D', width=2, dash='dot')
    ))
    
    # Equal-Weight (benchmark automático)
    if results_ew:
        fig1.add_trace(go.Scatter(
            x=results_ew['cum_ret_portfolio'].index,
            y=results_ew['cum_ret_portfolio'].values,
            name='Equal-Weight (Benchmark)',
            line=dict(color='#F77F00', width=2, dash='dash')
        ))
    
    # Benchmarks adicionais (se houver)
    if results_benchmarks:
        # Cores para benchmarks adicionais
        bench_colors = ['#9B2226', '#5F0F40', '#0F4C5C', '#E36414', '#FB8B24']
        for idx, bench_result in enumerate(results_benchmarks):
            color = bench_colors[idx % len(bench_colors)]
            fig1.add_trace(go.Scatter(
                x=bench_result['cum_ret'].index,
                y=bench_result['cum_ret'].values,
                name=bench_result['name'],
                line=dict(color=color, width=2, dash='dashdot')
            ))
    
    fig1.update_layout(
        title=f"Performance Acumulada | {params['strategy_name']} | Rebal: {params['rebalance_freq']} | Lookback: {params['lookback_days']}d",
        xaxis_title="",
        yaxis_title="Valor Acumulado (Base 1)",
        template="plotly_white",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Gráfico 2: Drawdown
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=results['drawdown'].index,
        y=results['drawdown'].values * 100,
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#D62828', width=2),
        fillcolor='rgba(214, 40, 40, 0.3)'
    ))
    
    fig2.update_layout(
        title="Drawdown (%)",
        xaxis_title="",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode='x unified',
        height=400
    )
    
    fig2.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico 3: Rolling Returns
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=rolling['portfolio'].index,
            y=rolling['portfolio'].values * 100,
            name='Portfólio',
            line=dict(color='#2E86AB', width=2.5)
        ))
        
        fig3.add_trace(go.Scatter(
            x=rolling['cdi'].index,
            y=rolling['cdi'].values * 100,
            name='CDI',
            line=dict(color='#06A77D', width=2.5, dash='dot')
        ))
        
        fig3.update_layout(
            title="Rolling Return (252 dias)",
            xaxis_title="",
            yaxis_title="Retorno Anualizado (%)",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        
        fig3.add_hline(y=0, line_dash="solid", line_color="black", line_width=0.8, opacity=0.5)
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_g2:
        fig4 = go.Figure()
        
        fig4.add_trace(go.Scatter(
            x=rolling['vs_cdi'].index,
            y=rolling['vs_cdi'].values,
            name='% CDI (Rolling 252d)',
            line=dict(color='#7209B7', width=2.5)
        ))
        
        fig4.add_hline(
            y=100, 
            line_dash="dash", 
            line_color="#06A77D", 
            line_width=2,
            annotation_text="100% CDI"
        )
        
        fig4.add_hline(
            y=rolling['avg_pct_cdi'], 
            line_dash="dashdot", 
            line_color="#F77F00", 
            line_width=2.5,
            annotation_text=f"Média: {rolling['avg_pct_cdi']:.1f}%"
        )
        
        fig4.update_layout(
            title="Rolling Return - % do CDI",
            xaxis_title="",
            yaxis_title="% do CDI",
            template="plotly_white",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    # Gráfico 4: Alocação Dinâmica
    st.subheader("Alocação Dinâmica de Ativos")
    
    daily_weights = results['daily_weights']
    
    fig5 = go.Figure()
    
    for ativo in params['ativos']:
        fig5.add_trace(go.Scatter(
            x=daily_weights.index,
            y=daily_weights[ativo].values,
            name=ativo,
            stackgroup='one',
            mode='none'
        ))
    
    # Adicionar linhas verticais nos rebalanceamentos
    for reb_date in results['rebalance_dates']:
        fig5.add_vline(
            x=reb_date, 
            line_dash="dash", 
            line_color="red", 
            opacity=0.4,
            line_width=1
        )
    
    fig5.update_layout(
        title="Alocação Dinâmica com Drift e Rebalanceamentos (linhas vermelhas)",
        xaxis_title="",
        yaxis_title="Alocação",
        template="plotly_white",
        hovermode='x unified',
        height=500,
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    st.divider()
    
    # --- TABELA DE REBALANCEAMENTOS ---
    st.header("Histórico de Rebalanceamentos")
    
    rebal_weights = results['rebalance_weights'] * 100
    rebal_weights.index = rebal_weights.index.strftime('%d/%m/%Y')
    
    st.dataframe(
        rebal_weights.style.format("{:.2f}%"),
        use_container_width=True
    )
    
    # Download
    csv = rebal_weights.to_csv()
    st.download_button(
        label="Download Histórico (CSV)",
        data=csv,
        file_name=f"rebalanceamentos_{params['strategy']}.csv",
        mime="text/csv"
    )

else:
    st.info("Configure os parâmetros e execute o backtest para ver os resultados.")
