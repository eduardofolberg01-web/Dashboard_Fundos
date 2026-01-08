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

st.set_page_config(page_title="Universo de Análise RV & MM", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #0d3b66; }
</style>
""", unsafe_allow_html=True)

st.title("Universo de Análise RV & MM")

# --- Carregar Dados ---
@st.cache_data
def load_universo_data():
    """Carrega os dados do arquivo Universo Investiveis RV e MM.xlsx"""
    try:
        df = pd.read_excel(
            'Universo Investiveis RV e MM.xlsx', 
            sheet_name='Base Analise'
        )
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

# Carregar dados
df_universo = load_universo_data()

if df_universo is None:
    st.stop()

# Remover CNPJs duplicados (manter apenas a primeira ocorrência)
if 'CNPJ do Fundo' in df_universo.columns:
    df_universo = df_universo.drop_duplicates(subset=['CNPJ do Fundo'], keep='first')
    

# Extrair data de referência do nome da coluna (ex: "Retorno\nMês Atual\n02/01/2026")
try:
    # Pegar a primeira coluna de retorno para extrair a data
    col_retorno = [c for c in df_universo.columns if 'Retorno' in str(c) and '02/01/2026' in str(c)][0]
    data_referencia = col_retorno.split('\n')[-1]
    st.info(f"**Dados atualizados até:** {data_referencia}")
except:
    pass

# --- Filtros na Barra Lateral ---
with st.sidebar:
    st.header("Filtros")
    
    # Obter classes únicas
    classes_disponiveis = sorted(df_universo['Classe CVM do Fundo'].dropna().unique().tolist())
    
    # Inicializar session_state para o filtro se não existir
    if 'universo_classe_selecionada' not in st.session_state:
        st.session_state.universo_classe_selecionada = "Todas"
    
    # Encontrar o índice da classe selecionada
    opcoes = ["Todas"] + classes_disponiveis
    try:
        index_atual = opcoes.index(st.session_state.universo_classe_selecionada)
    except ValueError:
        index_atual = 0
        st.session_state.universo_classe_selecionada = "Todas"
    
    classe_selecionada = st.selectbox(
        "Filtrar por Classe CVM",
        options=opcoes,
        index=index_atual,
        key='selectbox_classe_universo'
    )
    
    # Atualizar session_state
    st.session_state.universo_classe_selecionada = classe_selecionada
    
    # Aplicar filtro
    if classe_selecionada != "Todas":
        df_filtrado = df_universo[df_universo['Classe CVM do Fundo'] == classe_selecionada].copy()
    else:
        df_filtrado = df_universo.copy()
    
    st.divider()
    st.info(f"**{len(df_filtrado)}** fundos no universo filtrado")

# --- Preparar DataFrame para Exibição ---
if not df_filtrado.empty:
    # Renomear colunas para formato mais limpo
    # Mapeamento de colunas originais para nomes de exibição
    
    # Identificar colunas dinamicamente
    cols_retorno = [c for c in df_filtrado.columns if 'Retorno\n' in str(c) and 'meses' in str(c)]
    cols_sharpe = [c for c in df_filtrado.columns if 'Sharpe ao período' in str(c)]
    cols_vol = [c for c in df_filtrado.columns if 'Volatilidade Anualizada' in str(c)]
    
    # Criar mapeamento de renomeação
    rename_map = {
        'nome_fundo': 'Nome do Fundo',
        'CNPJ do Fundo': 'CNPJ',
        'Classe CVM do Fundo': 'Classe CVM',
        'Data da Constituição do Fundo': 'Data Constituição'
    }
    
    # Adicionar colunas de retorno
    for col in cols_retorno:
        if '12 meses' in col:
            rename_map[col] = 'Retorno 12m'
        elif '24 meses' in col:
            rename_map[col] = 'Retorno 24m'
        elif '36 meses' in col:
            rename_map[col] = 'Retorno 36m'
        elif '48 meses' in col:
            rename_map[col] = 'Retorno 48m'
        elif '60 meses' in col:
            rename_map[col] = 'Retorno 60m'
    
    # Adicionar colunas de Sharpe
    for col in cols_sharpe:
        if '12 meses' in col:
            rename_map[col] = 'Sharpe 12m'
        elif '24 meses' in col:
            rename_map[col] = 'Sharpe 24m'
        elif '36 meses' in col:
            rename_map[col] = 'Sharpe 36m'
        elif '48 meses' in col:
            rename_map[col] = 'Sharpe 48m'
        elif '60 meses' in col:
            rename_map[col] = 'Sharpe 60m'
    
    # Adicionar colunas de Volatilidade
    for col in cols_vol:
        if '12 meses' in col:
            rename_map[col] = 'Vol 12m'
        elif '24 meses' in col:
            rename_map[col] = 'Vol 24m'
        elif '36 meses' in col:
            rename_map[col] = 'Vol 36m'
    
    # Selecionar e renomear colunas
    cols_to_show = ['nome_fundo', 'CNPJ do Fundo', 'Classe CVM do Fundo', 'Data da Constituição do Fundo']
    cols_to_show += cols_retorno
    cols_to_show += cols_sharpe
    cols_to_show += cols_vol
    
    # Filtrar apenas colunas que existem
    cols_to_show = [c for c in cols_to_show if c in df_filtrado.columns]
    
    df_display = df_filtrado[cols_to_show].copy()
    df_display = df_display.rename(columns=rename_map)
    
    # Os valores já vêm em formato percentual do Excel (4.5 = 4.5%)
    # Não é necessário multiplicar por 100
    
    # Ordenar por Retorno 12m (decrescente)
    if 'Retorno 12m' in df_display.columns:
        df_display = df_display.sort_values('Retorno 12m', ascending=False)
    
    # Converter Data Constituição para string formatada
    if 'Data Constituição' in df_display.columns:
        df_display['Data Constituição'] = pd.to_datetime(
            df_display['Data Constituição'], 
            errors='coerce'
        ).dt.strftime('%d/%m/%Y')
        # Substituir NaT por string vazia
        df_display['Data Constituição'] = df_display['Data Constituição'].fillna('-')
    
    # --- Configuração de Colunas para Exibição ---
    column_config = {
        "Nome do Fundo": st.column_config.TextColumn("Nome do Fundo", width=400, pinned=True),
        "CNPJ": st.column_config.TextColumn("CNPJ", width="small"),
        "Classe CVM": st.column_config.TextColumn("Classe CVM", width="small"),
        "Data Constituição": st.column_config.TextColumn("Data Constituição", width="small"),
        "Qtd 1º Quartil": st.column_config.NumberColumn(
            "Qtd 1º Quartil",
            help="Quantidade de janelas em que o fundo está no 1º quartil",
            width="small",
            format="%d"
        ),
    }
    
    # Colunas de Retorno (percentual com 2 casas) - largura pequena
    cols_ret = [c for c in df_display.columns if 'Retorno' in c]
    for c in cols_ret:
        column_config[c] = st.column_config.NumberColumn(c, format="%.2f%%", width="small")
    
    # Colunas de Sharpe (decimal com 2 casas) - largura pequena
    cols_sharpe_display = [c for c in df_display.columns if 'Sharpe' in c]
    for c in cols_sharpe_display:
        column_config[c] = st.column_config.NumberColumn(c, format="%.2f", width="small")
    
    # Colunas de Volatilidade (percentual com 2 casas) - largura pequena
    cols_vol_display = [c for c in df_display.columns if 'Vol' in c]
    for c in cols_vol_display:
        column_config[c] = st.column_config.NumberColumn(c, format="%.2f%%", width="small")
    
    # --- Calcular Quartis por Classe CVM ---
    # Adicionar colunas de quartil para cada janela de retorno
    cols_retorno_janelas = ['Retorno 12m', 'Retorno 24m', 'Retorno 36m', 'Retorno 48m', 'Retorno 60m']
    
    for col_ret in cols_retorno_janelas:
        if col_ret in df_display.columns:
            # Criar coluna de quartil
            quartil_col = f'{col_ret}_Quartil'
            df_display[quartil_col] = np.nan
            
            # Calcular quartil por classe CVM
            for classe in df_display['Classe CVM'].unique():
                mask_classe = df_display['Classe CVM'] == classe
                valores_classe = df_display.loc[mask_classe, col_ret].dropna()
                
                if len(valores_classe) > 0:
                    # Calcular quartis (Q1 = melhor, Q4 = pior)
                    quartis = pd.qcut(valores_classe, q=4, labels=[1, 2, 3, 4], duplicates='drop')
                    # Inverter para que Q1 seja o melhor (maiores retornos)
                    quartis = 5 - quartis.astype(int)
                    df_display.loc[mask_classe & df_display[col_ret].notna(), quartil_col] = quartis
    
    # --- Aplicar Estilo Condicional com Quartis ---
    def style_quartil_retorno(row):
        """Aplicar cor de fundo baseado no quartil de retorno"""
        styles = [''] * len(row)
        
        for i, col in enumerate(df_display.columns):
            if col in cols_retorno_janelas:
                quartil_col = f'{col}_Quartil'
                if quartil_col in row.index:
                    quartil = row[quartil_col]
                    
                    if pd.notna(quartil):
                        # Cores de fundo por quartil
                        if quartil == 1:  # Primeiro quartil (melhor)
                            bg_color = '#90EE90'  # Verde claro
                            styles[i] = f'background-color: {bg_color}; font-weight: bold; position: relative;'
                        elif quartil == 2:  # Segundo quartil
                            bg_color = '#D4EDDA'  # Verde muito claro
                            styles[i] = f'background-color: {bg_color};'
                        elif quartil == 3:  # Terceiro quartil
                            bg_color = '#FFF3CD'  # Amarelo claro
                            styles[i] = f'background-color: {bg_color};'
                        elif quartil == 4:  # Quarto quartil (pior)
                            bg_color = '#F8D7DA'  # Vermelho claro
                            styles[i] = f'background-color: {bg_color};'
        
        return styles
    
    def color_sharpe(val):
        """Colorir Sharpe: verde se >= 1, laranja se >= 0.5, vermelho se < 0.5"""
        if pd.isna(val):
            return None
        if val >= 1.0:
            color = '#2ca02c'  # Verde
        elif val >= 0.5:
            color = '#ff7f0e'  # Laranja
        else:
            color = '#d62728'  # Vermelho
        return f'color: {color}; font-weight: bold'
    
    # Salvar valores numéricos para estatísticas
    df_stats = df_display[['Retorno 12m', 'Sharpe 12m', 'Vol 12m']].copy() if all(c in df_display.columns for c in ['Retorno 12m', 'Sharpe 12m', 'Vol 12m']) else pd.DataFrame()
    
    # Calcular quantidade de vezes que o fundo está no 1º quartil
    cols_quartil = [f'{col}_Quartil' for col in cols_retorno_janelas if f'{col}_Quartil' in df_display.columns]
    
    if cols_quartil:
        # Contar quantas vezes o fundo está no 1º quartil (valor = 1)
        df_display['Qtd 1º Quartil'] = df_display[cols_quartil].apply(
            lambda row: (row == 1).sum(),
            axis=1
        )
    else:
        df_display['Qtd 1º Quartil'] = 0
    
    # NÃO converter valores para string - manter numéricos
    # Remover colunas de quartil da exibição (são auxiliares)
    cols_to_hide = [c for c in df_display.columns if '_Quartil' in c]
    df_display_final = df_display.drop(columns=cols_to_hide)
    
    # Reordenar colunas: colocar "Qtd 1º Quartil" antes das colunas de retorno
    if 'Qtd 1º Quartil' in df_display_final.columns:
        # Identificar colunas base
        cols_base = ['Nome do Fundo', 'CNPJ', 'Classe CVM', 'Data Constituição']
        cols_base = [c for c in cols_base if c in df_display_final.columns]
        
        # Identificar colunas de métricas
        cols_retorno_final = [c for c in df_display_final.columns if 'Retorno' in c]
        cols_sharpe_final = [c for c in df_display_final.columns if 'Sharpe' in c]
        cols_vol_final = [c for c in df_display_final.columns if 'Vol' in c]
        
        # Montar ordem: base + Qtd 1º Quartil + retornos + sharpe + vol
        new_order = cols_base + ['Qtd 1º Quartil'] + cols_retorno_final + cols_sharpe_final + cols_vol_final
        
        # Garantir que todas as colunas existem
        new_order = [c for c in new_order if c in df_display_final.columns]
        
        # Reordenar
        df_display_final = df_display_final[new_order]
        
        # Ordenar por Qtd 1º Quartil (decrescente) e depois por Retorno 12m (decrescente)
        sort_cols = ['Qtd 1º Quartil']
        if 'Retorno 12m' in df_display_final.columns:
            sort_cols.append('Retorno 12m')
        
        df_display_final = df_display_final.sort_values(sort_cols, ascending=False)
    
    # Criar Styler final
    styler = df_display_final.style
    
    # Aplicar estilo de quartil aos retornos com medalha
    def style_quartil_retorno_final(row):
        """Aplicar cor de fundo baseado no quartil de retorno"""
        styles = [''] * len(row)
        
        for i, col in enumerate(df_display_final.columns):
            if col in cols_retorno_janelas:
                # Buscar quartil do df_display original
                idx = row.name
                if idx in df_display.index:
                    quartil_col = f'{col}_Quartil'
                    if quartil_col in df_display.columns:
                        quartil = df_display.loc[idx, quartil_col]
                        
                        if pd.notna(quartil):
                            # Cores de fundo por quartil (paleta suave)
                            if quartil == 1:  # Primeiro quartil (melhor)
                                bg_color = '#C8E6C9'  # Verde claro suave
                                styles[i] = f'background-color: {bg_color}; font-weight: bold;'
                            elif quartil == 2:  # Segundo quartil
                                bg_color = '#FFF9C4'  # Amarelo claro
                                styles[i] = f'background-color: {bg_color};'
                            elif quartil == 3:  # Terceiro quartil
                                bg_color = '#FFE0B2'  # Laranja claro
                                styles[i] = f'background-color: {bg_color};'
                            elif quartil == 4:  # Quarto quartil (pior)
                                bg_color = '#FFCDD2'  # Vermelho claro
                                styles[i] = f'background-color: {bg_color};'
        
        return styles
    
    styler = styler.apply(style_quartil_retorno_final, axis=1)
    
    # Aplicar cores ao Sharpe
    for col in cols_sharpe_display:
        if col in df_display_final.columns:
            styler = styler.map(color_sharpe, subset=[col])
    
    
    # Criar cópias formatadas das colunas de retorno com medalha (solução nativa Python/Streamlit)
    for col_ret in cols_retorno_janelas:
        if col_ret in df_display_final.columns:
            quartil_col = f'{col_ret}_Quartil'
            
            if quartil_col in df_display.columns:
                # Formatar valores como string com medalha para Q1
                df_display_final[col_ret] = df_display_final.apply(
                    lambda row: (
                        f"{row[col_ret]:.2f}% 🥇" if pd.notna(row[col_ret]) and 
                        df_display.loc[row.name, quartil_col] == 1 
                        else (f"{row[col_ret]:.2f}%" if pd.notna(row[col_ret]) else "")
                    ),
                    axis=1
                )
    
    # Atualizar column_config para usar TextColumn nas colunas de retorno (já que agora são strings)
    for col_ret in cols_retorno_janelas:
        if col_ret in df_display_final.columns:
            column_config[col_ret] = st.column_config.TextColumn(col_ret, width=90)
    
    # Recriar styler com valores já formatados
    styler = df_display_final.style
    styler = styler.apply(style_quartil_retorno_final, axis=1)
    
    # Aplicar cores ao Sharpe
    for col in cols_sharpe_display:
        if col in df_display_final.columns:
            styler = styler.map(color_sharpe, subset=[col])
    
    # --- Estatísticas do 1º Quartil por Classe (ANTES DA TABELA) ---
    st.divider()
    st.subheader("Estatísticas do 1º Quartil")
    st.caption("Média e Mediana dos retornos dos fundos que estão no 1º quartil de cada janela")
    
    # Preparar dados: filtrar apenas fundos do 1º quartil em cada janela
    janelas = ['12m', '24m', '36m', '48m', '60m']
    classes = df_display['Classe CVM'].unique()
    
    # Criar abas para cada classe
    tabs = st.tabs([f"📈 {classe}" for classe in classes])
    
    for idx, classe in enumerate(classes):
        with tabs[idx]:
            # Filtrar fundos da classe
            mask_classe = df_display['Classe CVM'] == classe
            
            # Criar DataFrame de estatísticas
            stats_data = []
            
            for janela in janelas:
                col_ret = f'Retorno {janela}'
                col_quartil = f'{col_ret}_Quartil'
                
                if col_ret in df_display.columns and col_quartil in df_display.columns:
                    # Filtrar fundos do 1º quartil desta janela e classe
                    mask_q1 = (df_display[col_quartil] == 1) & mask_classe & df_display[col_ret].notna()
                    valores_q1 = df_display.loc[mask_q1, col_ret]
                    
                    if len(valores_q1) > 0:
                        stats_data.append({
                            'Janela': janela,
                            'Qtd Fundos': len(valores_q1),
                            'Média': valores_q1.mean(),
                            'Mediana': valores_q1.median(),
                            'Mínimo': valores_q1.min(),
                            'Máximo': valores_q1.max()
                        })
            
            if stats_data:
                df_stats_classe = pd.DataFrame(stats_data)
                
                # Exibir métricas em colunas
                st.markdown(f"**{classe}**")
                
                cols = st.columns(len(janelas))
                
                for i, janela in enumerate(janelas):
                    with cols[i]:
                        row = df_stats_classe[df_stats_classe['Janela'] == janela]
                        
                        if not row.empty:
                            media = row['Média'].iloc[0]
                            mediana = row['Mediana'].iloc[0]
                            qtd = int(row['Qtd Fundos'].iloc[0])
                            
                            st.metric(
                                label=f"**{janela}**",
                                value=f"{media:.2f}%",
                                delta=None,
                                help=f"Média do 1º quartil ({qtd} fundos)"
                            )
                            st.caption(f"Mediana: {mediana:.2f}%")
                        else:
                            st.metric(
                                label=f"**{janela}**",
                                value="-",
                                delta=None
                            )
                
                # Tabela detalhada (opcional, colapsável)
                with st.expander("📋 Ver tabela detalhada"):
                    # Formatar DataFrame para exibição
                    df_display_stats = df_stats_classe.copy()
                    df_display_stats['Média'] = df_display_stats['Média'].apply(lambda x: f"{x:.2f}%")
                    df_display_stats['Mediana'] = df_display_stats['Mediana'].apply(lambda x: f"{x:.2f}%")
                    df_display_stats['Mínimo'] = df_display_stats['Mínimo'].apply(lambda x: f"{x:.2f}%")
                    df_display_stats['Máximo'] = df_display_stats['Máximo'].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(
                        df_display_stats,
                        hide_index=True,
                        use_container_width=True
                    )
            else:
                st.info(f"Sem dados de 1º quartil para {classe}")
    
    st.divider()
    
    # --- Exibir Tabela ---
    st.dataframe(
        styler,
        column_config=column_config,
        height=600,
        use_container_width=True,
        hide_index=True
    )
    
    # --- Tabelas de Fundos do 1º Quartil por Período ---
    st.divider()
    st.subheader("🏆 Fundos do 1º Quartil por Período")
    st.caption("Fundos que alcançaram o 1º quartil em cada janela de retorno")
    
    # Criar duas colunas para as tabelas
    col_acoes, col_mm = st.columns(2)
    
    janelas = ['12m', '24m', '36m', '48m', '60m']
    
    # Função para criar tabela de 1º quartil
    def criar_tabela_quartil(classe_nome, janelas_list):
        """Cria DataFrame com fundos do 1º quartil para cada janela"""
        # Filtrar fundos da classe
        mask_classe = df_display['Classe CVM'] == classe_nome
        
        # Dicionário para armazenar fundos por janela
        fundos_por_janela = {}
        max_fundos = 0
        
        for janela in janelas_list:
            col_ret = f'Retorno {janela}'
            col_quartil = f'{col_ret}_Quartil'
            
            if col_ret in df_display.columns and col_quartil in df_display.columns:
                # Filtrar fundos do 1º quartil
                mask_q1 = (df_display[col_quartil] == 1) & mask_classe & df_display[col_ret].notna()
                fundos_q1 = df_display.loc[mask_q1, ['Nome do Fundo', col_ret]].copy()
                
                # Ordenar por retorno (decrescente)
                fundos_q1 = fundos_q1.sort_values(col_ret, ascending=False)
                
                # Armazenar lista de fundos
                fundos_por_janela[janela] = fundos_q1['Nome do Fundo'].tolist()
                max_fundos = max(max_fundos, len(fundos_q1))
        
        # Criar DataFrame com colunas para cada janela
        data = {}
        for janela in janelas_list:
            if janela in fundos_por_janela:
                # Preencher com fundos e completar com vazios se necessário
                fundos = fundos_por_janela[janela]
                fundos_padded = fundos + [''] * (max_fundos - len(fundos))
                data[janela] = fundos_padded
            else:
                data[janela] = [''] * max_fundos
        
        return pd.DataFrame(data)
    
    # Tabela de Fundos de Ações
    with col_acoes:
        st.markdown("###  Fundo de Ações")
        df_acoes_q1 = criar_tabela_quartil('Fundo de Ações', janelas)
        
        if not df_acoes_q1.empty and df_acoes_q1.values.any():
            st.dataframe(
                df_acoes_q1,
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.info("Nenhum fundo de ações no 1º quartil")
    
    # Tabela de Fundos Multimercado
    with col_mm:
        st.markdown("###  Fundo Multimercado")
        df_mm_q1 = criar_tabela_quartil('Fundo Multimercado', janelas)
        
        if not df_mm_q1.empty and df_mm_q1.values.any():
            st.dataframe(
                df_mm_q1,
                hide_index=True,
                use_container_width=True,
                height=400
            )
        else:
            st.info("Nenhum fundo multimercado no 1º quartil")
    
    # --- Botão Download ---
    st.divider()
    csv = df_display_final.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📥 Baixar Tabela CSV",
        csv,
        "universo_analise_rv_mm.csv",
        "text/csv",
        key='download-csv'
    )

else:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
