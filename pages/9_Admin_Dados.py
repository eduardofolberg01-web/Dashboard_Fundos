import streamlit as st
import pandas as pd
import sys
import os
import time

# Adicionar root ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from modulos.db_manager import RiskDatabase

st.set_page_config(
    page_title="Administração de Dados",
    layout="wide",
    page_icon="⚙️"
)

st.title("⚙️ Administração de Dados")

db = RiskDatabase(os.path.join(root_dir, 'risk_system.db'))

# Abas
tab1, tab2 = st.tabs(["Atualização Semanal", "Estatísticas da Base"])

with tab1:
    st.header("Sincronização de Dados")
    st.markdown("""
    Use esta ferramenta para atualizar o banco de dados do sistema com as novas cotas do Excel.
    
    **Fluxo:**
    1. Atualize seu Excel `Quant_Fundos.xlsm` normalmente.
    2. Clique no botão abaixo.
    3. O sistema importará apenas os dados novos.
    """)
    
    arquivo_excel = os.path.join(root_dir, 'Quant_Fundos.xlsm')
    
    if os.path.exists(arquivo_excel):
        data_modificacao = os.path.getmtime(arquivo_excel)
        data_str = time.strftime('%d/%m/%Y %H:%M:%S', time.localtime(data_modificacao))
        st.info(f"📁 Arquivo Excel detectado: `Quant_Fundos.xlsm` (Última modificação: {data_str})")
        
        if st.button("🔄 Sincronizar Agora", type="primary"):
            with st.status("Processando atualização...", expanded=True) as status:
                st.write("Conectando ao banco de dados...")
                try:
                    # Aqui chamaríamos uma versão otimizada do importador que só pega o delta
                    # Por enquanto, usando o importador completo como exemplo
                    st.write("Lendo dados do Excel...")
                    db.importar_excel_principal(arquivo_excel)
                    st.write("✅ Dados importados com sucesso!")
                    status.update(label="Atualização Completa!", state="complete", expanded=False)
                    st.success("Base de dados atualizada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro durante a atualização: {e}")
                    status.update(label="Erro na atualização", state="error")
    else:
        st.error(f"Arquivo `Quant_Fundos.xlsm` não encontrado em {root_dir}")

with tab2:
    st.header("Diagnóstico do Banco de Dados")
    
    if st.button("Carregar Estatísticas"):
        try:
            stats = db.obter_stats()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ativos Cadastrados", stats.get('ativos', 0))
            c2.metric("Total de Cotas", f"{stats.get('cotas', 0):,}")
            c3.metric("Data Início", stats.get('data_inicio', '-'))
            c4.metric("Data Fim (Última Cota)", stats.get('data_fim', '-'))
            
            st.success("Conexão com banco de dados OK!")
        except Exception as e:
            st.error(f"Não foi possível conectar ao banco de dados: {e}")

