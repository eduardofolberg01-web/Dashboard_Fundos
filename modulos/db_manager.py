import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime

class RiskDatabase:
    def __init__(self, db_path='risk_system.db'):
        self.db_path = db_path
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def create_schema(self):
        """Cria a estrutura de tabelas do banco de dados"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # 1. Tabela de Ativos (Fundos e Benchmarks)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ativos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            codigo_excel VARCHAR(100) UNIQUE,
            nome_exibicao VARCHAR(200),
            tipo VARCHAR(50),
            criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 2. Tabela de Classificações
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS classificacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo_id INTEGER,
            classe VARCHAR(100),
            subclasse VARCHAR(100),
            grupo VARCHAR(100),
            FOREIGN KEY (ativo_id) REFERENCES ativos(id)
        )
        ''')
        
        # 3. Tabela de Cotas (Série Temporal)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cotas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ativo_id INTEGER,
            data DATE,
            retorno DECIMAL(18, 10),
            FOREIGN KEY (ativo_id) REFERENCES ativos(id),
            UNIQUE(ativo_id, data)
        )
        ''')
        
        # Índices para performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cotas_ativo_data ON cotas(ativo_id, data)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_cotas_data ON cotas(data)')
        
        conn.commit()
        conn.close()
        print("Schema do banco de dados criado/verificado com sucesso.")

    def importar_excel_principal(self, file_path='Quant_Fundos.xlsm'):
        """Importa dados do arquivo Excel principal"""
        print(f"Iniciando importação de {file_path}...")
        
        if not os.path.exists(file_path):
            print(f"Erro: Arquivo {file_path} não encontrado.")
            return

        conn = self.get_connection()
        cursor = conn.cursor()
        
        # --- 1. Importar Classificação ---
        print("Lendo aba Classificação...")
        df_class = pd.read_excel(file_path, sheet_name='Classificação')
        
        # Limpar nomes de colunas
        df_class.columns = [str(c).strip().replace('\n', ' ') for c in df_class.columns]
        
        mapa_codigo_id = {}
        
        for _, row in df_class.iterrows():
            codigo = str(row.get('Codigo_Excel', '')).strip()
            nome = str(row.get('Nome_Exibiçao', '')).strip()
            tipo = str(row.get('Tipo', '')).strip()
            
            if not codigo or pd.isna(codigo):
                continue
                
            # Inserir Ativo
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO ativos (codigo_excel, nome_exibicao, tipo)
                    VALUES (?, ?, ?)
                ''', (codigo, nome, tipo))
                
                # Pegar o ID (seja do novo inserido ou do existente)
                cursor.execute('SELECT id FROM ativos WHERE codigo_excel = ?', (codigo,))
                ativo_id = cursor.fetchone()[0]
                mapa_codigo_id[codigo] = ativo_id
                
                # Inserir Classificação (Deletar anterior para garantir atualização)
                cursor.execute('DELETE FROM classificacoes WHERE ativo_id = ?', (ativo_id,))
                
                classe = row.get('Classe')
                subclasse = row.get('Subclasse')
                grupo = row.get('Grupo')
                
                # Tratar NaNs
                classe = str(classe) if pd.notna(classe) else None
                subclasse = str(subclasse) if pd.notna(subclasse) else None
                grupo = str(grupo) if pd.notna(grupo) else None
                
                cursor.execute('''
                    INSERT INTO classificacoes (ativo_id, classe, subclasse, grupo)
                    VALUES (?, ?, ?, ?)
                ''', (ativo_id, classe, subclasse, grupo))
                
            except Exception as e:
                print(f"Erro ao importar ativo {codigo}: {e}")
        
        conn.commit()
        print("Classificações importadas.")
        
        # --- 2. Importar Cotas ---
        print("Lendo aba Cotas (isso pode demorar um pouco)...")
        # Ler cotas tratando 'nd' como NaN
        df_cotas = pd.read_excel(file_path, sheet_name='Cotas', na_values=['nd'])
        
        if 'Data' in df_cotas.columns:
            df_cotas['Data'] = pd.to_datetime(df_cotas['Data'], dayfirst=True)
            df_cotas = df_cotas.set_index('Data')
        
        # Limpar headers
        df_cotas.columns = [str(c).strip() for c in df_cotas.columns]
        
        # Preparar dados para insert em lote
        # Helper para normalizar strings para comparação
        def normalize_key(text):
            return str(text).replace('\n', ' ').strip()
            
        print("Processando dados para inserção no banco...")
        
        data_to_insert = []
        count = 0
        total_cols = len(df_cotas.columns)
        
        # Para cada coluna (ativo) no Excel de cotas
        for i, col in enumerate(df_cotas.columns):
            # Tentar encontrar o ativo correspondente pelo código
            # Normalizamos ambos os lados para ignorar diferenças de quebra de linha
            col_norm = normalize_key(col)
            
            ativo_id = None
            for codigo, aid in mapa_codigo_id.items():
                if normalize_key(codigo) in col_norm:
                    ativo_id = aid
                    break
            
            if not ativo_id:
                # Se não achou mapeamento direto, pula
                continue
                
            # Pegar série de dados válida
            serie = df_cotas[col].dropna()
            
            for data, valor in serie.items():
                if pd.notna(valor):
                    # Formato data string YYYY-MM-DD para SQLite
                    data_str = data.strftime('%Y-%m-%d')
                    data_to_insert.append((ativo_id, data_str, float(valor)))
            
            # Commit parcial a cada 10 ativos para não estourar memória
            if len(data_to_insert) > 10000:
                print(f"Inserindo lote... ({i+1}/{total_cols} colunas processadas)")
                cursor.executemany('''
                    INSERT OR REPLACE INTO cotas (ativo_id, data, retorno)
                    VALUES (?, ?, ?)
                ''', data_to_insert)
                conn.commit()
                data_to_insert = []
        
        # Inserir restante
        if data_to_insert:
            cursor.executemany('''
                INSERT OR REPLACE INTO cotas (ativo_id, data, retorno)
                VALUES (?, ?, ?)
            ''', data_to_insert)
            conn.commit()
            
        conn.close()
        print("Importação concluída com sucesso!")

    def obter_stats(self):
        """Retorna estatísticas básicas do banco"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        cursor.execute('SELECT COUNT(*) FROM ativos')
        stats['ativos'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cotas')
        stats['cotas'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT MIN(data), MAX(data) FROM cotas')
        min_date, max_date = cursor.fetchone()
        stats['data_inicio'] = min_date
        stats['data_fim'] = max_date
        
        conn.close()
        return stats

# Bloco para execução direta se necessário
if __name__ == "__main__":
    db = RiskDatabase()
    db.create_schema()
    db.importar_excel_principal()
    print(db.obter_stats())
