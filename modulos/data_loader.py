import pandas as pd
import numpy as np
import os
import sqlite3
from datetime import datetime

class SFODataLoader:
    """
    Nova versão do SFODataLoader que utiliza Banco de Dados SQLite para alta performance.
    Mantém compatibilidade total com a interface da versão anterior.
    """
    def __init__(self, file_path='dados/Quant_Fundos.xlsm'):
        # Mantemos o file_path apenas para referência ou fallback, mas o foco é o DB
        self.file_path = file_path
        
        # Caminho do banco de dados (assumindo na raiz do projeto)
        # Tenta achar o DB relativo ao script ou ao cwd
        self.db_path = 'risk_system.db'
        if not os.path.exists(self.db_path):
            # Tenta um nível acima se estiver rodando de dentro de pages/
            if os.path.exists(os.path.join('..', 'risk_system.db')):
                self.db_path = os.path.join('..', 'risk_system.db')
            # Tenta caminho absoluto baseado no file_path
            elif os.path.exists(os.path.join(os.path.dirname(os.path.abspath(file_path)), 'risk_system.db')):
                self.db_path = os.path.join(os.path.dirname(os.path.abspath(file_path)), 'risk_system.db')
        
        self.mapa_nomes = {} # Cache de metadados
        self._carregar_metadados()

    def _get_connection(self):
        """Abre conexão com o banco"""
        return sqlite3.connect(self.db_path)

    def _carregar_metadados(self):
        """Carrega tabela de ativos e classificações para memória (rápido e pequeno)"""
        if not os.path.exists(self.db_path):
            print(f"AVISO CRÍTICO: Banco de dados {self.db_path} não encontrado.")
            return

        conn = self._get_connection()
        query = """
            SELECT 
                a.id, 
                a.codigo_excel as Codigo_Excel, 
                a.nome_exibicao as Nome_Exibicao, 
                a.tipo as Tipo,
                c.classe as Classe,
                c.subclasse as Subclasse,
                c.grupo as Grupo
            FROM ativos a
            LEFT JOIN classificacoes c ON a.id = c.ativo_id
        """
        try:
            df = pd.read_sql(query, conn)
            
            # Construir o mapa_nomes compatível com o formato antigo
            for _, row in df.iterrows():
                nome = row['Nome_Exibicao']
                if pd.isna(nome): continue
                
                self.mapa_nomes[nome] = {
                    'id_db': row['id'],  # Guardamos o ID numérico para queries rápidas
                    'coluna_original': row['Codigo_Excel'], # Mantido para compatibilidade
                    'tipo': row['Tipo'],
                    'classe': row['Classe'],
                    'subclasse': row['Subclasse'],
                    'grupo': row['Grupo']
                }
        except Exception as e:
            print(f"Erro ao carregar metadados do banco: {e}")
        finally:
            conn.close()

    def buscar_dados(self, lista_de_nomes):
        """
        Retorna um DataFrame com as séries temporais.
        Agora busca no SQLite com MUITO mais velocidade.
        """
        if isinstance(lista_de_nomes, str):
            lista_de_nomes = [lista_de_nomes]
            
        ids_para_buscar = []
        nomes_validos = []
        
        for nome in lista_de_nomes:
            if nome in self.mapa_nomes:
                ids_para_buscar.append(self.mapa_nomes[nome]['id_db'])
                nomes_validos.append(nome)
            else:
                print(f"Aviso: Ativo '{nome}' não encontrado no cadastro.")
                
        if not ids_para_buscar:
            return pd.DataFrame()
            
        conn = self._get_connection()
        
        # Query Otimizada: Buscamos apenas os IDs necessários
        # O placeholder '?' depende do DBAPI, para SQLite é ?
        placeholders = ','.join(['?'] * len(ids_para_buscar))
        
        query = f"""
            SELECT c.data, a.nome_exibicao, c.retorno
            FROM cotas c
            JOIN ativos a ON c.ativo_id = a.id
            WHERE c.ativo_id IN ({placeholders})
            ORDER BY c.data
        """
        
        try:
            # params precisa ser tupla
            df_long = pd.read_sql(query, conn, params=tuple(ids_para_buscar))
            
            if df_long.empty:
                return pd.DataFrame()
                
            # Converter data para datetime
            df_long['data'] = pd.to_datetime(df_long['data'])
            
            # Pivotar para formato Wide (uma coluna por ativo)
            # index=data, columns=nome_exibicao, values=retorno
            df_wide = df_long.pivot(index='data', columns='nome_exibicao', values='retorno')
            
            # Reordenar colunas conforme solicitado (opcional, mas bom pra manter ordem)
            cols_existentes = [col for col in nomes_validos if col in df_wide.columns]
            df_wide = df_wide[cols_existentes]
            
            return df_wide
            
        except Exception as e:
            print(f"Erro ao buscar dados no banco: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def listar_ativos(self, tipo=None, classe=None, subclasse=None, grupo=None):
        """Lista nomes amigáveis com filtros (memória local, instantâneo)"""
        resultado = []
        
        for nome, info in self.mapa_nomes.items():
            # Filtro por tipo (Normalização segura)
            info_tipo = str(info.get('tipo', ''))
            req_tipo = str(tipo) if tipo else None
            
            if req_tipo and info_tipo.lower() != req_tipo.lower():
                continue
            
            # Filtro por classe
            if classe is not None and info.get('classe') != classe:
                continue
            
            # Filtro por subclasse
            if subclasse is not None and info.get('subclasse') != subclasse:
                continue
                
            # Filtro por grupo
            if grupo is not None and info.get('grupo') != grupo:
                continue
            
            resultado.append(nome)
        
        return sorted(resultado)
    
    def listar_classes(self, tipo=None):
        classes = set()
        for nome, info in self.mapa_nomes.items():
            info_tipo = str(info.get('tipo', ''))
            req_tipo = str(tipo) if tipo else None
            if req_tipo and info_tipo.lower() != req_tipo.lower():
                continue
            if info.get('classe'):
                classes.add(info['classe'])
        return sorted(list(classes))
    
    def listar_subclasses(self, tipo=None, classe=None):
        subclasses = set()
        for nome, info in self.mapa_nomes.items():
            info_tipo = str(info.get('tipo', ''))
            req_tipo = str(tipo) if tipo else None
            
            if req_tipo and info_tipo.lower() != req_tipo.lower():
                continue
            if classe and info.get('classe') != classe:
                continue
            if info.get('subclasse'):
                subclasses.add(info['subclasse'])
        return sorted(list(subclasses))
        
    def listar_grupos(self, tipo=None, classe=None, subclasse=None):
        grupos = set()
        for nome, info in self.mapa_nomes.items():
            info_tipo = str(info.get('tipo', ''))
            req_tipo = str(tipo) if tipo else None
            
            if req_tipo and info_tipo.lower() != req_tipo.lower():
                continue
            if classe and info.get('classe') != classe:
                continue
            if subclasse and info.get('subclasse') != subclasse:
                continue
            if info.get('grupo'):
                grupos.add(info['grupo'])
        return sorted(list(grupos))

    def obter_data_mais_recente(self):
        """Busca a última data diretamente via SQL (instantâneo)"""
        if not os.path.exists(self.db_path):
            return None
            
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(data) FROM cotas")
            res = cursor.fetchone()
            if res and res[0]:
                return pd.to_datetime(res[0])
            return None
        except:
            return None
        finally:
            conn.close()
