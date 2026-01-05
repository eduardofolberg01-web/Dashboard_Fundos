import pandas as pd
import numpy as np
import os

class SFODataLoader:
    """
    Classe responsável pela ingestão, limpeza e mapeamento de dados 
    do Family Office (SFO).
    """
    def __init__(self, file_path='dados/Quant_Fundos.xlsm'):
        # Robustez contra caminho do arquivo
        if not os.path.exists(file_path):
            # Tenta encontrar na raiz se não estiver na pasta dados
            root_path = os.path.basename(file_path)
            if os.path.exists(root_path):
                file_path = root_path
            else:
                raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        self.file_path = file_path
        self.mapa_nomes = {}
        self.df_class = None
        self.df_cotas = None
        
        self._load_data()

    def _clean_string(self, val):
        """Remove espaços extras e quebras de linha de strings."""
        if pd.isna(val):
            return val
        if not isinstance(val, str):
            val = str(val)
        return val.strip().replace('\n', ' ')

    def _load_data(self):
        """Carrega e processa as abas do Excel."""
        # 1. Carregar Aba Classificação (Metadados)
        try:
            df_class = pd.read_excel(self.file_path, sheet_name='Classificação')
        except Exception as e:
            raise ValueError(f"Erro ao ler aba 'Classificação': {e}")

        # Limpar nomes de colunas da aba Classificação
        df_class.columns = [self._clean_string(col) for col in df_class.columns]
        
        # Limpar valores de texto nas colunas relevantes
        cols_to_clean = ['Codigo_Excel', 'Nome_Exibiçao', 'Tipo']
        for col in cols_to_clean:
            if col in df_class.columns:
                df_class[col] = df_class[col].apply(self._clean_string)
            
        self.df_class = df_class
        
        # 2. Carregar Aba Cotas (Série Temporal)
        try:
            # "nd" tratado como NaN conforme requisito
            df_cotas = pd.read_excel(self.file_path, sheet_name='Cotas', na_values=['nd'])
        except Exception as e:
            raise ValueError(f"Erro ao ler aba 'Cotas': {e}")
        
        # Limpar cabeçalhos da aba Cotas (Headers Sujos)
        raw_headers = df_cotas.columns.tolist()
        clean_headers = [self._clean_string(h) for h in raw_headers]
        df_cotas.columns = clean_headers
        
        # Configurar Index por Data
        if 'Data' in df_cotas.columns:
            # Forçar dayfirst=True para evitar confusão entre 02/01 (2 Jan) e 01/02 (1 Fev)
            df_cotas['Data'] = pd.to_datetime(df_cotas['Data'], dayfirst=True)
            df_cotas.set_index('Data', inplace=True)
            # CRÍTICO: Garantir ordenação cronológica (Ascendente)
            df_cotas.sort_index(ascending=True, inplace=True)
        
        # Garantir que todos os dados sejam numéricos após a limpeza do 'nd'
        for col in df_cotas.columns:
            df_cotas[col] = pd.to_numeric(df_cotas[col], errors='coerce')
            
        self.df_cotas = df_cotas
        
        # 3. Criar Mapeamento (Nome Amigável -> Coluna Suja)
        # O link é feito via substring do 'Codigo_Excel' na coluna da aba Cotas
        for _, row in df_class.iterrows():
            nome_amigavel = row.get('Nome_Exibiçao')
            codigo = row.get('Codigo_Excel')
            tipo = row.get('Tipo')
            classe = row.get('Classe')
            subclasse = row.get('Subclasse')
            grupo = row.get('Grupo')
            
            if pd.isna(nome_amigavel) or pd.isna(codigo):
                continue
                
            # Procurar o codigo (substring) nas colunas da aba Cotas
            match_col = None
            for col in self.df_cotas.columns:
                if str(codigo) in col:
                    match_col = col
                    break
            
            if match_col:
                self.mapa_nomes[nome_amigavel] = {
                    'coluna_original': match_col,
                    'tipo': tipo,
                    'classe': classe if not pd.isna(classe) else None,
                    'subclasse': subclasse if not pd.isna(subclasse) else None,
                    'grupo': grupo if not pd.isna(grupo) else None
                }

    def buscar_dados(self, lista_de_nomes):
        """
        Retorna um DataFrame com as colunas renomeadas para os nomes amigáveis.
        """
        if isinstance(lista_de_nomes, str):
            lista_de_nomes = [lista_de_nomes]
            
        colunas_para_extrair = []
        rename_dict = {}
        
        for nome in lista_de_nomes:
            if nome in self.mapa_nomes:
                original = self.mapa_nomes[nome]['coluna_original']
                colunas_para_extrair.append(original)
                rename_dict[original] = nome
            else:
                print(f"Aviso: Ativo '{nome}' não encontrado no mapeamento da base.")
                
        if not colunas_para_extrair:
            return pd.DataFrame(index=self.df_cotas.index)
            
        return self.df_cotas[colunas_para_extrair].rename(columns=rename_dict)

    def listar_ativos(self, tipo=None, classe=None, subclasse=None, grupo=None):
        """Lista nomes amigáveis disponíveis, opcionalmente filtrados por tipo, classe, subclasse e/ou grupo."""
        resultado = []
        
        for nome, info in self.mapa_nomes.items():
            # Filtro por tipo
            if tipo and str(info.get('tipo', '')).lower() != tipo.lower():
                continue
            
            # Filtro por classe
            if classe and info.get('classe') != classe:
                continue
            
            # Filtro por subclasse
            if subclasse and info.get('subclasse') != subclasse:
                continue
                
            # Filtro por grupo
            if grupo and info.get('grupo') != grupo:
                continue
            
            resultado.append(nome)
        
        return resultado
    
    def listar_classes(self, tipo=None):
        """Lista todas as classes disponíveis, opcionalmente filtradas por tipo."""
        classes = set()
        for nome, info in self.mapa_nomes.items():
            if tipo and str(info.get('tipo', '')).lower() != tipo.lower():
                continue
            if info.get('classe'):
                classes.add(info['classe'])
        return sorted(list(classes))
    
    def listar_subclasses(self, tipo=None, classe=None):
        """Lista todas as subclasses disponíveis, opcionalmente filtradas por tipo e/ou classe."""
        subclasses = set()
        for nome, info in self.mapa_nomes.items():
            if tipo and str(info.get('tipo', '')).lower() != tipo.lower():
                continue
            if classe and info.get('classe') != classe:
                continue
            if info.get('subclasse'):
                subclasses.add(info['subclasse'])
        return sorted(list(subclasses))
        
    def listar_grupos(self, tipo=None, classe=None, subclasse=None):
        """Lista todos os grupos disponíveis, opcionalmente filtrados."""
        grupos = set()
        for nome, info in self.mapa_nomes.items():
            if tipo and str(info.get('tipo', '')).lower() != tipo.lower():
                continue
            if classe and info.get('classe') != classe:
                continue
            if subclasse and info.get('subclasse') != subclasse:
                continue
            if info.get('grupo'):
                grupos.add(info['grupo'])
        return sorted(list(grupos))

