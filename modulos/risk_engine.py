
import pandas as pd
import numpy as np
import statsmodels.api as sm

class RiskEngine:
    """
    Módulo responsável por cálculos vetorizados de risco e performance 
    para séries temporais financeiras (Retornos Diários).
    """
    
    def __init__(self):
        # Mapeamento de janelas em dias úteis (Assumindo 21 dias/mês)
        self.windows = {
            '6m': 126,
            '12m': 252,
            '24m': 504,
            '36m': 756
        }
    
    def calcular_retornos_periodo(self, series):
        """
        Calcula retorno acumulado para janelas específicas:
        6m, 12m, 24m, 36m e Inception.
        """
        if series.empty:
            return {}
        
        # Remover NaNs iniciais/finais para garantir cálculo correto
        series = series.dropna()
        
        # Converter para Wealth Index para facilitar cálculos de período
        # Assume series são retornos diários (ex: 0.01 para 1%)
        wealth_index = (1 + series).cumprod()
        
        # Retorno Inception
        # (Valor Final / Valor Inicial) - 1
        # Como wealth_index é acumulado dos retornos, o último valor é (1+R_total)
        # desde que o primeiro valor da série seja o primeiro retorno.
        # Precisamos normalizar: se series começa em t0, wealth começa em t0.
        # Retorno Total = wealth_index[-1] - 1
        try:
            ret_inception = wealth_index.iloc[-1] - 1
        except IndexError:
            ret_inception = np.nan

        results = {
            'Inception': ret_inception
        }
        
        # Data final para referência (último dado disponível)
        if hasattr(series.index, 'max'):
             # Não usamos datas de calendário aqui, usamos dias úteis (inteiros)
             # pois a série já deve estar indexada ou ordenada.
             # Vamos usar "lookback" por iloc.
             pass
        
        n_days = len(series)
        
        for label, window_days in self.windows.items():
            if n_days >= window_days:
                # Retorno janela = Wealth[-1] / Wealth[-window-1] - 1
                # Exemplo: janela 1 dia. Wealth[-1] / Wealth[-2] - 1. 2 dias: Weal[-1]/Weal[-3]
                # Se window_days > len, não calcula.
                
                # Para evitar problemas com index, calculamos o produtório do slice
                # (1+r).prod() - 1
                try:
                    window_return = (1 + series.tail(window_days)).prod() - 1
                    results[label] = window_return
                except:
                    results[label] = np.nan
            else:
                results[label] = np.nan
                
        return results

    def calcular_rolling_stats(self, series, benchmark_series=None):
        """
        Calcula Volatilidade e Beta (se benchmark for fornecido) em janelas móveis
        de 1m(21d), 3m(63d), 6m(126d), 12m(252d).
        """
        if series.empty:
            return pd.DataFrame()

        # Se benchmark fornecido, alinhar as séries primeiro
        if benchmark_series is not None:
            # Criar DataFrame temporário para garantir alinhamento
            df_temp = pd.DataFrame({
                'asset': series,
                'benchmark': benchmark_series
            }).dropna()
            
            series_aligned = df_temp['asset']
            benchmark_aligned = df_temp['benchmark']
        else:
            series_aligned = series.dropna()
            benchmark_aligned = None

        # Definição das janelas móveis
        rolling_windows = {
            '1m': 21,
            '3m': 63,
            '6m': 126,
            '12m': 252
        }
        
        stats_dict = {}
        
        for label, window in rolling_windows.items():
            # Min periods = metade da janela para evitar dados muito sujos no começo, 
            # mas permitir cálculo se tiver dados razoáveis.
            min_p = max(5, window // 2)
            
            # --- Volatilidade Anualizada ---
            # Std Dev * Sqrt(252)
            vol_col_name = f'Vol_{label}'
            vol_rolling = series_aligned.rolling(window=window, min_periods=min_p).std() * np.sqrt(252)
            stats_dict[vol_col_name] = vol_rolling
            
            # --- Beta (se benchmark existe) ---
            if benchmark_aligned is not None:
                beta_col_name = f'Beta_{label}'
                
                # MÉTODO CORRIGIDO: Usar correlação * (std_asset / std_bench)
                # Beta = Corr(Asset, Bench) * (Vol_Asset / Vol_Bench)
                # Isso é matematicamente equivalente a Cov/Var mas mais estável numericamente
                
                corr_rolling = series_aligned.rolling(window=window, min_periods=min_p).corr(benchmark_aligned)
                std_asset = series_aligned.rolling(window=window, min_periods=min_p).std()
                std_bench = benchmark_aligned.rolling(window=window, min_periods=min_p).std()
                
                # Beta = Correlação * (Volatilidade Asset / Volatilidade Benchmark)
                # Proteção contra divisão por zero
                beta_rolling = corr_rolling * (std_asset / std_bench.replace(0, np.nan))
                
                stats_dict[beta_col_name] = beta_rolling
        
        df_stats = pd.DataFrame(stats_dict, index=series_aligned.index)
        return df_stats

    def calcular_drawdown(self, series):
        """
        Calcula a série de Drawdown e o Máximo Drawdown.
        Retornos: (Series de Drawdown, Valor Max Drawdown)
        """
        if series.empty:
            return pd.Series(), 0.0
            
        # Wealth Index
        compounded = (1 + series).cumprod()
        
        # Topo Histórico Acumulado
        peaks = compounded.cummax()
        
        # Drawdown: (Valor Atual - Pico) / Pico
        drawdown = (compounded - peaks) / peaks
        
        max_drawdown = drawdown.min() # Valor negativo mais baixo
        
        return drawdown, max_drawdown

    def calcular_metricas_risco_retorno(self, series, rf=0.0):
        """
        Calcula Sharpe, Sortino e Calmar Ratios.
        rf: Risk Free rate anual (padrão 0.0 para simplificação se não fornecido)
        """
        if series.empty:
            return {}
            
        series = series.dropna()
        n_days = len(series)
        if n_days < 5: # Precaução
            return {'Sharpe': np.nan, 'Sortino': np.nan, 'Calmar': np.nan}
        
        # Anualização
        ann_factor = 252
        
        # --- Sharpe Ratio ---
        # (Retorno Médio Anual - Rf) / Vol Anual
        avg_return_ann = series.mean() * ann_factor
        vol_ann = series.std() * np.sqrt(ann_factor)
        
        if vol_ann > 0:
            sharpe = (avg_return_ann - rf) / vol_ann
        else:
            sharpe = np.nan
            
        # --- Sortino Ratio ---
        # Considera apenas volatilidade negativa (Downside Risk)
        # Retorno alvo para downside é geralmente 0 ou Rf. Usaremos 0 aqui.
        negative_returns = series[series < 0]
        # Desvio padrão dos retornos negativos (ajustado para anual)
        # Existem várias formas, a mais simples é std dos retornos negativos * sqrt(252)
        # Outra é sqrt(mean(min(0, r)^2)) * sqrt(252). Vamos usar std dos negativos para simplificar,
        # mas a definição estrita usa Lower Partial Moments.
        # Vamos usar LPM de ordem 2 (raiz quadrada da média dos quadrados dos retornos negativos).
        downside_sq_sum = (np.minimum(series, 0) ** 2).mean()
        downside_dev = np.sqrt(downside_sq_sum) * np.sqrt(ann_factor)
        
        if downside_dev > 0:
            sortino = (avg_return_ann - rf) / downside_dev
        else:
            sortino = np.nan
            
        # --- Calmar Ratio ---
        # Retorno Anualizado Composto / Abs(Max Drawdown)
        # Retorno Composto Anual (CAGR)
        total_ret = (1 + series).prod()
        # n_years = n_days / 252
        if n_days > 0 and total_ret > 0:
            cagr = total_ret ** (ann_factor / n_days) - 1
        else:
            # Fallback para média aritmética se série for ruim ou curta
            cagr = avg_return_ann
            
        _, max_dd = self.calcular_drawdown(series)
        abs_max_dd = abs(max_dd)
        
        if abs_max_dd > 0:
            calmar = cagr / abs_max_dd
        else:
            calmar = np.nan # Sem drawdown?
            
        return {
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Calmar': calmar,
            'Volatilidade_Ano': vol_ann,
            'Retorno_Ano_Medio': avg_return_ann
        }

    def _executar_otimizacao_rbsa(self, y, X, constrained=True):
        """
        Método auxiliar que executa a otimização RBSA em arrays numpy.
        Retorna: weights (array), alpha (float), r2 (float)
        constrained: Se True, soma <= 1 e 0 <= w <= 1. Se False, soma livre e w >= 0.
        """
        from scipy.optimize import minimize
        
        n_factors = X.shape[1]
        
        # Centrar variáveis para remover Alpha da otimização dos pesos
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        
        X_centered = X - X_mean
        y_centered = y - y_mean
        
        # Função de Custo (Residual Sum of Squares)
        def objective_function(weights):
            residuals = y_centered - (X_centered @ weights)
            return np.sum(residuals**2)
        
        constraints = []
        
        if constrained:
            # Soma deve ser MENOR ou IGUAL a 1 (1 - soma >= 0)
            constraints = ({'type': 'ineq', 'fun': lambda w: 1 - np.sum(w)})
            # Limites (Bounds): 0 <= w <= 1
            bounds = tuple((0, 1) for _ in range(n_factors))
        else:
            # Sem restrição de soma (Alavancagem permitida)
            # Apenas No-Shorting (w >= 0)
            # Upper bound None permite Beta > 1
            bounds = tuple((0, None) for _ in range(n_factors))

        # Chute inicial: divisão igualitária
        initial_guess = np.array([1/(n_factors+1)] * n_factors)
        
        try:
            result = minimize(
                objective_function, 
                initial_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                tol=1e-8
            )
            
            weights = result.x
            
            # Recuperar Alpha (Intercepto)
            alpha = y_mean - np.dot(weights, X_mean)
            
            # Calcular R-Squared
            fitted_values = alpha + (X @ weights)
            ss_res = np.sum((y - fitted_values) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot == 0:
                 r2 = 0.0
            else:
                r2 = 1 - (ss_res / ss_tot)
                
            return weights, alpha, r2
            
        except Exception:
            # Em caso de erro na otimização, retorna nulos
            return np.zeros(n_factors), np.nan, np.nan

    def analise_estilo_rbsa(self, fundo_series, benchmarks_df, constrained=True):
        """
        Return Based Style Analysis (RBSA) - Regressão com Restrições (Pontual).
        Calcula a alocação de estilo para todo o período fornecido.
        constrained: Se True, limita soma <= 100%. Se False, permite alavancagem.
        """
        # Alinhar dados (Inner Join nas datas)
        data = pd.concat([fundo_series, benchmarks_df], axis=1, join='inner').dropna()
        
        if data.empty:
            return {}, np.nan, pd.Series()
        
        y = data.iloc[:, 0].values # Retornos do Fundo
        X = data.iloc[:, 1:].values # Retornos dos Benchmarks
        cols = data.columns[1:] # Nomes dos Benchmarks
        
        # Executar otimização usando o método auxiliar
        weights, alpha, r2 = self._executar_otimizacao_rbsa(y, X, constrained=constrained)
        
        if np.isnan(r2):
            return {}, np.nan, pd.Series()

        # Recalcular fitted series para retorno
        # (Poderíamos retornar do helper, mas o helper foca em escalares/arrays p/ performance)
        fitted_values = alpha + (X @ weights)
        fitted_series = pd.Series(fitted_values, index=data.index)
        
        # Montar dicionário de coeficientes
        coefs = dict(zip(cols, weights))
        
        # Adicionar o Resíduo/Caixa no retorno pontual também
        if constrained:
            coefs['Residual_Cash'] = 1.0 - np.sum(weights)
        else:
            # Se não é constrained, não temos "sobra de caixa" para 100%, 
            # na verdade temos alavancagem (se soma > 1).
            # Podemos não reportar Residual_Cash ou reportar como 0 para não quebrar charts.
            # Ou melhor: reportar 'Soma_Exposicao' para indicar alavancagem?
            # Para compatibilidade com o gráfico de pizza/barra atual, vamos usar Residual_Cash = 0 ou negativo?
            # Melhor 0. A "Alavancagem" será vista pela soma das barras.
            coefs['Residual_Cash'] = 0.0
            
        coefs['const'] = alpha
        
        return coefs, r2, fitted_series


    def calcular_rolling_rbsa(self, fundo_series, benchmarks_df, window=126, step=21):
        """
        Calcula a evolução histórica da Análise de Estilo (Rolling RBSA).
        """
        # Alinhar dados
        data = pd.concat([fundo_series, benchmarks_df], axis=1, join='inner').dropna()
        
        if len(data) < window:
            return pd.DataFrame()
            
        y_full = data.iloc[:, 0].values
        X_full = data.iloc[:, 1:].values
        cols = data.columns[1:]
        dates = data.index
        
        results = []
        result_dates = []
        
        # Loop Rolling
        for i in range(window, len(data), step):
            # Recorte da janela (janela termina em i, pega os 'window' anteriores)
            start_idx = i - window
            end_idx = i
            
            y_window = y_full[start_idx:end_idx]
            X_window = X_full[start_idx:end_idx]
            date_ref = dates[end_idx-1] # Data do final da janela
            
            # Otimização
            weights, alpha, r2 = self._executar_otimizacao_rbsa(y_window, X_window)
            
            if not np.isnan(r2):
                row = dict(zip(cols, weights))
                row['Alpha'] = alpha # Alpha diário
                row['R2'] = r2
                
                # ADICIONADO: Resíduo de Alocação (Caixa / Não explicado pelos betas)
                row['Residual_Cash'] = 1.0 - np.sum(weights)
                
                results.append(row)
                result_dates.append(date_ref)
        
        if not results:
            return pd.DataFrame()
            
        df_rolling = pd.DataFrame(results, index=result_dates)
        return df_rolling
