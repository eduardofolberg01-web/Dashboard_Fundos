import pandas as pd
import numpy as np
from datetime import datetime

class DashboardMetrics:
    """
    Módulo dedicado para cálculos do Painel Geral.
    Réplica da lógica validada do RiskEngine para garantir consistência.
    """
    
    def __init__(self):
        pass

    def _ajustar_escala(self, series):
        """
        Verifica se os dados estão em percentual (ex: 1.0 para 1%) 
        ou decimal (0.01 para 1%) e normaliza para decimal.
        """
        if series.empty: 
            return series
            
        # Heurística: se a média absoluta for > 0.1 (10% ao dia é improvável), divide por 100
        if series.abs().mean() > 0.1:
            return series / 100.0
        return series

    def calcular_ytd(self, series):
        """Calcula o retorno Year-to-Date (YTD). Retorna em percentual (ex: 5.0 para 5%)."""
        if series.empty:
            return np.nan
            
        series = self._ajustar_escala(series)
        current_year = series.index[-1].year
        
        # Filtrar apenas o ano atual
        series_ytd = series[series.index.year == current_year]
        
        if series_ytd.empty:
            return np.nan
            
        # (1 + r).prod() - 1, multiplicado por 100 para retornar em %
        return ((1 + series_ytd).prod() - 1) * 100.0

    def calcular_retorno_periodo_custom(self, series, meses=None):
        """
        Calcula retorno acumulado para N meses. Retorna em percentual (ex: 5.0 para 5%).
        Se meses=None, calcula desde o início (Inception).
        """
        if series.empty:
            return np.nan
            
        series = self._ajustar_escala(series)
        
        if meses is None:
            # Retorno Total, multiplicado por 100 para retornar em %
            return ((1 + series).prod() - 1) * 100.0
            
        days = int(meses * 21) # Janela de Dias Úteis
        
        # Se não tem dados suficientes para a janela cheia
        if len(series) < days:
             return np.nan # Retorna NaN para indicar falta de histórico
             
        series_window = series.tail(days)
        return ((1 + series_window).prod() - 1) * 100.0

    def calcular_percentual_cdi(self, retorno_fundo, retorno_cdi):
        """Calcula % do CDI. Retorna em percentual (ex: 120.0 para 120% do CDI)."""
        if pd.isna(retorno_fundo) or pd.isna(retorno_cdi) or retorno_cdi == 0:
            return np.nan
        # Ambos já vêm em escala percentual (ex: 5.0 = 5%), então apenas divide
        return (retorno_fundo / retorno_cdi) * 100.0

    def calcular_volatilidade_periodo(self, series, meses=None):
        """
        Calcula volatilidade anualizada para prazo específico. Retorna em percentual (ex: 15.0 para 15%).
        Método: std() * sqrt(252)
        """
        if series.empty:
            return np.nan
            
        series = self._ajustar_escala(series)
        
        if meses is not None:
            days = int(meses * 21)
            if len(series) < days:
                return np.nan
            series_window = series.tail(days)
        else:
            series_window = series
            
        # Volatilidade Anualizada, multiplicada por 100 para retornar em %
        val = series_window.std() * np.sqrt(252) * 100.0
        return val

    def calcular_drawdown_recuperacao(self, series):
        """
        Retorna Max Drawdown (em percentual, ex: -15.0 para -15%) e Tempo de Recuperação (em dias úteis).
        """
        if series.empty:
            return np.nan, np.nan
            
        series = self._ajustar_escala(series)
            
        # Wealth Index
        wealth = (1 + series).cumprod()
        peaks = wealth.cummax()
        drawdown = (wealth - peaks) / peaks
        
        min_dd = drawdown.min()
        
        # Se drawdown insignificante
        if min_dd > -0.0001: 
            return 0.0, 0
            
        idx_min_dd = drawdown.idxmin()
        
        # Recuperação
        peak_at_min = peaks.loc[idx_min_dd]
        series_after = wealth.loc[idx_min_dd:]
        
        # Filtra datas POSTERIORES ao vale onde wealth >= pico anterior
        recovered = series_after[series_after >= peak_at_min]
        
        # Precisamos encontrar a primeira data APÓS idx_min_dd
        recovered_after = recovered[recovered.index > idx_min_dd]
        
        if not recovered_after.empty:
             date_recovered = recovered_after.index[0]
             delta = date_recovered - idx_min_dd
             days_recovery = delta.days
        else:
            days_recovery = np.nan
            
        # Retorna drawdown em percentual (multiplicado por 100)
        return min_dd * 100.0, days_recovery

    def calcular_beta_periodo(self, asset_series, bench_series, meses):
        """Calcula Beta contra benchmark em janela específica."""
        if asset_series.empty or bench_series is None or bench_series.empty:
            return np.nan
            
        days = int(meses * 21)
        
        # Alinhar dados e ajustar escala
        # Importante: Garantir alinhamento de datas ANTES de cortar tail
        df = pd.concat([asset_series, bench_series], axis=1).dropna()
        
        if len(df) < days:
            return np.nan
            
        # Ajusta escala APÓS alinhar
        y_full = self._ajustar_escala(df.iloc[:, 0])
        x_full = self._ajustar_escala(df.iloc[:, 1])
            
        y = y_full.tail(days)
        x = x_full.tail(days)
        
        # Beta = Cov(x,y) / Var(x)
        # Método RiskEngine: Corr * (StdY/StdX)
        corr = y.corr(x)
        std_y = y.std()
        std_x = x.std()
        
        if std_x == 0:
            return np.nan
            
        beta = corr * (std_y / std_x)
        return beta

    def calcular_var_95(self, series):
        """Calcula VaR 95% Diário (Histórico). Retorna em percentual (ex: -2.5 para -2.5%)."""
        if series.empty:
            return np.nan
            
        series = self._ajustar_escala(series)
        
        # VaR Histórico 95% (Percentil 5), multiplicado por 100 para retornar em %
        var_95_daily = np.percentile(series, 5) * 100.0
        
        return var_95_daily
