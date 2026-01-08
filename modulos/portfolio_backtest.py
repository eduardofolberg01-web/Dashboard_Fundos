"""
Módulo de Backtesting de Portfólios com Rebalanceamento
Implementa múltiplas estratégias de otimização com simulação realista de drift
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==========================================
# 1. FUNÇÕES AUXILIARES
# ==========================================

def get_portfolio_metrics(weights, mean_returns, cov_matrix):
    """
    Calcula retorno e volatilidade anualizados de um portfólio.
    
    Args:
        weights: Array de pesos dos ativos
        mean_returns: Retornos médios diários
        cov_matrix: Matriz de covariância diária
    
    Returns:
        tuple: (retorno anualizado, volatilidade anualizada)
    """
    weights = np.array(weights)
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return ret, vol


def risk_contribution(weights, cov_matrix):
    """
    Calcula a contribuição de risco de cada ativo ao risco total do portfólio.
    RC_i = w_i * (Cov * w)_i / sigma_p
    """
    weights = np.array(weights)
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    portfolio_vol = np.sqrt(portfolio_variance)
    
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / (portfolio_vol + 1e-6)
    
    return risk_contrib


# ==========================================
# 2. ESTRATÉGIAS DE OTIMIZAÇÃO
# ==========================================

def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, bounds=None):
    """
    Estratégia 1: Máximo Sharpe Ratio
    Maximiza (Retorno - Risk Free) / Volatilidade
    """
    num_assets = len(mean_returns)
    
    def neg_sharpe_ratio(weights):
        p_ret, p_vol = get_portfolio_metrics(weights, mean_returns, cov_matrix)
        return -(p_ret - risk_free_rate) / (p_vol + 1e-6)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    result = minimize(
        neg_sharpe_ratio, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    return result.x if result.success else init_guess


def optimize_min_variance(mean_returns, cov_matrix, bounds=None):
    """
    Estratégia 2: Mínima Variância
    Minimiza a volatilidade do portfólio
    """
    num_assets = len(mean_returns)
    
    def portfolio_variance(weights):
        return get_portfolio_metrics(weights, mean_returns, cov_matrix)[1]
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    result = minimize(
        portfolio_variance, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    return result.x if result.success else init_guess


def optimize_max_return_fixed_risk(mean_returns, cov_matrix, target_vol, bounds=None):
    """
    Estratégia 3: Máximo Retorno para Risco Fixo
    Maximiza retorno sujeito a volatilidade <= target_vol
    """
    num_assets = len(mean_returns)
    
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Verifica se target_vol é atingível
    min_var_weights = optimize_min_variance(mean_returns, cov_matrix, bounds)
    min_vol = get_portfolio_metrics(min_var_weights, mean_returns, cov_matrix)[1]
    
    if target_vol < min_vol:
        target_vol = min_vol
    
    def neg_portfolio_return(weights):
        return -get_portfolio_metrics(weights, mean_returns, cov_matrix)[0]
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: target_vol - get_portfolio_metrics(x, mean_returns, cov_matrix)[1]}
    )
    
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    result = minimize(
        neg_portfolio_return, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    return result.x if result.success else min_var_weights


def optimize_risk_parity(mean_returns, cov_matrix, bounds=None):
    """
    Estratégia 4: Paridade de Risco (Risk Parity)
    Equaliza as contribuições de risco de cada ativo
    """
    num_assets = len(mean_returns)
    
    def risk_parity_objective(weights):
        risk_contrib = risk_contribution(weights, cov_matrix)
        target_risk_contrib = 1.0 / num_assets
        risk_contrib_normalized = risk_contrib / (np.sum(risk_contrib) + 1e-6)
        return np.sum((risk_contrib_normalized - target_risk_contrib) ** 2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Initial guess usando inverse volatility
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vol = 1.0 / (vols + 1e-6)
    init_guess = inv_vol / np.sum(inv_vol)
    
    result = minimize(
        risk_parity_objective, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    return result.x if result.success else init_guess


def optimize_equal_weight(mean_returns, cov_matrix, bounds=None):
    """
    Estratégia 5: Pesos Iguais (Benchmark)
    Distribui igualmente entre os ativos
    """
    num_assets = len(mean_returns)
    return np.array([1.0 / num_assets] * num_assets)


# ==========================================
# 3. FUNÇÃO UNIFICADA DE OTIMIZAÇÃO
# ==========================================

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate, strategy='max_sharpe', target_vol=None, bounds=None):
    """
    Função unificada para otimização de portfólio.
    
    Args:
        mean_returns: Retornos médios diários
        cov_matrix: Matriz de covariância diária
        risk_free_rate: Taxa livre de risco anualizada
        strategy: Estratégia de otimização
        target_vol: Volatilidade alvo (apenas para 'max_return_fixed_risk')
        bounds: Restrições de peso para cada ativo
    
    Returns:
        np.array: Pesos otimizados
    """
    if strategy == 'max_sharpe':
        return optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate, bounds)
    elif strategy == 'min_variance':
        return optimize_min_variance(mean_returns, cov_matrix, bounds)
    elif strategy == 'max_return_fixed_risk':
        if target_vol is None:
            target_vol = 0.15
        return optimize_max_return_fixed_risk(mean_returns, cov_matrix, target_vol, bounds)
    elif strategy == 'risk_parity':
        return optimize_risk_parity(mean_returns, cov_matrix, bounds)
    elif strategy == 'equal_weight':
        return optimize_equal_weight(mean_returns, cov_matrix, bounds)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")


# ==========================================
# 4. ENGINE DE BACKTEST
# ==========================================

class PortfolioBacktester:
    """
    Engine de backtesting com rebalanceamento e drift realista.
    """
    
    def __init__(self, df_returns, df_cdi, assets):
        """
        Args:
            df_returns: DataFrame com retornos diários dos ativos
            df_cdi: Series com retornos diários do CDI
            assets: Lista de nomes dos ativos
        """
        self.df_returns = df_returns[assets].copy()
        self.df_cdi = df_cdi.copy()
        self.assets = assets
        self.rebalance_schedule = {}
        self.results = None
    
    def run(self, lookback_days, rebalance_freq, strategy, target_vol=None, bounds=None):
        """
        Executa o backtest completo.
        
        Args:
            lookback_days: Dias de histórico para calibração
            rebalance_freq: Frequência de rebalanceamento ('Mensal', 'Trimestral', 'Semestral', 'Anual')
            strategy: Estratégia de otimização
            target_vol: Volatilidade alvo (opcional)
            bounds: Restrições de peso
        
        Returns:
            dict: Resultados do backtest
        """
        # Gera datas de rebalanceamento
        freq_map = {
            'Mensal': 'M',
            'Trimestral': 'Q',
            'Semestral': '6M',
            'Anual': 'Y'
        }
        
        rebalance_dates = self.df_returns.resample(freq_map[rebalance_freq]).last().index
        
        # Otimiza em cada data de rebalanceamento
        for date in rebalance_dates:
            start_train = date - pd.Timedelta(days=lookback_days)
            
            if start_train < self.df_returns.index[0]:
                continue
            
            train_window = self.df_returns.loc[start_train:date]
            cdi_window = self.df_cdi.loc[start_train:date]
            
            if len(train_window) < 20:
                continue
            
            mu = train_window.mean()
            sigma = train_window.cov()
            rf = cdi_window.mean() * 252
            
            try:
                opt_weights = optimize_portfolio(mu, sigma, rf, strategy=strategy, target_vol=target_vol, bounds=bounds)
                self.rebalance_schedule[date] = opt_weights
            except Exception as e:
                print(f"Erro na otimização em {date}: {e}")
                continue
        
        if not self.rebalance_schedule:
            return None
        
        # Simula portfólio com drift A PARTIR DA PRIMEIRA DATA DE REBALANCEAMENTO
        # Garante coerência: só começa quando há dados suficientes para otimização
        start_sim_date = sorted(self.rebalance_schedule.keys())[0]
        
        sim_data = self.df_returns.loc[start_sim_date:]
        cdi_sim_data = self.df_cdi.loc[start_sim_date:]
        
        portfolio_values = []
        daily_weights = []
        weights_current = self.rebalance_schedule[start_sim_date].copy()
        portfolio_value = 1.0
        
        for date, row in sim_data.iterrows():
            # Rebalanceamento
            if date in self.rebalance_schedule:
                weights_current = self.rebalance_schedule[date].copy()
            
            # Aplica retornos
            returns_today = row.values
            weights_after_returns = weights_current * (1 + returns_today)
            
            # Atualiza valor
            new_portfolio_value = portfolio_value * np.sum(weights_after_returns)
            portfolio_values.append(new_portfolio_value)
            
            # Drift natural
            weights_current = weights_after_returns / np.sum(weights_after_returns)
            daily_weights.append(weights_current.copy())
            
            portfolio_value = new_portfolio_value
        
        # Calcula métricas
        portfolio_series = pd.Series(portfolio_values, index=sim_data.index)
        portfolio_returns = portfolio_series.pct_change().dropna()
        
        cum_ret_portfolio = portfolio_series / portfolio_series.iloc[0]
        cum_cdi = (1 + cdi_sim_data).cumprod()
        
        # Métricas
        total_return = cum_ret_portfolio.iloc[-1] - 1
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        
        cdi_total_return = cum_cdi.iloc[-1] - 1
        cdi_ann_return = cdi_sim_data.mean() * 252
        
        sharpe = (ann_return - cdi_ann_return) / ann_vol if ann_vol > 0 else 0
        
        running_max = cum_ret_portfolio.cummax()
        drawdown = (cum_ret_portfolio - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Rolling metrics
        rolling_window = 252
        portfolio_rolling = portfolio_returns.rolling(window=rolling_window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        cdi_rolling = cdi_sim_data.rolling(window=rolling_window).apply(
            lambda x: (1 + x).prod() - 1, raw=False
        )
        rolling_vs_cdi = (portfolio_rolling / cdi_rolling) * 100
        
        # Armazena resultados
        self.results = {
            'portfolio_series': portfolio_series,
            'portfolio_returns': portfolio_returns,
            'cum_ret_portfolio': cum_ret_portfolio,
            'cum_cdi': cum_cdi,
            'drawdown': drawdown,
            'daily_weights': pd.DataFrame(daily_weights, index=sim_data.index, columns=self.assets),
            'rebalance_dates': list(self.rebalance_schedule.keys()),
            'rebalance_weights': pd.DataFrame.from_dict(
                self.rebalance_schedule, orient='index', columns=self.assets
            ),
            'start_date': start_sim_date,  # Data de início da simulação
            'metrics': {
                'total_return': total_return,
                'ann_return': ann_return,
                'ann_vol': ann_vol,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'cdi_total_return': cdi_total_return,
                'cdi_ann_return': cdi_ann_return
            },
            'rolling': {
                'portfolio': portfolio_rolling,
                'cdi': cdi_rolling,
                'vs_cdi': rolling_vs_cdi,
                'avg_pct_cdi': rolling_vs_cdi.mean()
            }
        }
        
        return self.results
