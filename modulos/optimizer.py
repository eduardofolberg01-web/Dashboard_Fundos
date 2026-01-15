import numpy as np
import pandas as pd
from scipy.optimize import minimize

class PortfolioOptimizer:
    """
    Motor de Otimização (Markowitz Mean-Variance).
    Otimizado para estabilidade numérica usando SLSQP.
    """
    
    def __init__(self, df_ret, risk_free_rate=0.0):
        # Assume que df_ret JÁ SÃO retornos diários (ex: 0.01 para 1%)
        self.df_ret = df_ret
        self.risk_free_rate = risk_free_rate
        
        # Anualização (252 dias úteis)
        self.mean_ret = df_ret.mean() * 252 
        self.cov_matrix = df_ret.cov() * 252 
        self.assets = df_ret.columns.tolist()
        self.n_assets = len(self.assets)

    def _portfolio_stats(self, weights):
        """Calcula Retorno, Volatilidade e Sharpe para um dado vetor de pesos."""
        weights = np.array(weights)
        
        # Retorno Esperado = Soma(Peso * Retorno Medio)
        ret = np.sum(self.mean_ret * weights)
        
        # Volatilidade = Raiz(w.T * Cov * w)
        var = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        vol = np.sqrt(var)
        
        # Sharpe Ratio
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        
        return ret, vol, sharpe

    def _format_result(self, result):
        """Formata a saída do scipy.optimize em um dicionário legível."""
        if result.success:
            ret, vol, sharpe = self._portfolio_stats(result.x)
            return {
                'Return': ret,
                'Volatility': vol,
                'Sharpe': sharpe,
                'Weights': dict(zip(self.assets, result.x)) # Pesos com nomes dos ativos
            }
        return None

    # --- OTIMIZADORES PÚBLICOS ---

    def otimizar_min_vol(self, bounds):
        """
        Encontra o portfólio de Mínima Volatilidade Global.
        """
        # Objetivo: Minimizar Volatilidade
        def fun_vol(weights):
            return self._portfolio_stats(weights)[1]

        # Restrição: Soma dos pesos = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Chute inicial: Pesos iguais (1/N)
        init_guess = [1.0 / self.n_assets] * self.n_assets
        
        result = minimize(fun_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return self._format_result(result)

    def otimizar_vol_alvo(self, target_vol, bounds):
        """
        Maximiza Retorno sujeito a Volatilidade <= Target.
        
        Correção Técnica: Usamos desigualdade ('ineq') em vez de igualdade ('eq').
        Isso significa: "Ache o maior retorno possível sem estourar esse risco",
        o que é muito mais fácil matematicamente do que "Ache exatamente esse risco".
        """
        # Objetivo: Minimizar Retorno Negativo (ou seja, Maximizar Retorno)
        def fun_neg_ret(weights):
            return -self._portfolio_stats(weights)[0]

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},      # Soma 1
            # Scipy ineq: fun(x) >= 0. Portanto: Target - Vol >= 0  => Vol <= Target
            {'type': 'ineq', 'fun': lambda x: target_vol - self._portfolio_stats(x)[1]} 
        )
        
        init_guess = [1.0 / self.n_assets] * self.n_assets
        
        result = minimize(fun_neg_ret, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return self._format_result(result)

    def otimizar_max_sharpe(self, bounds):
        """
        Encontra o portfólio de Máximo Sharpe Ratio.
        
        Maximiza (Retorno - Risk Free) / Volatilidade, ou seja,
        o portfólio com melhor relação risco-retorno.
        """
        # Objetivo: Minimizar Sharpe Negativo (Maximizar Sharpe)
        def fun_neg_sharpe(weights):
            ret, vol, sharpe = self._portfolio_stats(weights)
            return -sharpe  # Negativo para minimizar
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        init_guess = [1.0 / self.n_assets] * self.n_assets
        
        result = minimize(fun_neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        
        return self._format_result(result)

    def gerar_fronteira_pontos(self, n_points=20, bounds=None):
        """
        Gera a Fronteira Eficiente Iterando sobre RETORNOS.
        
        Por que iterar retornos e não volatilidade?
        Porque a função de risco é convexa. Para cada nível de retorno desejado,
        existe apenas uma carteira de risco mínimo. Isso garante que desenhamos
        a parte superior da bala (fronteira eficiente) e não a parte inferior.
        """
        # 1. Encontrar o Piso (Min Vol)
        min_vol_port = self.otimizar_min_vol(bounds)
        if not min_vol_port: return []

        # 2. Encontrar o Teto (Max Return Teórico)
        # O retorno máximo possível é simplesmente investir 100% no ativo de maior retorno
        # (respeitando os bounds, mas aqui simplificamos pegando o max return dos ativos)
        max_asset_ret = np.max(self.mean_ret)
        
        # Definir Grid de Retornos Alvo (do Min Vol até 99% do Máximo possível)
        ret_min = min_vol_port['Return']
        target_returns = np.linspace(ret_min, max_asset_ret * 0.99, n_points)
        
        portfolios = []
        
        # Adiciona o Min Vol como primeiro ponto
        portfolios.append(min_vol_port)
        
        # Para cada retorno alvo, minimiza o risco
        for ret_target in target_returns[1:]:
            res = self._otimizar_min_vol_para_retorno(ret_target, bounds)
            if res:
                portfolios.append(res)
        
        return portfolios

    def _otimizar_min_vol_para_retorno(self, target_ret, bounds):
        """
        Helper: Minimiza Volatilidade dado que Retorno >= Target
        """
        def fun_vol(weights):
            return self._portfolio_stats(weights)[1]

        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # Soma 1
            # Retorno >= Target => Retorno - Target >= 0
            {'type': 'ineq', 'fun': lambda x: self._portfolio_stats(x)[0] - target_ret} 
        )
        
        init_guess = [1.0 / self.n_assets] * self.n_assets
        
        result = minimize(fun_vol, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return self._format_result(result)