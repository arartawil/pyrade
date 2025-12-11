"""
CEC 2022 Test Function Suite for Single Objective Bound Constrained Numerical Optimization

Reference:
    CEC 2022 Competition on Single Objective Bound Constrained Numerical Optimization
    Test functions are defined for D=2, 10, 20 dimensions
    
Functions:
    F1: Zakharov Function (unimodal)
    F2: Rosenbrock's Function (unimodal)
    F3: Schaffer's F7 Function (multimodal)
    F4: Step Rastrigin Function (multimodal)
    F5: Levy Function (multimodal)
    F6: Hybrid Function 1 (3 functions)
    F7: Hybrid Function 2 (6 functions)
    F8: Hybrid Function 3 (5 functions)
    F9: Composition Function 1 (5 functions)
    F10: Composition Function 2 (3 functions)
    F11: Composition Function 3 (5 functions)
    F12: Composition Function 4 (6 functions)
"""

import numpy as np
import os
from typing import Union, Callable

# Constants
INF = 1.0e99
EPS = 1.0e-14
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029


class CEC2022:
    """
    CEC 2022 Benchmark Suite
    
    Parameters
    ----------
    func_num : int
        Function number (1-12)
    dim : int
        Dimension (2, 10, or 20)
    data_dir : str, optional
        Directory containing input data files
    """
    
    def __init__(self, func_num: int, dim: int, data_dir: str = None):
        if func_num < 1 or func_num > 12:
            raise ValueError(f"Function number must be 1-12, got {func_num}")
        
        if dim not in [2, 10, 20]:
            raise ValueError(f"Dimension must be 2, 10, or 20, got {dim}")
        
        if dim == 2 and func_num in [6, 7, 8]:
            raise ValueError(f"Function {func_num} is not defined for D=2")
        
        self.func_num = func_num
        self.dim = dim
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'cec2022_data')
        
        # Load data
        self.OShift = None
        self.M = None
        self.SS = None
        self._load_data()
        
        # Bias values
        self.bias = [300, 400, 600, 800, 900, 1800, 2000, 2200, 2300, 2400, 2600, 2700]
    
    def _load_data(self):
        """Load shift data, rotation matrix, and shuffle data"""
        # Load rotation matrix
        M_file = os.path.join(self.data_dir, f'M_{self.func_num}_D{self.dim}.txt')
        if not os.path.exists(M_file):
            raise FileNotFoundError(f"Cannot find {M_file}")
        
        if self.func_num < 9:
            self.M = np.loadtxt(M_file).reshape(self.dim, self.dim)
        else:
            # Composition functions have multiple matrices
            cf_num = 12 if self.func_num >= 9 else 1
            self.M = np.loadtxt(M_file).reshape(cf_num, self.dim, self.dim)
        
        # Load shift data
        shift_file = os.path.join(self.data_dir, f'shift_data_{self.func_num}.txt')
        if not os.path.exists(shift_file):
            raise FileNotFoundError(f"Cannot find {shift_file}")
        
        if self.func_num < 9:
            self.OShift = np.loadtxt(shift_file)[:self.dim]
        else:
            # Composition functions have multiple shifts
            cf_num = 12
            data = np.loadtxt(shift_file)
            self.OShift = data.reshape(cf_num, self.dim)
        
        # Load shuffle data for hybrid functions
        if self.func_num in [6, 7, 8]:
            shuffle_file = os.path.join(self.data_dir, f'shuffle_data_{self.func_num}_D{self.dim}.txt')
            if os.path.exists(shuffle_file):
                self.SS = np.loadtxt(shuffle_file, dtype=int) - 1  # Convert to 0-indexed
    
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate function at point x
        
        Parameters
        ----------
        x : np.ndarray
            Input vector (1D array of length dim)
            
        Returns
        -------
        float
            Function value
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        results = []
        for xi in x:
            if len(xi) != self.dim:
                raise ValueError(f"Input dimension {len(xi)} does not match expected {self.dim}")
            
            result = self._evaluate(xi)
            results.append(result)
        
        return results[0] if len(results) == 1 else np.array(results)
    
    def _evaluate(self, x: np.ndarray) -> float:
        """Evaluate single point"""
        func_map = {
            1: lambda x: self._zakharov(x) + self.bias[0],
            2: lambda x: self._rosenbrock(x) + self.bias[1],
            3: lambda x: self._schaffer_f7(x) + self.bias[2],
            4: lambda x: self._step_rastrigin(x) + self.bias[3],
            5: lambda x: self._levy(x) + self.bias[4],
            6: lambda x: self._hf02(x) + self.bias[5],
            7: lambda x: self._hf10(x) + self.bias[6],
            8: lambda x: self._hf06(x) + self.bias[7],
            9: lambda x: self._cf01(x) + self.bias[8],
            10: lambda x: self._cf02(x) + self.bias[9],
            11: lambda x: self._cf06(x) + self.bias[10],
            12: lambda x: self._cf07(x) + self.bias[11],
        }
        
        return func_map[self.func_num](x)
    
    # ==================== Basic Functions ====================
    
    def _shift_rotate(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray, 
                      sh_rate: float = 1.0, s_flag: bool = True, r_flag: bool = True) -> np.ndarray:
        """Shift and rotate transformation"""
        if s_flag:
            sr_x = (x - Os) * sh_rate
        else:
            sr_x = x * sh_rate
        
        if r_flag:
            sr_x = Mr @ sr_x
        
        return sr_x
    
    def _zakharov(self, x: np.ndarray) -> float:
        """Zakharov function"""
        z = self._shift_rotate(x, self.OShift, self.M, 1.0, True, True)
        sum1 = np.sum(z ** 2)
        sum2 = np.sum(0.5 * np.arange(1, self.dim + 1) * z)
        return sum1 + sum2 ** 2 + sum2 ** 4
    
    def _rosenbrock(self, x: np.ndarray) -> float:
        """Rosenbrock's function"""
        z = self._shift_rotate(x, self.OShift, self.M, 2.048/100.0, True, True)
        z = z + 1.0  # Shift to origin
        result = 0.0
        for i in range(self.dim - 1):
            result += 100.0 * (z[i]**2 - z[i+1])**2 + (z[i] - 1.0)**2
        return result
    
    def _schaffer_f7(self, x: np.ndarray) -> float:
        """Schaffer's F7 function"""
        y = self._shift_rotate(x, self.OShift, self.M, 1.0, True, True)
        result = 0.0
        for i in range(self.dim - 1):
            si = np.sqrt(y[i]**2 + y[i+1]**2)
            result += np.sqrt(si) + np.sqrt(si) * np.sin(50.0 * si**0.2)**2
        return (result / (self.dim - 1)) ** 2
    
    def _step_rastrigin(self, x: np.ndarray) -> float:
        """Step Rastrigin function"""
        y = x.copy()
        for i in range(self.dim):
            if abs(y[i] - self.OShift[i]) > 0.5:
                y[i] = self.OShift[i] + np.floor(2 * (y[i] - self.OShift[i]) + 0.5) / 2
        
        z = self._shift_rotate(y, self.OShift, self.M, 5.12/100.0, True, True)
        return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)
    
    def _levy(self, x: np.ndarray) -> float:
        """Levy function"""
        z = self._shift_rotate(x, self.OShift, self.M, 1.0, True, True)
        w = 1.0 + (z - 0.0) / 4.0
        
        term1 = np.sin(PI * w[0]) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * PI * w[-1]) ** 2)
        
        sum_term = 0.0
        for i in range(self.dim - 1):
            sum_term += (w[i] - 1) ** 2 * (1 + 10 * np.sin(PI * w[i] + 1) ** 2)
        
        return term1 + sum_term + term3
    
    def _ellips(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray, 
                s_flag: bool = True, r_flag: bool = True) -> float:
        """Ellipsoidal function"""
        z = self._shift_rotate(x, Os, Mr, 1.0, s_flag, r_flag)
        result = 0.0
        for i in range(self.dim):
            result += (10.0 ** (6.0 * i / (self.dim - 1))) * z[i] ** 2
        return result
    
    def _bent_cigar(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                    s_flag: bool = True, r_flag: bool = True) -> float:
        """Bent Cigar function"""
        z = self._shift_rotate(x, Os, Mr, 1.0, s_flag, r_flag)
        return z[0] ** 2 + 1e6 * np.sum(z[1:] ** 2)
    
    def _discus(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                s_flag: bool = True, r_flag: bool = True) -> float:
        """Discus function"""
        z = self._shift_rotate(x, Os, Mr, 1.0, s_flag, r_flag)
        return 1e6 * z[0] ** 2 + np.sum(z[1:] ** 2)
    
    def _rastrigin(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                   s_flag: bool = True, r_flag: bool = True) -> float:
        """Rastrigin function"""
        z = self._shift_rotate(x, Os, Mr, 5.12/100.0, s_flag, r_flag)
        return np.sum(z**2 - 10.0 * np.cos(2.0 * PI * z) + 10.0)
    
    def _hgbat(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
               s_flag: bool = True, r_flag: bool = True) -> float:
        """HGBat function"""
        z = self._shift_rotate(x, Os, Mr, 5.0/100.0, s_flag, r_flag) - 1.0
        r2 = np.sum(z ** 2)
        sum_z = np.sum(z)
        return abs(r2**2 - sum_z**2) ** 0.25 + (0.5 * r2 + sum_z) / self.dim + 0.5
    
    def _katsuura(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                  s_flag: bool = True, r_flag: bool = True) -> float:
        """Katsuura function"""
        z = self._shift_rotate(x, Os, Mr, 5.0/100.0, s_flag, r_flag)
        result = 1.0
        tmp3 = self.dim ** 1.2
        
        for i in range(self.dim):
            temp = 0.0
            for j in range(1, 33):
                tmp1 = 2.0 ** j
                tmp2 = tmp1 * z[i]
                temp += abs(tmp2 - np.floor(tmp2 + 0.5)) / tmp1
            result *= (1.0 + (i + 1) * temp) ** (10.0 / tmp3)
        
        return result * (10.0 / self.dim / self.dim) - 10.0 / self.dim / self.dim
    
    def _ackley(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                s_flag: bool = True, r_flag: bool = True) -> float:
        """Ackley function"""
        z = self._shift_rotate(x, Os, Mr, 1.0, s_flag, r_flag)
        sum1 = np.sum(z ** 2)
        sum2 = np.sum(np.cos(2.0 * PI * z))
        return E - 20.0 * np.exp(-0.2 * np.sqrt(sum1 / self.dim)) - np.exp(sum2 / self.dim) + 20.0
    
    def _schwefel(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                  s_flag: bool = True, r_flag: bool = True) -> float:
        """Schwefel function"""
        z = self._shift_rotate(x, Os, Mr, 1000.0/100.0, s_flag, r_flag)
        z = z + 4.209687462275036e+002
        
        result = 0.0
        for zi in z:
            if zi > 500:
                result -= (500.0 - (zi % 500)) * np.sin(np.sqrt(abs(500.0 - (zi % 500))))
                result += ((zi - 500.0) / 100) ** 2 / self.dim
            elif zi < -500:
                result -= (-500.0 + (abs(zi) % 500)) * np.sin(np.sqrt(abs(500.0 - (abs(zi) % 500))))
                result += ((zi + 500.0) / 100) ** 2 / self.dim
            else:
                result -= zi * np.sin(np.sqrt(abs(zi)))
        
        return result + 4.189828872724338e+002 * self.dim
    
    def _happycat(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                  s_flag: bool = True, r_flag: bool = True) -> float:
        """HappyCat function"""
        z = self._shift_rotate(x, Os, Mr, 5.0/100.0, s_flag, r_flag) - 1.0
        r2 = np.sum(z ** 2)
        sum_z = np.sum(z)
        return abs(r2 - self.dim) ** 0.25 + (0.5 * r2 + sum_z) / self.dim + 0.5
    
    def _grie_rosen(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                    s_flag: bool = True, r_flag: bool = True) -> float:
        """Griewank-Rosenbrock function"""
        z = self._shift_rotate(x, Os, Mr, 5.0/100.0, s_flag, r_flag) + 1.0
        result = 0.0
        
        for i in range(self.dim - 1):
            temp = 100.0 * (z[i]**2 - z[i+1])**2 + (z[i] - 1.0)**2
            result += temp**2 / 4000.0 - np.cos(temp) + 1.0
        
        # Wrap-around term
        temp = 100.0 * (z[-1]**2 - z[0])**2 + (z[-1] - 1.0)**2
        result += temp**2 / 4000.0 - np.cos(temp) + 1.0
        
        return result
    
    def _escaffer6(self, x: np.ndarray, Os: np.ndarray, Mr: np.ndarray,
                   s_flag: bool = True, r_flag: bool = True) -> float:
        """Expanded Scaffer's F6 function"""
        z = self._shift_rotate(x, Os, Mr, 1.0, s_flag, r_flag)
        result = 0.0
        
        for i in range(self.dim - 1):
            temp1 = np.sin(np.sqrt(z[i]**2 + z[i+1]**2)) ** 2
            temp2 = 1.0 + 0.001 * (z[i]**2 + z[i+1]**2)
            result += 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        
        # Wrap-around term
        temp1 = np.sin(np.sqrt(z[-1]**2 + z[0]**2)) ** 2
        temp2 = 1.0 + 0.001 * (z[-1]**2 + z[0]**2)
        result += 0.5 + (temp1 - 0.5) / (temp2 ** 2)
        
        return result
    
    # ==================== Hybrid Functions ====================
    
    def _hf02(self, x: np.ndarray) -> float:
        """Hybrid Function 2 (3 basic functions)"""
        cf_num = 3
        Gp = [0.4, 0.4, 0.2]
        G_nx = [int(np.ceil(Gp[i] * self.dim)) for i in range(cf_num - 1)]
        G_nx.append(self.dim - sum(G_nx))
        G = [0] + list(np.cumsum(G_nx[:-1]))
        
        z = self._shift_rotate(x, self.OShift, self.M, 1.0, True, True)
        y = z[self.SS]  # Shuffle
        
        fit = np.zeros(cf_num)
        fit[0] = self._bent_cigar(y[G[0]:G[0]+G_nx[0]], np.zeros(G_nx[0]), np.eye(G_nx[0]), False, False)
        fit[1] = self._hgbat(y[G[1]:G[1]+G_nx[1]], np.zeros(G_nx[1]), np.eye(G_nx[1]), False, False)
        fit[2] = self._rastrigin(y[G[2]:G[2]+G_nx[2]], np.zeros(G_nx[2]), np.eye(G_nx[2]), False, False)
        
        return np.sum(fit)
    
    def _hf10(self, x: np.ndarray) -> float:
        """Hybrid Function 10 (6 basic functions)"""
        cf_num = 6
        Gp = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2]
        G_nx = [int(np.ceil(Gp[i] * self.dim)) for i in range(cf_num - 1)]
        G_nx.append(self.dim - sum(G_nx))
        G = [0] + list(np.cumsum(G_nx[:-1]))
        
        z = self._shift_rotate(x, self.OShift, self.M, 1.0, True, True)
        y = z[self.SS]  # Shuffle
        
        fit = np.zeros(cf_num)
        fit[0] = self._hgbat(y[G[0]:G[0]+G_nx[0]], np.zeros(G_nx[0]), np.eye(G_nx[0]), False, False)
        fit[1] = self._katsuura(y[G[1]:G[1]+G_nx[1]], np.zeros(G_nx[1]), np.eye(G_nx[1]), False, False)
        fit[2] = self._ackley(y[G[2]:G[2]+G_nx[2]], np.zeros(G_nx[2]), np.eye(G_nx[2]), False, False)
        fit[3] = self._rastrigin(y[G[3]:G[3]+G_nx[3]], np.zeros(G_nx[3]), np.eye(G_nx[3]), False, False)
        fit[4] = self._schwefel(y[G[4]:G[4]+G_nx[4]], np.zeros(G_nx[4]), np.eye(G_nx[4]), False, False)
        fit[5] = self._schaffer_f7(y[G[5]:G[5]+G_nx[5]], np.zeros(G_nx[5]), np.eye(G_nx[5]), False, False)
        
        return np.sum(fit)
    
    def _hf06(self, x: np.ndarray) -> float:
        """Hybrid Function 6 (5 basic functions)"""
        cf_num = 5
        Gp = [0.3, 0.2, 0.2, 0.1, 0.2]
        G_nx = [int(np.ceil(Gp[i] * self.dim)) for i in range(cf_num - 1)]
        G_nx.append(self.dim - sum(G_nx))
        G = [0] + list(np.cumsum(G_nx[:-1]))
        
        z = self._shift_rotate(x, self.OShift, self.M, 1.0, True, True)
        y = z[self.SS]  # Shuffle
        
        fit = np.zeros(cf_num)
        fit[0] = self._katsuura(y[G[0]:G[0]+G_nx[0]], np.zeros(G_nx[0]), np.eye(G_nx[0]), False, False)
        fit[1] = self._happycat(y[G[1]:G[1]+G_nx[1]], np.zeros(G_nx[1]), np.eye(G_nx[1]), False, False)
        fit[2] = self._grie_rosen(y[G[2]:G[2]+G_nx[2]], np.zeros(G_nx[2]), np.eye(G_nx[2]), False, False)
        fit[3] = self._schwefel(y[G[3]:G[3]+G_nx[3]], np.zeros(G_nx[3]), np.eye(G_nx[3]), False, False)
        fit[4] = self._ackley(y[G[4]:G[4]+G_nx[4]], np.zeros(G_nx[4]), np.eye(G_nx[4]), False, False)
        
        return np.sum(fit)
    
    # ==================== Composition Functions ====================
    
    def _cf_cal(self, x: np.ndarray, Os: np.ndarray, delta: np.ndarray, 
                bias: np.ndarray, fit: np.ndarray, cf_num: int) -> float:
        """Composition function calculation"""
        w = np.zeros(cf_num)
        
        for i in range(cf_num):
            dist = np.sum((x - Os[i]) ** 2)
            if dist != 0:
                w[i] = np.exp(-dist / (2 * self.dim * delta[i]**2)) / np.sqrt(dist)
            else:
                w[i] = INF
        
        w_max = np.max(w)
        w_sum = np.sum(w)
        
        if w_max == 0:
            w = np.ones(cf_num)
            w_sum = cf_num
        
        fit = fit + bias
        return np.sum(w / w_sum * fit)
    
    def _cf01(self, x: np.ndarray) -> float:
        """Composition Function 1 (5 functions)"""
        cf_num = 5
        delta = np.array([10, 20, 30, 40, 50])
        bias = np.array([0, 200, 300, 100, 400])
        fit = np.zeros(cf_num)
        
        fit[0] = 10000 * self._rosenbrock(x, self.OShift[0], self.M[0], True, True) / 1e4
        fit[1] = 10000 * self._ellips(x, self.OShift[1], self.M[1], True, True) / 1e10
        fit[2] = 10000 * self._bent_cigar(x, self.OShift[2], self.M[2], True, True) / 1e30
        fit[3] = 10000 * self._discus(x, self.OShift[3], self.M[3], True, True) / 1e10
        fit[4] = 10000 * self._ellips(x, self.OShift[4], self.M[4], True, False) / 1e10
        
        return self._cf_cal(x, self.OShift, delta, bias, fit, cf_num)
    
    def _cf02(self, x: np.ndarray) -> float:
        """Composition Function 2 (3 functions)"""
        cf_num = 3
        delta = np.array([20, 10, 10])
        bias = np.array([0, 200, 100])
        fit = np.zeros(cf_num)
        
        fit[0] = self._schwefel(x, self.OShift[0], self.M[0], True, False)
        fit[1] = self._rastrigin(x, self.OShift[1], self.M[1], True, True)
        fit[2] = self._hgbat(x, self.OShift[2], self.M[2], True, True)
        
        return self._cf_cal(x, self.OShift, delta, bias, fit, cf_num)
    
    def _cf06(self, x: np.ndarray) -> float:
        """Composition Function 6 (5 functions)"""
        cf_num = 5
        delta = np.array([20, 20, 30, 30, 20])
        bias = np.array([0, 200, 300, 400, 200])
        fit = np.zeros(cf_num)
        
        fit[0] = 10000 * self._escaffer6(x, self.OShift[0], self.M[0], True, True) / 2e7
        fit[1] = self._schwefel(x, self.OShift[1], self.M[1], True, True)
        fit[2] = 1000 * self._rastrigin(x, self.OShift[2], self.M[2], True, True) / 100
        fit[3] = self._rosenbrock(x, self.OShift[3], self.M[3], True, True)
        fit[4] = 10000 * self._rastrigin(x, self.OShift[4], self.M[4], True, True) / 1e3
        
        return self._cf_cal(x, self.OShift, delta, bias, fit, cf_num)
    
    def _cf07(self, x: np.ndarray) -> float:
        """Composition Function 7 (6 functions)"""
        cf_num = 6
        delta = np.array([10, 20, 30, 40, 50, 60])
        bias = np.array([0, 300, 500, 100, 400, 200])
        fit = np.zeros(cf_num)
        
        fit[0] = 10000 * self._hgbat(x, self.OShift[0], self.M[0], True, True) / 1000
        fit[1] = 10000 * self._rastrigin(x, self.OShift[1], self.M[1], True, True) / 1e3
        fit[2] = 10000 * self._schwefel(x, self.OShift[2], self.M[2], True, True) / 4e3
        fit[3] = 10000 * self._bent_cigar(x, self.OShift[3], self.M[3], True, True) / 1e30
        fit[4] = 10000 * self._ellips(x, self.OShift[4], self.M[4], True, True) / 1e10
        fit[5] = 10000 * self._escaffer6(x, self.OShift[5], self.M[5], True, True) / 2e7
        
        return self._cf_cal(x, self.OShift, delta, bias, fit, cf_num)


def cec2022_func(func_num: int, dim: int = 10) -> Callable:
    """
    Create a CEC 2022 benchmark function
    
    Parameters
    ----------
    func_num : int
        Function number (1-12)
    dim : int
        Dimension (2, 10, or 20)
        
    Returns
    -------
    Callable
        Function that takes a vector and returns fitness
        
    Example
    -------
    >>> f = cec2022_func(1, 10)
    >>> fitness = f(np.random.randn(10))
    """
    benchmark = CEC2022(func_num, dim)
    return benchmark


# Convenience function names
def get_cec2022_bounds(dim: int) -> np.ndarray:
    """
    Get bounds for CEC 2022 functions
    
    Parameters
    ----------
    dim : int
        Dimension
        
    Returns
    -------
    np.ndarray
        Bounds array of shape (dim, 2)
    """
    return np.array([[-100, 100]] * dim)
