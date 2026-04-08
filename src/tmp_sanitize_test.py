import sys
sys.path.insert(0, r'C:\Users\HP\Desktop\MathSensei\src')
from core.python_pipeline import sanitize_generated_python
raw = '''from sympy import *
Define the polynomials
poly1 = 7*x^4 - 3*x^3 - 3*x^2 - 8*x + 1
poly2 = 8*x^4 + 2*x^3 - 7*x^2 + 3*x + 4
product = expand(poly1 * poly2)
print('Coefficient of x^3:', product.coeff(x, 3))
'''
print(sanitize_generated_python(raw))
