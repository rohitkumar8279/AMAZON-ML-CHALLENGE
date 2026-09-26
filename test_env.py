import sys

print('Python version:', sys.version)

try:
    import numpy
    print('NumPy OK:', numpy.__version__)
except Exception as e:
    print('NumPy Error:', type(e).__name__, e)

try:
    import pandas
    print('Pandas OK:', pandas.__version__)
except Exception as e:
    print('Pandas Error:', type(e).__name__, e)

try:
    import scipy
    print('SciPy OK:', scipy.__version__)
except Exception as e:
    print('SciPy Error:', type(e).__name__, e)

try:
    import sklearn
    print('Scikit-learn OK:', sklearn.__version__)
except Exception as e:
    print('Scikit-learn Error:', type(e).__name__, e)

try:
    import rapidfuzz
    print('RapidFuzz OK:', rapidfuzz.__version__)
except Exception as e:
    print('RapidFuzz Error:', type(e).__name__, e)

try:
    import lightgbm
    print('LightGBM OK:', lightgbm.__version__)
except Exception as e:
    print('LightGBM Error:', type(e).__name__, e)
