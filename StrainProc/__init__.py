__all__ = ["readwrite","fits","calibstn","projcomp","prepdata",
           "datamanage","toplot","genproc","tides"]

# include the capability to calculate predicted strain with Okada?
# requires okada_wrapper from Ben Thompson
# https://github.com/tbenthompson/okada_wrapper.git
incpred = False

if incpred:
    from . import predict
    __all__.append("predict")

from . import tides
from . import genproc
from . import calibstn
from . import readwrite
from . import projcomp
from . import fits 
from . import datamanage
from . import toplot
from . import prepdata

