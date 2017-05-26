import readwrite
import fits 
import calibstn
import projcomp
import prepdata
import toplot

__all__ = ["readwrite","fits","calibstn","projcomp","prepdata","toplot"]

# include the capability to calculate predicted strain with Okada?
# requires okada_wrapper from Ben Thompson
# https://github.com/tbenthompson/okada_wrapper.git
incpred = False

if incpred:
    import predict
    __all__.append("predict")


# include codes to retrieve and organize data
incdata = False

if incdata:
    import datamanage
    __all__.append("datamanage")
