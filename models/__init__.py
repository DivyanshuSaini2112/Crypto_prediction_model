try:
    from .orbit import Orbit
except ImportError:
    class Orbit:  # type: ignore
        def __init__(self, *a, **kw):
            raise ImportError("Orbit requires orbit-ml: pip install orbit-ml==1.1.3")


MODELS = {
    'orbit': Orbit
}
