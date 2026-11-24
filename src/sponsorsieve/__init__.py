"""SponsorSieve: a small local model that finds ads and sponsor reads in podcast transcripts."""
__version__ = "0.1.0"

from .spans import Span  # noqa: E402,F401


def __getattr__(name):
    if name == "Detector":
        from .detector import Detector

        return Detector
    raise AttributeError(name)
