"""
This plugin searches for New York Times Access Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class NYTimesAccessTokenDetector(RegexBasedDetector):
    """Scans for New York Times Access Tokens."""

    secret_type = "New York Times Access Token"

    denylist = [
        re.compile(
            r"""(?i)(?:nytimes|new-york-times,|newyorktimes)(?:[0-9a-z\-_\t .]{0,20})(?:[\s|']|[\s|"]){0,3}(?:=|>|:{1,3}=|\|\|:|<=|=>|:|\?=)(?:'|\"|\s|=|\x60){0,5}([a-z0-9=_\-]{32})(?:['|\"|\n|\r|\s|\x60|;]|$)"""
        )
    ]
