"""
This plugin searches for DigitalOcean tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class DigitaloceanDetector(RegexBasedDetector):
    """Scans for various DigitalOcean Tokens."""

    secret_type = "DigitalOcean Token"

    denylist = [
        # OAuth Access Token
        re.compile(r"""(?i)\b(doo_v1_[a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
        # Personal Access Token
        re.compile(r"""(?i)\b(dop_v1_[a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
        # OAuth Refresh Token
        re.compile(r"""(?i)\b(dor_v1_[a-f0-9]{64})(?:['|\"|\n|\r|\s|\x60|;]|$)"""),
    ]
