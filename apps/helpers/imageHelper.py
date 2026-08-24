"""Task utilities for dealing with images"""

import re


class ImageHelper:
    @staticmethod
    def sanitize_filename(name: str, replacement: str = "_") -> str:
        return re.sub(r'[<>:"/\.\\|?*\x00-\x1f]', replacement, name)
