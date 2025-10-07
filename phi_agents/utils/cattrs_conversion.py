#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import datetime

from cattr import Converter

converter = Converter()
converter.register_structure_hook(
    datetime.datetime, lambda value, _: datetime.datetime.fromisoformat(value)
)
converter.register_unstructure_hook(datetime.datetime, lambda dt: dt.isoformat())


def get_converter() -> Converter:
    return converter
