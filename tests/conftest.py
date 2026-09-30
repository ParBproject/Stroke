"""One holdout fit shared by the casebook tests."""

import pytest

from casebook.analyze import export


@pytest.fixture(scope="session")
def payload():
    return export()
