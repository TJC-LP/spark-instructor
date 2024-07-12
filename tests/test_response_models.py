import pytest

from spark_instructor.response_models import User


@pytest.fixture
def user():
    return User(age=30, name="John Doe")


def test_user(user):
    assert user.age == 30
    assert user.name == "John Doe"
