import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from backend.services.auth_service import AuthService
from backend.models.schemas import UserCreate
from backend.core.security import create_access_token, decode_token, hash_password, verify_password


@pytest.mark.asyncio
async def test_register_success(db_session: AsyncSession):
    svc = AuthService(db_session)
    user = await svc.register(UserCreate(
        username="newuser",
        email="newuser@test.com",
        password="securepass123",
    ))
    assert user.user_id is not None
    assert user.username == "newuser"
    assert user.email == "newuser@test.com"
    assert user.role.value == "analyst"


@pytest.mark.asyncio
async def test_register_duplicate_username(db_session: AsyncSession):
    from fastapi import HTTPException
    svc = AuthService(db_session)
    await svc.register(UserCreate(username="dup", email="dup@test.com", password="pass12345"))
    with pytest.raises(HTTPException) as exc_info:
        await svc.register(UserCreate(username="dup", email="another@test.com", password="pass12345"))
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_login_success(db_session: AsyncSession):
    svc = AuthService(db_session)
    await svc.register(UserCreate(username="loginuser", email="login@test.com", password="mypassword1"))
    tokens = await svc.login("loginuser", "mypassword1")
    assert tokens.access_token
    assert tokens.refresh_token
    assert tokens.token_type == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(db_session: AsyncSession):
    from fastapi import HTTPException
    svc = AuthService(db_session)
    await svc.register(UserCreate(username="wrongpass", email="wp@test.com", password="correct123"))
    with pytest.raises(HTTPException) as exc_info:
        await svc.login("wrongpass", "wrong")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_login_nonexistent_user(db_session: AsyncSession):
    from fastapi import HTTPException
    svc = AuthService(db_session)
    with pytest.raises(HTTPException) as exc_info:
        await svc.login("ghost", "password")
    assert exc_info.value.status_code == 401


def test_access_token_creation_and_decode():
    payload = {"sub": "42", "role": "analyst"}
    token = create_access_token(payload)
    decoded = decode_token(token)
    assert decoded["sub"] == "42"
    assert decoded["type"] == "access"


def test_password_hash_and_verify():
    password = "mysecretpassword"
    hashed = hash_password(password)
    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_invalid_token_raises():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        decode_token("invalid.token.string")


@pytest.mark.asyncio
async def test_refresh_token(db_session: AsyncSession):
    svc = AuthService(db_session)
    await svc.register(UserCreate(username="refreshuser", email="refresh@test.com", password="pass12345"))
    tokens = await svc.login("refreshuser", "pass12345")
    new_tokens = await svc.refresh(tokens.refresh_token)
    assert new_tokens.access_token
    assert new_tokens.access_token != tokens.access_token
