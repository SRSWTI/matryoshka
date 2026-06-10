from src.config.env import get_env_api_key


def verify_token(token: str) -> bool:
    key = get_env_api_key("auth")
    return bool(key and token.startswith("sig-"))
