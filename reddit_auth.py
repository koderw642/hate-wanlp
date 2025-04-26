import requests
import secrets
import logging
from fastapi import HTTPException
from urllib.parse import urlencode

class Config:
    REDDIT_CLIENT_ID = "3aHkvg1zTxdOo3fsekvpnw"
    REDDIT_CLIENT_SECRET = "1va7WqM7ZgtXO9X1eI0f0ePgQtrMjA"
    REDDIT_REDIRECT_URI = "http://localhost:8000/auth/callback"
    REDDIT_AUTH_URL = "https://www.reddit.com/api/v1/authorize"
    REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
    REDDIT_API_URL = "https://oauth.reddit.com"

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory session storage
sessions = {}

def generate_state() -> str:
    """
    Generate a unique state string for OAuth2 security.
    """
    state = secrets.token_urlsafe(16)
    sessions[state] = True
    logger.info(f"Generated state: {state}")
    return state

def validate_state(state: str) -> bool:
    """
    Validate the state parameter to prevent CSRF attacks.
    """
    if state in sessions:
        del sessions[state]  # Remove the state after validation
        logger.info(f"State validated and removed: {state}")
        return True
    logger.warning(f"Invalid state detected: {state}")
    return False

def exchange_code_for_token(code: str) -> dict:
    """
    Exchange the authorization code for an access token.
    """
    headers = {
        "User-Agent": "ArabicHateSpeechDetector/1.0",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": Config.REDDIT_REDIRECT_URI
    }
    try:
        response = requests.post(
            Config.REDDIT_TOKEN_URL,
            data=data,
            auth=(Config.REDDIT_CLIENT_ID, Config.REDDIT_CLIENT_SECRET),
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        token_data = response.json()
        logger.info(f"Token exchange successful: {token_data}")
        return token_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Token exchange failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to exchange code for token")

def get_reddit_user_info(token):
    """
    Fetch Reddit user information using the provided access token.
    """
    try:
        response = requests.get(
            "https://oauth.reddit.com/api/v1/me",
            headers={"Authorization": f"Bearer {token}", "User-Agent": "ArabicHateSpeechDetector/1.0"},
            timeout=10
        )

        user_info = response.json()
        logger.info(f"Reddit user info fetched successfully: {user_info}")
        return user_info
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Reddit user info: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")