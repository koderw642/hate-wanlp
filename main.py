# main.py
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from urllib.parse import urlencode
from typing import Optional
import logging
from reddit_auth import (
    Config,
    generate_state,
    validate_state,
    exchange_code_for_token,
    get_reddit_user_info
)
from arabic_model import hate_speech_detector
from algerian_model import predict
# from reddit_auth import create_session
import time
# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class TextInput(BaseModel):
    text: str

# WebSocket Endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message received: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Detection Endpoints
@app.post("/detect-standard-arabic/")
async def detect_standard_arabic(input_data: TextInput):
    try:
        logger.info(f"Processing Standard Arabic text: {input_data.text[:50]}...")
        result = hate_speech_detector.detect(input_data.text)
        logger.info(f"Standard Arabic result: {result}")
        return result
    except Exception as e:
        logger.error(f"Standard Arabic detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-algerian-dialect/")
async def detect_algerian_dialect(data: TextInput):
    try:
        prediction = predict(data.text)
        return {
            "text": data.text,
            "hate_speech": prediction["hate_speech"],
            "topic": prediction["topic"]
        }
    except Exception as e:
        logger.error(f"Algerian dialect detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Reddit Authentication Endpoints
@app.get("/login/reddit")
async def login_reddit():
    state = generate_state()
    auth_url = (
        f"{Config.REDDIT_AUTH_URL}?"
        f"client_id={Config.REDDIT_CLIENT_ID}&"
        f"response_type=code&"
        f"state={state}&"
        f"redirect_uri={Config.REDDIT_REDIRECT_URI}&"
        f"duration=permanent&"
        f"scope=read identity history modposts edit report"  # Add 'edit' scope
    )
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def reddit_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        raise HTTPException(status_code=400, detail=f"Reddit authorization failed: {error}")
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")
    if not validate_state(state):
        raise HTTPException(status_code=400, detail="Invalid state parameter")
    token_data = exchange_code_for_token(code)
    user_data = get_reddit_user_info(token_data["access_token"])
    params = {

    "access_token": token_data["access_token"],
    "expires_in": token_data["expires_in"],  # Token lifetime in seconds
    "user_name": user_data.get("name"),
    "user_id": user_data.get("id"),
    "icon_img": user_data.get("icon_img", "")

    }
    query_string = urlencode(params)  # Convert params to a query string
    redirect_url = f"http://localhost:8001/dashboard?{query_string}"
    return RedirectResponse(url=redirect_url)
@app.post("/reddit/delete-comment/")
async def delete_comment(request: Request):
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        token = auth_header.split(" ")[1]

        data = await request.json()
        comment_id = data.get("comment_id")
        if not comment_id:
            raise HTTPException(status_code=400, detail="Missing comment_id")

        if not comment_id.startswith("t1_"):
            comment_id = f"t1_{comment_id}"

        # First try standard deletion
        response = requests.post(
            "https://oauth.reddit.com/api/del",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "ArabicHateSpeechDetector/1.0"
            },
            data={"id": comment_id},
            timeout=10
        )

        if response.status_code == 200:
            return {"status": "success", "message": "Comment deleted successfully"}

        # If not authorized, check if user is post author
        if response.status_code == 403:
            # Get comment info
            comment_info = requests.get(
                f"https://oauth.reddit.com/api/info?id={comment_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "ArabicHateSpeechDetector/1.0"
                },
                timeout=10
            )
            if comment_info.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch comment info")

            comment_data = comment_info.json()
            parent_post_id = comment_data["data"]["children"][0]["data"]["link_id"]
            comment_author = comment_data["data"]["children"][0]["data"]["author"]

            # Get user info
            user_info = requests.get(
                "https://oauth.reddit.com/api/v1/me",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "ArabicHateSpeechDetector/1.0"
                },
                timeout=10
            )
            if user_info.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch user info")

            current_user = user_info.json()["name"]

            # Check if current user is comment author (should have worked first try)
            if current_user == comment_author:
                raise HTTPException(
                    status_code=403,
                    detail="Unexpected error deleting your own comment"
                )

            # Get post info
            post_info = requests.get(
                f"https://oauth.reddit.com/api/info?id={parent_post_id}",
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "ArabicHateSpeechDetector/1.0"
                },
                timeout=10
            )
            if post_info.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to fetch post info")

            post_data = post_info.json()
            post_author = post_data["data"]["children"][0]["data"]["author"]

            # If user is post author, remove the comment
            if current_user == post_author:
                remove_response = requests.post(
                    "https://oauth.reddit.com/api/remove",
                    headers={
                        "Authorization": f"Bearer {token}",
                        "User-Agent": "ArabicHateSpeechDetector/1.0"
                    },
                    data={
                        "id": comment_id,
                        "spam": False
                    },
                    timeout=10
                )
                if remove_response.status_code == 200:
                    return {"status": "success", "message": "Comment removed from your post"}
                else:
                    raise HTTPException(
                        status_code=remove_response.status_code,
                        detail="Failed to remove comment from post"
                    )
            else:
                raise HTTPException(
                    status_code=403,
                    detail="Not authorized to delete this comment"
                )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to delete comment: {response.text}"
            )

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to Reddit API timed out")
    except Exception as e:
        logger.error(f"Error in delete-comment: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
 
@app.post("/reddit/report-comment/")
async def report_comment(request: Request):
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        token = auth_header.split(" ")[1]
        data = await request.json()
        comment_id = data.get("comment_id")
        if not comment_id:
            raise HTTPException(status_code=400, detail="Missing comment_id")
        response = requests.post(
            "https://oauth.reddit.com/api/report",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "ArabicHateSpeechDetector/1.0"
            },
            data={"thing_id": f"t1_{comment_id}", "reason": "hate_speech"},  # Ensure t1_ prefix
            timeout=10
        )
        if response.status_code != 200:
            logger.error(f"Reddit API error: {response.status_code} - {response.text}")
            raise HTTPException(status_code=response.status_code, detail=f"Reddit API error: {response.text}")
        return {"status": "success", "message": "Comment reported successfully"}
    except Exception as e:
        logger.error(f"Error reporting comment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to report comment: {str(e)}")
@app.post("/reddit/report-comment/")
async def report_comment(request: Request):
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid token")
        token = auth_header.split(" ")[1]
        data = await request.json()
        comment_id = data.get("comment_id")
        if not comment_id:
            raise HTTPException(status_code=400, detail="Missing comment_id")
        response = requests.post(
            "https://oauth.reddit.com/api/report",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "ArabicHateSpeechDetector/1.0"
            },
            data={"thing_id": f"t1_{comment_id}", "reason": "hate_speech"},  # Ensure t1_ prefix
            timeout=10
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to report comment")
        return {"status": "success", "message": "Comment reported successfully"}
    except Exception as e:
        logger.error(f"Error reporting comment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to report comment: {str(e)}")

@app.post("/reddit/approve-comment/")
async def approve_comment(request: Request):
    try:
        access_token = request.headers.get("Authorization", "").replace("Bearer ", "")
        body = await request.json()
        comment_id = body.get("comment_id")
        if not comment_id:
            raise HTTPException(status_code=400, detail="Missing comment_id")
        if not comment_id.startswith('t1_'):
            comment_id = f't1_{comment_id}'
        headers = {
            "Authorization": f"bearer {access_token}",
            "User-Agent": "ArabicHateSpeechDetector/1.0",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        response = requests.post(
            "https://oauth.reddit.com/api/approve",
            headers=headers,
            data={"id": comment_id},
            timeout=10
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Reddit error: {response.text}")
        return {"status": "success", "message": "Comment approved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# User Analysis Endpoint
@app.get("/analyze-me/")
async def analyze_me(request: Request):
    try:
        # Authentication check
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization token")
        
        token = auth_header.split(" ")[1]
        
        # Get user info first to verify token
        try:
            user_info = get_reddit_user_info(token)
            username = user_info.get("name")
            if not username:
                raise HTTPException(status_code=400, detail="Could not get username")
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

        # Fetch user comments
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "ArabicHateSpeechDetector/1.0"
        }
        
        # Get both posts and comments
        posts_response = requests.get(
            f"{Config.REDDIT_API_URL}/user/{username}/submitted",
            headers=headers,
            params={"limit": 100}
        )
        posts_response.raise_for_status()
        
        comments_response = requests.get(
            f"{Config.REDDIT_API_URL}/user/{username}/comments",
            headers=headers,
            params={"limit": 100}
        )
        comments_response.raise_for_status()

        # Process posts with their comments
        posts_with_comments = []
        for post in posts_response.json().get("data", {}).get("children", []):
            post_data = post["data"]
            post_id = post_data["id"]
            
            # Get comments for this post
            post_comments_response = requests.get(
                f"{Config.REDDIT_API_URL}/comments/{post_id}",
                headers=headers,
                params={"limit": 50}
            )
            post_comments_response.raise_for_status()
            
            # Analyze post content
            post_text = post_data.get("selftext", "") or post_data.get("title", "")
            post_prediction = predict(post_text)
            
            # Process comments for this post
            post_comments = []
            post_hate_comments = []
            
            for comment in post_comments_response.json()[1]["data"]["children"]:
                comment_data = comment["data"]
                comment_text = comment_data.get("body", "")
                if not comment_text:
                    continue
                    
                comment_prediction = predict(comment_text)
                comment_item = {
                    "id": comment_data.get("id"),
                    "text": comment_text,
                    "url": f"https://reddit.com{comment_data.get('permalink', '')}",
                    "created_utc": comment_data.get("created_utc"),
                    "hate_speech": comment_prediction["hate_speech"],
                    "topic": comment_prediction["topic"]
                }
                
                post_comments.append(comment_item)
                if comment_prediction["hate_speech"] == "Hate Speech":
                    post_hate_comments.append(comment_item)
            
            posts_with_comments.append({
                "post": {
                    "id": post_id,
                    "title": post_data.get("title"),
                    "text": post_text,
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                    "hate_speech": post_prediction["hate_speech"],
                    "topic": post_prediction["topic"]
                },
                "comments": post_comments,
                "hate_comments": post_hate_comments
            })

        # Process standalone comments
        standalone_comments = []
        standalone_hate_comments = []
        
        for comment in comments_response.json().get("data", {}).get("children", []):
            comment_data = comment["data"]
            comment_text = comment_data.get("body", "")
            if not comment_text:
                continue
                
            comment_prediction = predict(comment_text)
            comment_item = {
                "id": comment_data.get("id"),
                "text": comment_text,
                "url": f"https://reddit.com{comment_data.get('permalink', '')}",
                "created_utc": comment_data.get("created_utc"),
                "hate_speech": comment_prediction["hate_speech"],
                "topic": comment_prediction["topic"]
            }
            
            standalone_comments.append(comment_item)
            if comment_prediction["hate_speech"] == "Hate Speech":
                standalone_hate_comments.append(comment_item)

        # Calculate statistics
        total_comments = (
            sum(len(p["comments"]) for p in posts_with_comments) + 
            len(standalone_comments)
        )
        
        total_hate_comments = (
            sum(len(p["hate_comments"]) for p in posts_with_comments) + 
            len(standalone_hate_comments)
        )

        return {
            "user": {
                "name": username,
                "icon_img": user_info.get("icon_img", "").split("?")[0],
                "total_karma": user_info.get("total_karma", 0)
            },
            "stats": {
                "total_posts": len(posts_with_comments),
                "total_comments": total_comments,
                "hate_comments": total_hate_comments,
                "hate_percentage": (total_hate_comments / total_comments * 100) if total_comments > 0 else 0
            },
            "posts_with_comments": posts_with_comments,
            "standalone_comments": standalone_comments,
            "all_hate_comments": [
                *[hc for p in posts_with_comments for hc in p["hate_comments"]],
                *standalone_hate_comments
            ]
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Reddit API request failed: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )  
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    