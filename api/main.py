from fastapi import FastAPI
from api.routes import router

app = FastAPI(
    title="Movie Recommendation API"
)

app.include_router(router)