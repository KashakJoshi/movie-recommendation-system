from fastapi import APIRouter
from api.service import get_recommendations

router = APIRouter()


@router.get("/recommend")
def recommend(user_id: int):

    return get_recommendations(user_id)