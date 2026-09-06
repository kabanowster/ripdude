import redis
from langgraph.checkpoint.redis import AsyncRedisSaver

from ripdude.config import REDIS_URL, AGENT_NAME, REDIS_TTL_DAYS

redis_client = redis.asyncio.Redis.from_url(REDIS_URL)

TTL_MINUTES = REDIS_TTL_DAYS * 24 * 60

redis_saver = AsyncRedisSaver(
    redis_client=redis_client,
    checkpoint_prefix=f"{AGENT_NAME}:checkpoint",
    ttl={"checkpoint": TTL_MINUTES, "refresh_on_read": True}
)