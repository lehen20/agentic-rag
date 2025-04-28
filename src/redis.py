import langchain
from langchain_community.cache import RedisCache
import redis

def get_redis():
    
    # Connect to Redis (change host/port if needed)
    redis_client = redis.Redis(host="localhost", port=6379, db=0)

    # Set Redis as cache
    langchain.llm_cache = RedisCache(redis_client)