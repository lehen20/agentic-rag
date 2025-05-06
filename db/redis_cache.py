import redis
from langchain_community.cache import RedisCache
import langchain
import yaml

with open("config/settings.yml") as f:
    config = yaml.safe_load(f)

redis_config = config["redis"]
redis_client = redis.Redis(**redis_config)
langchain.llm_cache = RedisCache(redis_client)

