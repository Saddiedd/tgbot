from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(alias="TELEGRAM_BOT_TOKEN")
    telegram_request_timeout: int = Field(default=10, alias="TELEGRAM_REQUEST_TIMEOUT")
    telegram_proxy_url: str | None = Field(default=None, alias="TELEGRAM_PROXY_URL")
    mistral_api_key: str | None = Field(default=None, alias="MISTRAL_API_KEY")
    mistral_embedding_model: str = Field(default="mistral-embed", alias="MISTRAL_EMBEDDING_MODEL")
    mistral_chat_model: str = Field(default="mistral-small-latest", alias="MISTRAL_CHAT_MODEL")
    analysis_provider: str = Field(default="auto", alias="ANALYSIS_PROVIDER")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="gemma4:latest", alias="OLLAMA_CHAT_MODEL")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    lancedb_path: str = Field(default="./data/lancedb", alias="LANCEDB_PATH")
    lancedb_table: str = Field(default="kompas_docs", alias="LANCEDB_TABLE")
    database_url: str = Field(default="sqlite+aiosqlite:///./data/bot.db", alias="DATABASE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def load_settings() -> Settings:
    return Settings()
