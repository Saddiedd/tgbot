from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(validation_alias=AliasChoices("TELEGRAM_BOT_TOKEN", "TELEGRAM_BOT_API_KEY"))
    llm_provider: str = Field(default="openai", validation_alias=AliasChoices("LLM_PROVIDER", "CHAT_PROVIDER"))
    openai_api_key: str | None = Field(default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "MISTRAL_API_KEY"))
    openai_model: str = Field(default="gpt-4o-mini", validation_alias=AliasChoices("OPENAI_MODEL", "OLLAMA_CHAT_MODEL"))
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")
    lancedb_path: str = Field(default="./data/lancedb", alias="LANCEDB_PATH")
    lancedb_table: str = Field(default="kompas_docs", alias="LANCEDB_TABLE")
    database_url: str = Field(default="sqlite+aiosqlite:///./data/bot.db", alias="DATABASE_URL")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


def load_settings() -> Settings:
    return Settings()
