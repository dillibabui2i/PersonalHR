from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    admin_email: str = "admin@personalhr.local"
    admin_password_hash: str = ""
    extension_key: str = ""
    web_origin: str = "http://127.0.0.1:3000"
    data_dir: str = "./data"
    flow_database: str = "./data/flows.db"
    max_upload_megabytes: int = 15

    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3.5:2b"
    planner_model: str = "qwen3.5:2b"
    history_turns: int = 6
    employee_fact_limit: int = 20
    ollama_keep_alive: str = "30m"
    ollama_num_ctx: int = 8192
    answer_max_tokens: int = 1536
    answer_cache_minutes: int = 1440
    embed_model: str = "nomic-embed-text"
    embed_dimensions: int = 768
    reranker_model: str = "BAAI/bge-reranker-base"

    chunk_max_characters: int = 1200
    chunk_min_characters: int = 200
    chunk_breakpoint_percentile: float = 90

    rag_expansion_count: int = 3
    skip_expansion_rerank_score: float = 0.85
    rag_vector_limit: int = 16
    rag_keyword_limit: int = 16
    rag_fusion_constant: int = 60
    rag_candidate_pool: int = 24
    rag_result_limit: int = 8
    rag_max_vector_distance: float = 0.95
    rag_min_rerank_score: float = 0.03
    rag_min_relative_rerank_score: float = 0.05
    rag_max_chunks_per_section: int = 3


def load_settings() -> Settings:
    return Settings()
