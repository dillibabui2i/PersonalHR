import uvicorn

from app.core.settings import load_settings


def main() -> None:
    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
