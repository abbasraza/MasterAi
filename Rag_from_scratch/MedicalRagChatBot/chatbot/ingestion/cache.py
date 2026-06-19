import json
from pathlib import Path


class CacheManager:

    def __init__(self, cache_dir: Path = Path("./json_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def exists(self, pdf_path: Path) -> bool:
        return self._cache_file(pdf_path).exists()

    def load(self, pdf_path: Path) -> dict:
        return json.loads(self._cache_file(pdf_path).read_text())

    def save(self, pdf_path: Path, data: dict):
        self._cache_file(pdf_path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def _cache_file(self, pdf_path: Path) -> Path:
        return self.cache_dir / f"{pdf_path.stem}.json"