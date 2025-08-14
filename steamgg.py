import argparse
import asyncio
import json
import math
import re
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

from urllib.parse import urljoin
from urllib.parse import urlparse

try:
    import requests
except Exception as exc:  # pragma: no cover
    print("Требуется библиотека 'requests'. Установите: pip install requests", file=sys.stderr)
    raise

try:
    from bs4 import BeautifulSoup
except Exception as exc:  # pragma: no cover
    print("Требуется библиотека 'beautifulsoup4'. Установите: pip install beautifulsoup4", file=sys.stderr)
    raise

try:
    import aiohttp
except Exception as exc:  # pragma: no cover
    print("Требуется библиотека 'aiohttp'. Установите: pip install aiohttp", file=sys.stderr)
    raise

try:
    import orjson  # type: ignore
    ORJSON_AVAILABLE = True
except Exception:
    ORJSON_AVAILABLE = False

try:
    from requests.adapters import HTTPAdapter
except Exception:
    HTTPAdapter = None  # type: ignore

try:
    from colorama import Fore, Style, init as colorama_init
    COLORAMA_AVAILABLE = True
except Exception:
    COLORAMA_AVAILABLE = False
    class Dummy:
        RESET_ALL = ""
        GREEN = ""
        DIM = ""
    # Fallbacks when colorama is not available
    Fore = Dummy()  # type: ignore
    Style = Dummy()  # type: ignore


BASE_URL = "https://steamgg.net/"
AZ_URL = urljoin(BASE_URL, "a-z-games-all-games/")


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "ru,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
        }
    )
    return session


def configure_http_pool(session: requests.Session, pool_size: int) -> None:
    if HTTPAdapter is None:
        return
    pool_size = max(10, pool_size)
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)


def fetch_with_retries(session: requests.Session, url: str, *, max_attempts: int = 4, timeout: int = 30) -> Optional[str]:
    delay_seconds = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, timeout=timeout)
            if 200 <= response.status_code < 300:
                return response.text
            # Retry on 5xx or 429
            if response.status_code in {429} or 500 <= response.status_code < 600:
                raise RuntimeError(f"HTTP {response.status_code}")
            # Non-retryable
            return None
        except Exception:
            if attempt == max_attempts:
                return None
            time.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2.0, 10.0)
    return None


def parse_letters_filter(letters: Optional[str]) -> Optional[Set[str]]:
    if not letters:
        return None
    chosen = set()
    for token in letters.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "_":
            chosen.add("_")
        elif len(token) == 1 and token.isalpha():
            chosen.add(token.upper())
    return chosen or None


def _parse_letter_game_map_from_html(html: str, letters_filter: Optional[Set[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    if not html:
        return [], {}
    soup = BeautifulSoup(html, "lxml")
    letters_order: List[str] = []
    letter_to_urls: Dict[str, List[str]] = {}
    for letter_section in soup.select("div.letter-section"):
        section_id = letter_section.get("id", "")
        letter_token = None
        match = re.search(r"a-z-listing-letter-([A-Za-z_])-(?:\d+)$", section_id)
        if match:
            letter_token = match.group(1)
        if letter_token is None:
            continue
        letter_key = letter_token.upper() if letter_token != "_" else "_"
        if letters_filter is not None:
            allowed = (letter_key in letters_filter) or (letter_key == "_" and "_" in letters_filter)
            if not allowed:
                continue
        urls_for_letter: List[str] = []
        for a in letter_section.select("li > a[href]"):
            href = a.get("href", "").strip()
            if not href:
                continue
            absolute = urljoin(BASE_URL, href)
            if absolute not in urls_for_letter:
                urls_for_letter.append(absolute)
        if urls_for_letter:
            letter_to_urls[letter_key] = urls_for_letter
            letters_order.append(letter_key)
    if "_" in letters_order:
        letters_order = [l for l in letters_order if l != "_"] + ["_"]
    return letters_order, letter_to_urls


async def async_collect_letter_game_map(session: aiohttp.ClientSession, letters_filter: Optional[Set[str]]) -> Tuple[List[str], Dict[str, List[str]]]:
    # Faster, low-timeout index fetch with small retries
    delay = 0.3
    for attempt in range(3):
        try:
            async with session.get(AZ_URL, timeout=aiohttp.ClientTimeout(total=12)) as resp:
                if 200 <= resp.status < 300:
                    html = await resp.text()
                    return _parse_letter_game_map_from_html(html, letters_filter)
                if resp.status in (429,) or 500 <= resp.status < 600:
                    pass
                else:
                    return [], {}
        except Exception:
            if attempt == 2:
                return [], {}
        await asyncio.sleep(delay)
        delay = min(delay * 2.0, 3.0)
    return [], {}


def extract_first_meta_content(soup: BeautifulSoup, properties: List[str]) -> Optional[str]:
    for prop in properties:
        tag = soup.find("meta", attrs={"property": prop})
        if tag and tag.get("content"):
            return tag.get("content").strip()
    return None


def normalize_iso8601(dt_str: Optional[str]) -> Optional[str]:
    if not dt_str:
        return None
    s = dt_str.strip()
    # Common cases: 2024-07-04T18:04:18+00:00 -> 2024-07-04T18:04:18Z
    m = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\+00:00|Z)?$", s)
    if m:
        return f"{m.group(1)}Z"
    # If already contains timezone, leave as-is
    return s


def extract_title(soup: BeautifulSoup) -> Optional[str]:
    # Primary title in <h2>
    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        return h2.get_text(strip=True)
    # Fallback to <h1> if exists
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    return None


def _normalize_unit(unit: str) -> str:
    u = unit.upper()
    return "GB" if u == "GO" else u


def _try_parse_size(text: str) -> Optional[str]:
    m = re.search(r"(\d+(?:[\.,]\d+)?)\s*(TB|GB|MB|KB|B|GIB|MIB|KIB|GO)\b", text, flags=re.IGNORECASE)
    if m:
        number = m.group(1).replace(",", ".")
        unit = _normalize_unit(m.group(2))
        return f"{number} {unit}"
    m2 = re.search(r"(\d+)\s*(?:gigabytes|gb)\b", text, flags=re.IGNORECASE)
    if m2:
        return f"{m2.group(1)} GB"
    return None


def _has_storage_hint(s: str) -> bool:
    s = s.lower()
    if "storage" in s:
        return True
    if "hard drive" in s or "hard disk" in s or "harddrive" in s or "harddisk" in s:
        return True
    if "download size" in s or "file size" in s or "install size" in s or "size on disk" in s:
        return True
    if ("hard" in s and ("drive" in s or "disk" in s)):
        return True
    if ("available" in s and "space" in s):
        return True
    if ("disk" in s and "space" in s):
        return True
    if ("drive" in s and "space" in s):
        return True
    if "hdd" in s:
        return True
    return False


def extract_file_size(soup: BeautifulSoup) -> Optional[str]:
    # 1) Prefer explicit list item with known label (Storage, Hard Drive, Download size, File size, ...)
    for li in soup.find_all("li"):
        strong = li.find("strong")
        label = strong.get_text(" ", strip=True).lower() if strong else ""
        if any(
            key in label
            for key in (
                "storage",
                "hard drive",
                "hard disk",
                "download size",
                "file size",
                "install size",
                "size on disk",
            )
        ):
            text = li.get_text(" ", strip=True)
            # Remove the label to focus on value
            value = text.replace(strong.get_text(" ", strip=True) if strong else "", "").strip(" :\u00A0\t\r\n")
            parsed = _try_parse_size(value)
            if parsed:
                return parsed

    # 2) Any <li> mentioning disk/drive/storage/space
    for li in soup.find_all("li"):
        line = li.get_text(" ", strip=True)
        if not _has_storage_hint(line):
            continue
        parsed = _try_parse_size(line)
        if parsed:
            return parsed

    # 3) Paragraphs with <br> or plain lines
    for p in soup.find_all("p"):
        text = p.get_text("\n", strip=True)
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if not _has_storage_hint(line):
                continue
            parsed = _try_parse_size(line)
            if parsed:
                return parsed

    # 4) Fallback: scan whole page text for the first line that looks like storage line
    all_text = soup.get_text("\n", strip=True)
    for raw_line in all_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not _has_storage_hint(line):
            continue
        parsed = _try_parse_size(line)
        if parsed:
            return parsed
    return None


def extract_download_links(soup: BeautifulSoup, page_url: str) -> List[str]:
    links: List[str] = []

    page_host = urlparse(page_url).netloc.lower()

    # Primary: explicit button containers
    for a in soup.select("div.vc_btn3-container a[href], a.vc_btn3[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        absolute = urljoin(page_url, href)
        target_host = urlparse(absolute).netloc.lower()
        # Keep only external hosts
        if target_host and target_host != page_host:
            if absolute not in links:
                links.append(absolute)

    # Fallback: sometimes buttons might miss expected classes; take external links with button-ish text
    if not links:
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not href:
                continue
            text_lower = (a.get_text(strip=True) or "").lower()
            if not ("download" in text_lower or "direct" in text_lower):
                continue
            absolute = urljoin(page_url, href)
            target_host = urlparse(absolute).netloc.lower()
            if target_host and target_host != page_host:
                if absolute not in links:
                    links.append(absolute)

    return links


def parse_game_page_html(html: str, url: str) -> Optional[Dict[str, object]]:
    soup = BeautifulSoup(html, "lxml")
    raw_dt = extract_first_meta_content(
        soup,
        [
            "article:modified_time",
            "article:published_time",
            "og:updated_time",
        ],
    )
    upload_date = normalize_iso8601(raw_dt)
    title = extract_title(soup)
    file_size = extract_file_size(soup) or "Unknown"
    uris = extract_download_links(soup, url)
    if not title and not uris:
        return None
    return {
        "title": title or "",
        "uris": uris,
        "uploadDate": upload_date,
        "fileSize": file_size,
        "repackLinkSource": url,
    }


def chunked(iterable: List[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [iterable]
    return [iterable[i : i + size] for i in range(0, len(iterable), size)]


class ConsoleProgress:
    def __init__(self, letters: List[str]) -> None:
        self.letters = letters
        self.done: Set[str] = set()
        self.enabled = True
        if COLORAMA_AVAILABLE:
            try:
                colorama_init()
            except Exception:
                pass
        self.lines_count = len(self.letters)

    def render_block(self) -> None:
        lines: List[str] = []
        for letter in self.letters:
            if letter in self.done:
                lines.append(f"{Fore.GREEN}{letter}{Style.RESET_ALL}")
            else:
                lines.append(f"{Style.DIM}{letter}{Style.RESET_ALL}")
        block = "\n".join(lines)
        print(block, end="", flush=True)

    def update_render(self) -> None:
        if not self.enabled:
            return
        # Move cursor up to the start of the block, re-render, and ensure cursor remains at end of block
        sys.stdout.write(f"\x1b[{self.lines_count}A")
        sys.stdout.flush()
        self.render_block()
        print("", flush=True)

    def start(self) -> None:
        if not self.letters:
            return
        self.render_block()
        print("", flush=True)

    def set_done(self, letter: str) -> None:
        if letter not in self.letters:
            return
        self.done.add(letter)
        self.update_render()

    def finalize(self) -> None:
        if not self.letters:
            return
        # Ensure cursor is below the block
        print("", flush=True)


async def _fetch_all(session: aiohttp.ClientSession, urls: List[str], concurrency: int = 32) -> Dict[str, Optional[str]]:
    semaphore = asyncio.Semaphore(concurrency)
    results: Dict[str, Optional[str]] = {}

    async def fetch_one(u: str) -> None:
        nonlocal results
        delay = 0.25
        for attempt in range(5):
            try:
                async with semaphore:
                    async with session.get(u, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if 200 <= resp.status < 300:
                            results[u] = await resp.text()
                            return
                        if resp.status in (429,) or 500 <= resp.status < 600:
                            # retry
                            pass
                        else:
                            results[u] = None
                            return
            except Exception:
                if attempt == 4:
                    results[u] = None
                    return
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 3.0)
        results[u] = None

    await asyncio.gather(*(fetch_one(u) for u in urls))
    return results


def run(letters: Optional[str], limit: Optional[int], max_workers: int, out_path: str) -> Tuple[List[Dict[str, object]], List[str]]:
    # Show immediate status to avoid perceived idle time
    print("Загружаю список игр...", flush=True)

    selected_letters = parse_letters_filter(letters)
    results: List[Dict[str, object]] = []
    errors: List[str] = []

    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    )
    headers = {
        "User-Agent": user_agent,
        "Accept-Language": "ru,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    # Максимальная агрессивная параллельность по умолчанию
    concurrency = max(32, min(max_workers * 4, 128))

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def runner() -> Tuple[List[str], Dict[str, List[str]], Dict[str, Optional[str]]]:
        connector = aiohttp.TCPConnector(
            limit=concurrency,
            limit_per_host=min(32, concurrency),
            ttl_dns_cache=300,
            ssl=False,
        )
        timeout = aiohttp.ClientTimeout(total=18)
        async with aiohttp.ClientSession(headers=headers, connector=connector, timeout=timeout) as session:
            # Fetch index quickly
            letters_order, letter_map = await async_collect_letter_game_map(session, selected_letters)

            # Build URL list
            url_to_letter: Dict[str, str] = {}
            flat_urls: List[str] = []
            if limit is not None and limit > 0:
                remaining = limit
                for letter in letters_order:
                    urls = letter_map.get(letter, [])
                    if not urls:
                        continue
                    take = urls[: max(0, remaining)]
                    for u in take:
                        if u not in url_to_letter:
                            url_to_letter[u] = letter
                            flat_urls.append(u)
                    remaining -= len(take)
                    if remaining <= 0:
                        break
                letters_in_progress = letters_order[:]
            else:
                for letter in letters_order:
                    for u in letter_map.get(letter, []):
                        if u not in url_to_letter:
                            url_to_letter[u] = letter
                            flat_urls.append(u)
                letters_in_progress = letters_order[:]

            if not flat_urls:
                return letters_in_progress, {}, {}

            # Progress setup
            letter_remaining: Dict[str, int] = {l: 0 for l in letters_in_progress}
            for u in flat_urls:
                ltr = url_to_letter.get(u)
                if ltr in letter_remaining:
                    letter_remaining[ltr] += 1

            progress = ConsoleProgress(letters_in_progress)
            progress.start()
            for ltr, count in letter_remaining.items():
                if count <= 0:
                    progress.set_done(ltr)

            # Start fetching pages
            html_map = await _fetch_all(session, flat_urls, concurrency=concurrency)

            # Update progress as we parse
            for url, html in html_map.items():
                related_letter = url_to_letter.get(url)
                try:
                    if html:
                        item = parse_game_page_html(html, url)
                        if item:
                            results.append(item)
                        else:
                            errors.append(f"Не удалось распарсить страницу: {url}")
                    else:
                        errors.append(f"Не удалось загрузить страницу: {url}")
                except Exception as exc:
                    errors.append(f"Ошибка при обработке {url}: {exc}")
                finally:
                    if related_letter in letter_remaining:
                        letter_remaining[related_letter] -= 1
                        if letter_remaining[related_letter] <= 0:
                            progress.set_done(related_letter)

            progress.finalize()
            return letters_in_progress, letter_map, html_map

    letters_in_progress, letter_map, _ = loop.run_until_complete(runner())
    loop.close()

    if not results:
        return results, ["Не удалось получить список игр или список пуст"]

    payload = {"name": "SteamGG", "downloads": results}
    if ORJSON_AVAILABLE:
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Парсер SteamGG: сбор игр и ссылок на загрузку")
    parser.add_argument(
        "--letters",
        type=str,
        default=None,
        help="Список букв через запятую для фильтрации (например: A,B,C или _ для прочих). По умолчанию — все",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество игр для обработки (для теста)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Количество потоков для загрузки страниц (по умолчанию 32)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="steamgg.json",
        help="Путь к выходному JSON-файлу",
    )

    args = parser.parse_args()

    results, errors = run(args.letters, args.limit, args.workers, args.output)

    print(f"Сохранено записей: {len(results)} в файл: {args.output}")
    if errors:
        print("Предупреждения/ошибки:")
        for e in errors[:20]:
            print(" - ", e)
        if len(errors) > 20:
            print(f" ... и ещё {len(errors) - 20} сообщений")


if __name__ == "__main__":
    main()


