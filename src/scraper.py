#!/usr/bin/env python3
"""
Improved async scraper targeted to maximize AI-article yield.

Features:
- RSS / sitemap / category discovery
- Article-URL heuristics (only likely article URLs)
- Async aiohttp fetching with Selenium fallback for pages with little/no HTML
- IMPROVED content extraction with multiple strategies
- Detailed diagnostics counters
- Config-driven (config_sources.json)
"""
import asyncio, aiohttp, hashlib, json, logging, re, time, os
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from urllib.parse import urlparse, urljoin
import feedparser
from xml.etree import ElementTree as ET
from typing import Optional

# -----------------------------
# CONFIG & PATHS
# -----------------------------
ROOT = Path.cwd()
CONFIG_PATH = ROOT / "config_sources.json"
OUT_DIR = ROOT / "data"
HTML_CACHE_DIR = OUT_DIR / "html_cache"
OUT_JSONL = OUT_DIR / "articles.jsonl"
DISCOVERED_FILE = OUT_DIR / "discovered_urls.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)

if not CONFIG_PATH.exists():
    raise SystemExit(f"Missing config file at {CONFIG_PATH}")

with open(CONFIG_PATH, "r", encoding="utf8") as fh:
    CONFIG = json.load(fh)

SITES = CONFIG.get("sites", [])
KEYWORDS = [k.lower() for k in CONFIG.get("keywords", [])]
MIN_CONTENT_CHARS = CONFIG.get("min_content_chars", 200)
MAX_CONTENT_CHARS = CONFIG.get("max_content_chars", 50000)
MAX_URLS_PER_SITE = CONFIG.get("max_urls_per_site", 100000)
MAX_URLS = CONFIG.get("max_urls", 1000000)
CONCURRENT_REQUESTS = CONFIG.get("concurrent_requests", 20)  # REDUCED for stability
SITEMAP_MAX_NESTED = CONFIG.get("sitemap_max_nested", 5)

# IMPROVED: More flexible article URL regex
ARTICLE_URL_REGEX = re.compile(r"""(
    /20\d{2}/|/202\d/|/article/|/news/|/articles/|/posts?/|/blog/|
    /\d{4}/\d{2}/\d{2}/|/\d{4}/\d{2}/|/story/|/p/|
    /discover/|/technology/|/ai-|/artificial-intelligence|
    /machine-learning|/deep-learning|/neural|/robotics|
    /[a-z]+-[a-z]+-[a-z0-9]+|/[0-9]{4,}/|/tagged/|/topic/
)""", re.I | re.X)

# Patterns to EXCLUDE (avoid homepages, indexes, feeds)
EXCLUDE_PATTERNS = re.compile(r"""(
    /feed/?$|/rss/?$|/sitemap|/category/?$|/tag/?$|/author/|
    /page/\d+/?$|/search|/wp-content|/wp-admin|\.xml$|\.pdf$|
    /login|/register|/cart|/checkout
)""", re.I | re.X)

# -----------------------------
# LOGGING
# -----------------------------
LOGFILE = ROOT / "scrape.log"
logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.FileHandler(LOGFILE, encoding="utf8"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger("scraper")

# -----------------------------
# UTILITIES
# -----------------------------
def now_iso(): return time.strftime("%Y-%m-%d %H:%M:%S")
def sha256(text: str): return hashlib.sha256(text.encode("utf8")).hexdigest()
def clean_text(text: Optional[str]) -> str:
    return " ".join(text.split()) if text else ""

def keyword_match(text: str):
    tl = text.lower()
    return {k: (k in tl) for k in KEYWORDS}

def load_existing_hashes():
    """Load content hashes and URLs from existing articles.jsonl to avoid duplicates"""
    existing_hashes = set()
    existing_urls = set()
    
    if OUT_JSONL.exists():
        logger.info(f"Loading existing articles from {OUT_JSONL}...")
        try:
            with open(OUT_JSONL, "r", encoding="utf8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line.strip())
                        if "content_hash" in obj:
                            existing_hashes.add(obj["content_hash"])
                        if "url" in obj:
                            existing_urls.add(obj["url"])
                    except:
                        continue
            logger.info(f"Loaded {len(existing_hashes)} existing content hashes and {len(existing_urls)} URLs")
        except Exception as e:
            logger.warning(f"Error loading existing articles: {e}")
    
    return existing_hashes, existing_urls

def save_jsonl(obj: dict):
    with open(OUT_JSONL, "a", encoding="utf8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

def save_html_cache(url, html):
    try:
        with open(HTML_CACHE_DIR / (sha256(url) + ".html"), "w", encoding="utf8") as fh:
            fh.write(html)
    except Exception:
        pass

def is_likely_article(url: str) -> bool:
    """Check if URL looks like an article (not homepage/index/feed)"""
    if EXCLUDE_PATTERNS.search(url):
        return False
    if ARTICLE_URL_REGEX.search(url):
        return True
    # Additional heuristic: has slug-like path with hyphens
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split("/") if p]
    if len(path_parts) >= 2:
        last_part = path_parts[-1]
        if len(last_part) > 10 and "-" in last_part:
            return True
    return False

# -----------------------------
# IMPROVED CONTENT EXTRACTION
# -----------------------------
def extract_article_content(soup: BeautifulSoup) -> str:
    """
    IMPROVED: Multiple strategies to extract article content
    Tries to find the main article content, not navigation/footer
    """
    content_text = ""
    
    # Strategy 1: Look for common article containers
    article_selectors = [
        'article',
        '[role="article"]',
        '.article-content',
        '.post-content',
        '.entry-content',
        '.content-body',
        '.article-body',
        'main article',
        '.story-body',
        '[itemprop="articleBody"]'
    ]
    
    for selector in article_selectors:
        containers = soup.select(selector)
        if containers:
            # Get paragraphs from the first matching container
            paragraphs = []
            for container in containers[:2]:  # Max 2 containers to avoid duplication
                paras = container.find_all('p', recursive=True)
                paragraphs.extend(paras)
            
            if paragraphs:
                texts = [p.get_text(separator=" ", strip=True) for p in paragraphs]
                content_text = " ".join(texts)
                if len(content_text) > 500:  # Minimum viable content
                    logger.debug(f"Extracted content using selector: {selector}")
                    return clean_text(content_text)
    
    # Strategy 2: Find the largest block of paragraphs
    all_paragraphs = soup.find_all('p')
    if all_paragraphs:
        # Group paragraphs by parent
        parent_groups = {}
        for p in all_paragraphs:
            parent = p.parent
            if parent:
                parent_id = id(parent)
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = []
                parent_groups[parent_id].append(p)
        
        # Find the parent with most paragraph content
        best_parent = None
        best_length = 0
        for parent_id, paras in parent_groups.items():
            combined = " ".join([p.get_text(separator=" ", strip=True) for p in paras])
            if len(combined) > best_length:
                best_length = len(combined)
                best_parent = paras
        
        if best_parent and best_length > 500:
            texts = [p.get_text(separator=" ", strip=True) for p in best_parent]
            content_text = " ".join(texts)
            logger.debug(f"Extracted content from largest paragraph group")
            return clean_text(content_text)
    
    # Strategy 3: Fallback - all paragraphs but filter out short ones
    if all_paragraphs:
        # Filter out navigation/footer paragraphs (usually short)
        meaningful_paras = [
            p.get_text(separator=" ", strip=True) 
            for p in all_paragraphs 
            if len(p.get_text(strip=True)) > 50  # Skip short paragraphs
        ]
        content_text = " ".join(meaningful_paras)
        logger.debug(f"Extracted content using filtered paragraphs")
        return clean_text(content_text)
    
    return clean_text(content_text)

# -----------------------------
# SELENIUM (fallback) - FIXED POOL ISSUE
# -----------------------------
class SeleniumDriverPool:
    """Manages a pool of Selenium drivers to prevent connection exhaustion"""
    def __init__(self, pool_size=3):
        self.pool_size = pool_size
        self.drivers = []
        self.available = asyncio.Queue()
        self.initialized = False
        
    async def init(self):
        """Initialize driver pool"""
        if self.initialized:
            return
        for i in range(self.pool_size):
            try:
                driver = self._create_driver()
                self.drivers.append(driver)
                await self.available.put(driver)
            except Exception as e:
                logger.warning(f"Failed to create driver {i}: {e}")
        self.initialized = True
        logger.info(f"Selenium driver pool initialized with {len(self.drivers)} drivers")
    
    def _create_driver(self):
        """Create a single Selenium driver"""
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1600,900")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-extensions")
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        service = Service()
        return webdriver.Chrome(service=service, options=options)
    
    async def get_driver(self):
        """Get a driver from the pool"""
        if not self.initialized:
            await self.init()
        return await self.available.get()
    
    async def return_driver(self, driver):
        """Return a driver to the pool"""
        await self.available.put(driver)
    
    def cleanup(self):
        """Close all drivers"""
        for driver in self.drivers:
            try:
                driver.quit()
            except:
                pass
        logger.info("Selenium driver pool cleaned up")

async def fetch_selenium(driver_pool, url, wait_seconds=2.0):
    """Fetch using Selenium with driver pooling"""
    driver = None
    try:
        driver = await driver_pool.get_driver()
        result = await asyncio.to_thread(_selenium_fetch_sync, driver, url, wait_seconds)
        return result
    finally:
        if driver:
            await driver_pool.return_driver(driver)

def _selenium_fetch_sync(driver, url: str, wait_seconds: float = 2.0):
    try:
        driver.get(url)
        time.sleep(wait_seconds)
        return {"status": 200, "text": driver.page_source, "headers": {}}
    except Exception as e:
        logger.debug(f"Selenium fetch error {url} -> {e}")
        return None

# -----------------------------
# ASYNC FETCH - IMPROVED CONNECTION HANDLING
# -----------------------------
async def fetch_aio(session: aiohttp.ClientSession, url: str, timeout=20):
    """Fetch with improved error handling and timeout"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    try:
        async with session.get(url, headers=headers, timeout=timeout, ssl=False) as resp:
            text = await resp.text(errors="ignore")
            return {"status": resp.status, "text": text, "headers": dict(resp.headers)}
    except asyncio.TimeoutError:
        logger.debug(f"Timeout fetching {url}")
        return None
    except Exception as e:
        logger.debug(f"AIO fetch error {url} -> {e}")
        return None

# -----------------------------
# DISCOVERY HELPERS
# -----------------------------
def discover_rss(feed_url, max_urls=5000):
    """Parse RSS/Atom feed"""
    try:
        d = feedparser.parse(feed_url)
        urls = []
        for e in d.entries:
            if hasattr(e, "link") and e.link:
                urls.append(e.link)
            if len(urls) >= max_urls:
                break
        return urls
    except Exception as e:
        logger.debug(f"RSS parse error {feed_url} -> {e}")
        return []

async def discover_sitemap(session, sitemap_url, max_urls=MAX_URLS_PER_SITE, nested=SITEMAP_MAX_NESTED, visited=None):
    """Recursively parse sitemap XML"""
    if visited is None: 
        visited = set()
    if sitemap_url in visited or nested <= 0:
        return []
    visited.add(sitemap_url)
    
    urls = []
    fetched = await fetch_aio(session, sitemap_url, timeout=30)
    if not fetched or not fetched.get("text"):
        return []
    
    xml = fetched["text"]
    try:
        root = ET.fromstring(xml)
        # Check if this is a sitemap index
        if root.tag.lower().endswith("sitemapindex") or any(c.tag.lower().endswith("sitemap") for c in root):
            # Recursively fetch nested sitemaps
            for loc in root.findall(".//{*}loc"):
                if loc.text and len(urls) < max_urls:
                    nested_urls = await discover_sitemap(session, loc.text.strip(), max_urls - len(urls), nested-1, visited)
                    urls.extend(nested_urls)
        else:
            # Extract URLs from urlset
            for loc in root.findall(".//{*}loc"):
                if loc.text:
                    url = loc.text.strip()
                    if is_likely_article(url):
                        urls.append(url)
                    if len(urls) >= max_urls:
                        break
    except ET.ParseError:
        # Fallback: regex extraction
        raw_urls = re.findall(r"<loc>(.*?)</loc>", xml)
        urls.extend([u for u in raw_urls if is_likely_article(u)])
    
    return list(dict.fromkeys(urls))[:max_urls]

async def discover_category(session, url, js=False, driver_pool=None, depth=0, visited=None, per_site_cap=MAX_URLS_PER_SITE):
    """Crawl category/archive pages"""
    if visited is None: 
        visited = set()
    if depth > 10 or url in visited or len(visited) > per_site_cap:
        return []
    visited.add(url)
    
    html = ""
    if js and driver_pool:
        r = await fetch_selenium(driver_pool, url, wait_seconds=2.0)
        html = r["text"] if r else ""
    else:
        r = await fetch_aio(session, url)
        html = r["text"] if r else ""
    
    if not html:
        return []
    
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception as e:
        logger.debug(f"Category parse error {url} -> {e}")
        return []
    
    base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    urls = []
    
    # Extract all links
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Normalize URL
        if href.startswith("/"):
            href = urljoin(base_url, href)
        elif not href.startswith("http"):
            continue
        
        # Clean fragment
        href = href.split("#")[0].strip()
        
        # Check if article-like
        if is_likely_article(href):
            urls.append(href)
            if len(urls) >= per_site_cap:
                break
    
    # Follow pagination (next/older/page links)
    if len(urls) < per_site_cap and depth < 10:
        pagination_links = []
        for a in soup.find_all("a", href=True):
            text = (a.get_text() or "").lower()
            if any(word in text for word in ["next", "older", "previous", "more", "load"]):
                href = a["href"]
                if href.startswith("/"):
                    href = urljoin(base_url, href)
                if href.startswith("http") and href not in visited:
                    pagination_links.append(href)
        
        # Limit pagination crawling
        for plink in pagination_links[:5]:
            if len(urls) >= per_site_cap:
                break
            child_urls = await discover_category(session, plink, js, driver_pool, depth+1, visited, per_site_cap)
            urls.extend(child_urls)
    
    return list(dict.fromkeys(urls))[:per_site_cap]

# -----------------------------
# ARTICLE PROCESS - FIXED CONTENT EXTRACTION
# -----------------------------
async def process_article(url: str, session: aiohttp.ClientSession, driver_pool=None, counters=None, seen_hashes=None, seen_urls=None):
    try:
        counters["attempted"] += 1
        
        # Check if URL already scraped
        if seen_urls is not None and url in seen_urls:
            counters["duplicate_url"] += 1
            return None
        
        fetched = await fetch_aio(session, url)
        html_text = fetched["text"] if fetched else ""
        
        # Selenium fallback if needed
        if driver_pool and (not html_text or len(html_text) < 1000):
            r = await fetch_selenium(driver_pool, url, wait_seconds=2.0)
            html_text = r["text"] if r else html_text
            if r: 
                counters["selenium_used"] += 1
        
        if not html_text or not html_text.strip().startswith("<"):
            counters["failed_fetch"] += 1
            return None
        
        # Save cache
        save_html_cache(url, html_text)
        
        try:
            soup = BeautifulSoup(html_text, "lxml")
        except Exception as e:
            counters["parse_errors"] += 1
            logger.debug(f"Parse error for {url}: {e}")
            return None
        
        # IMPROVED: Extract content using multiple strategies
        content = extract_article_content(soup)
        content_length = len(content)
        
        # Check min length
        if content_length < MIN_CONTENT_CHARS:
            counters["too_short"] += 1
            return None
        
        # Check max length
        if content_length > MAX_CONTENT_CHARS:
            counters["too_long"] += 1
            return None
        
        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        
        # Check keywords
        text_combined = (title + " " + content).strip()
        matches = keyword_match(text_combined)
        
        if not any(matches.values()):
            counters["filtered_non_ai"] += 1
            return None
        
        # Check duplicate content
        content_hash = sha256(content)
        if seen_hashes is not None and content_hash in seen_hashes:
            counters["duplicate_content"] += 1
            return None
        
        # Save record with content_length field
        record = {
            "id": sha256(url),
            "url": url,
            "domain": urlparse(url).netloc,
            "title": title,
            "content": content,
            "content_length": content_length,
            "content_hash": content_hash,
            "keywords_found": {k: v for k, v in matches.items() if v},
            "scraped_at": now_iso()
        }
        save_jsonl(record)
        
        # Mark as seen
        if seen_hashes is not None: 
            seen_hashes.add(content_hash)
        if seen_urls is not None:
            seen_urls.add(url)
            
        counters["saved"] += 1
        
        # Log progress every 100 articles
        if counters["saved"] % 100 == 0:
            logger.info(f"Progress: {counters['saved']} articles saved, {counters['attempted']} attempted")
        
        return record
    except Exception as e:
        counters["exceptions"] += 1
        logger.debug(f"process_article exception {url} -> {e}")
        return None

# -----------------------------
# MAIN
# -----------------------------
async def main():
    counters = {
        "discovered": 0,
        "attempted": 0,
        "selenium_used": 0,
        "failed_fetch": 0,
        "parse_errors": 0,
        "too_short": 0,
        "too_long": 0,
        "filtered_non_ai": 0,
        "duplicate_url": 0,
        "duplicate_content": 0,
        "saved": 0,
        "exceptions": 0
    }

    # Load existing data to avoid duplicates
    existing_hashes, existing_urls = load_existing_hashes()
    seen_hashes = existing_hashes.copy()
    seen_urls = existing_urls.copy()

    # FIXED: Use driver pool instead of single driver
    driver_pool = None
    try:
        driver_pool = SeleniumDriverPool(pool_size=3)
        await driver_pool.init()
    except Exception as e:
        logger.warning(f"Selenium driver pool not created: {e}")
        driver_pool = None

    # FIXED: Better connection pooling configuration
    connector = aiohttp.TCPConnector(
        limit=CONCURRENT_REQUESTS,  # Total connections
        limit_per_host=10,  # Max per host to avoid overwhelming servers
        ttl_dns_cache=300,  # DNS cache TTL
        force_close=False,  # Reuse connections
        enable_cleanup_closed=True  # Clean up closed connections
    )
    timeout = aiohttp.ClientTimeout(total=20, connect=10)
    discovered = []
    per_site_counts = {}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # DISCOVERY PHASE
        logger.info(f"Starting discovery for {len(SITES)} sites...")
        
        for idx, site in enumerate(SITES, 1):
            name = site.get("domain") or site.get("name") or "unknown"
            typ = site.get("type")
            js = site.get("js", False)
            logger.info(f"[{idx}/{len(SITES)}] Discovering {name} (type={typ}, js={js})")
            
            urls = []
            try:
                if typ == "rss" and site.get("feed"):
                    urls = discover_rss(site["feed"], MAX_URLS_PER_SITE)
                elif typ == "sitemap" and site.get("feed"):
                    urls = await discover_sitemap(session, site["feed"], MAX_URLS_PER_SITE)
                elif typ == "category" and site.get("url"):
                    urls = await discover_category(session, site["url"], js, driver_pool, per_site_cap=MAX_URLS_PER_SITE)
                else:
                    logger.debug(f"Skipping {name}: missing fields")
                    continue
            except Exception as e:
                logger.warning(f"Discovery error for {name}: {e}")
            
            # Filter and dedupe
            final_urls = []
            for u in urls:
                if not isinstance(u, str): 
                    continue
                u = u.split("#")[0].strip()
                # Skip if already scraped
                if u in existing_urls:
                    continue
                if u.startswith("http") and is_likely_article(u):
                    final_urls.append(u)
                if len(final_urls) >= MAX_URLS_PER_SITE:
                    break
            
            final_urls = list(dict.fromkeys(final_urls))[:MAX_URLS_PER_SITE]
            per_site_counts[name] = len(final_urls)
            discovered.extend(final_urls)
            counters["discovered"] += len(final_urls)
            logger.info(f"  → Found {len(final_urls)} NEW candidate URLs from {name}")

        # Global dedupe
        discovered = list(dict.fromkeys(discovered))
        logger.info(f"\n{'='*60}")
        logger.info(f"Total discovered URLs (excluding already scraped): {len(discovered)}")
        logger.info(f"Previously scraped articles: {len(existing_urls)}")
        logger.info(f"{'='*60}\n")
        
        # Save discovered URLs
        with open(DISCOVERED_FILE, "w", encoding="utf8") as fh:
            for u in discovered:
                fh.write(u + "\n")

        # SCRAPING PHASE
        sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
        
        async def worker(u):
            async with sem:
                await process_article(u, session, driver_pool, counters, seen_hashes, seen_urls)
                await asyncio.sleep(0.1)  # Increased delay for stability

        tasks = [worker(u) for u in discovered[:MAX_URLS]]
        logger.info(f"Starting scrape of {len(tasks)} URLs (concurrency={CONCURRENT_REQUESTS})...")
        
        # FIXED: Process in smaller batches with progress updates
        BATCH = 1000
        for i in range(0, len(tasks), BATCH):
            batch = tasks[i:i+BATCH]
            batch_num = i//BATCH + 1
            total_batches = (len(tasks)-1)//BATCH + 1
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} URLs)...")
            await asyncio.gather(*batch)
            logger.info(f"Batch {batch_num} complete. Saved so far: {counters['saved']}")

    # Cleanup
    if driver_pool:
        driver_pool.cleanup()

    # Print diagnostics
    logger.info("\n" + "="*60)
    logger.info("SCRAPE DIAGNOSTICS")
    logger.info("="*60)
    logger.info(json.dumps(counters, indent=2))
    logger.info(f"Output file: {OUT_JSONL}")
    logger.info(f"Discovered URLs list: {DISCOVERED_FILE}")
    
    print("\n" + "="*60)
    print("SCRAPE DIAGNOSTICS")
    print("="*60)
    for k, v in counters.items():
        print(f"{k:20s}: {v:,}")
    print(f"\nTotal saved (this run): {counters['saved']:,}")
    print(f"Previously scraped: {len(existing_urls):,}")
    print(f"Total articles in dataset: {len(existing_urls) + counters['saved']:,}")
    print(f"Output file: {OUT_JSONL}")

if __name__ == "__main__":
    asyncio.run(main())