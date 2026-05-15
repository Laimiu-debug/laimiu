"""Web search and fetch tools."""

from __future__ import annotations

import json
from typing import Any

from laimiu.tools.base import BaseTool, ToolResult


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web for information"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (default 5)",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query:
            return ToolResult(success=False, error="No query provided")

        try:
            import urllib.request
            import urllib.parse

            # Use DuckDuckGo HTML search as a simple fallback
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Simple HTML parsing for results
            import re
            results = []
            # Extract result titles and URLs from DuckDuckGo HTML
            for match in re.finditer(
                r'<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>', html
            ):
                link = match.group(1)
                title = re.sub(r"<.*?>", "", match.group(2)).strip()
                if title and link:
                    results.append(f"- {title}\n  {link}")
                    if len(results) >= max_results:
                        break

            if not results:
                return ToolResult(success=True, output=f"No results found for: {query}")

            return ToolResult(
                success=True,
                output=f"Search results for '{query}':\n\n" + "\n\n".join(results),
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Web search failed: {e}")


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = "Fetch and read the content of a web page"
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters to return (default 5000)",
                "default": 5000,
            },
        },
        "required": ["url"],
    }

    async def execute(self, **kwargs: Any) -> ToolResult:
        url = kwargs.get("url", "")
        max_chars = kwargs.get("max_chars", 5000)

        if not url:
            return ToolResult(success=False, error="No URL provided")

        try:
            import urllib.request
            import re

            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode("utf-8", errors="replace")

            # Strip HTML tags for a readable version
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > max_chars:
                text = text[:max_chars] + "..."

            return ToolResult(success=True, output=text)
        except Exception as e:
            return ToolResult(success=False, error=f"Fetch failed: {e}")
