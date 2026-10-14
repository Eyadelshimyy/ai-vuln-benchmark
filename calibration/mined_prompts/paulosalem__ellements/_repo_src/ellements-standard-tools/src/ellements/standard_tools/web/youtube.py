"""YouTube search and transcript functionality."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Literal, cast
from urllib.parse import parse_qs, urlparse

from ellements.core import ToolRegistry
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi

# Fixes for youtube-search-python compatibility issues:
# - The library uses the old 'proxies' parameter which httpx no longer supports.
# - Some YouTube search results omit channel browse IDs, but the library assumes
#   they are always present and crashes while building result objects.
try:
    import httpx
    from youtubesearchpython.core.constants import userAgent
    from youtubesearchpython.core.requests import RequestCore
    from youtubesearchpython.handlers.componenthandler import (
        ComponentHandler,
        videoElementKey,
    )

    # Store original methods
    _original_sync_post = RequestCore.syncPostRequest
    _original_sync_get = RequestCore.syncGetRequest

    def _safe_youtube_identifier(value: Any) -> str:
        """Return a search result identifier only when it is a non-empty string."""
        return value if isinstance(value, str) and value else ""

    def _build_youtube_link(prefix: str, identifier: Any) -> str:
        """Build a YouTube link without crashing on missing identifiers."""
        safe_identifier = _safe_youtube_identifier(identifier)
        return f"{prefix}{safe_identifier}" if safe_identifier else ""

    # Create patched versions that work with httpx
    def _patched_sync_post(self: Any) -> httpx.Response:
        """Patched syncPostRequest that works with httpx."""
        if self.proxy:
            # Use Client with proxy
            with httpx.Client(proxy=self.proxy) as client:
                return client.post(
                    self.url,
                    headers={"User-Agent": userAgent},
                    json=self.data,
                    timeout=self.timeout
                )
        else:
            # No proxy, use direct call
            return httpx.post(
                self.url,
                headers={"User-Agent": userAgent},
                json=self.data,
                timeout=self.timeout
            )

    def _patched_sync_get(self: Any) -> httpx.Response:
        """Patched syncGetRequest that works with httpx."""
        if self.proxy:
            # Use Client with proxy
            with httpx.Client(proxy=self.proxy) as client:
                return client.get(
                    self.url,
                    headers={"User-Agent": userAgent},
                    timeout=self.timeout,
                    cookies={'CONSENT': 'YES+1'}
                )
        else:
            # No proxy, use direct call
            return httpx.get(
                self.url,
                headers={"User-Agent": userAgent},
                timeout=self.timeout,
                cookies={'CONSENT': 'YES+1'}
            )

    def _patched_get_video_component(
        self: ComponentHandler,
        element: dict[str, Any],
        shelfTitle: str | None = None,
    ) -> dict[str, Any]:
        """Patched video component builder that tolerates missing channel IDs."""
        video = element[videoElementKey]
        video_id = _safe_youtube_identifier(self._getValue(video, ['videoId']))
        channel_id = _safe_youtube_identifier(
            self._getValue(
                video,
                ['ownerText', 'runs', 0, 'navigationEndpoint', 'browseEndpoint', 'browseId'],
            )
        )

        component = {
            'type': 'video',
            'id': video_id,
            'title': self._getValue(video, ['title', 'runs', 0, 'text']),
            'publishedTime': self._getValue(video, ['publishedTimeText', 'simpleText']),
            'duration': self._getValue(video, ['lengthText', 'simpleText']),
            'viewCount': {
                'text': self._getValue(video, ['viewCountText', 'simpleText']),
                'short': self._getValue(video, ['shortViewCountText', 'simpleText']),
            },
            'thumbnails': self._getValue(video, ['thumbnail', 'thumbnails']),
            'richThumbnail': self._getValue(
                video,
                ['richThumbnail', 'movingThumbnailRenderer', 'movingThumbnailDetails', 'thumbnails', 0],
            ),
            'descriptionSnippet': self._getValue(
                video,
                ['detailedMetadataSnippets', 0, 'snippetText', 'runs'],
            ),
            'channel': {
                'name': self._getValue(video, ['ownerText', 'runs', 0, 'text']),
                'id': channel_id,
                'thumbnails': self._getValue(
                    video,
                    [
                        'channelThumbnailSupportedRenderers',
                        'channelThumbnailWithLinkRenderer',
                        'thumbnail',
                        'thumbnails',
                    ],
                ),
            },
            'accessibility': {
                'title': self._getValue(
                    video,
                    ['title', 'accessibility', 'accessibilityData', 'label'],
                ),
                'duration': self._getValue(
                    video,
                    ['lengthText', 'accessibility', 'accessibilityData', 'label'],
                ),
            },
        }
        component['link'] = _build_youtube_link(
            'https://www.youtube.com/watch?v=',
            video_id,
        )
        component['channel']['link'] = _build_youtube_link(
            'https://www.youtube.com/channel/',
            channel_id,
        )
        component['shelfTitle'] = shelfTitle
        return component

    # Apply patches
    RequestCore.syncPostRequest = _patched_sync_post
    RequestCore.syncGetRequest = _patched_sync_get
    ComponentHandler._getVideoComponent = _patched_get_video_component

except ImportError:
    pass  # If import fails, continue without patching

from ellements.core.chunking import TextProcessor
from ellements.core.exceptions import LLMError
from youtubesearchpython import (
    CustomSearch,
    Video,
    VideoDurationFilter,
    VideoSortOrder,
    VideosSearch,
    VideoUploadDateFilter,
)


class VideoMetadata(BaseModel):
    """Metadata for a YouTube video."""
    video_id: str = Field(description="YouTube video ID")
    title: str = Field(description="Video title")
    channel: str = Field(description="Channel name")
    channel_id: str = Field(default="", description="Channel ID")
    description: str = Field(default="", description="Video description")
    duration: str = Field(default="", description="Video duration")
    view_count: int = Field(default=0, description="Number of views")
    publish_date: str = Field(default="", description="Publish date")
    thumbnail_url: str = Field(default="", description="Thumbnail URL")
    url: str = Field(description="Full YouTube URL")


class TranscriptSegment(BaseModel):
    """A single segment of a video transcript."""
    text: str = Field(description="Text content of the segment")
    start: float = Field(description="Start time in seconds")
    duration: float = Field(description="Duration in seconds")


class VideoTranscript(BaseModel):
    """Full transcript for a YouTube video."""
    video_id: str = Field(description="YouTube video ID")
    language: str = Field(description="Language code of the transcript")
    is_generated: bool = Field(default=False, description="Whether transcript is auto-generated")
    segments: list[TranscriptSegment] = Field(description="List of transcript segments")

    def get_full_text(self, separator: str = " ") -> str:
        """Get the full transcript as a single text string.

        Args:
            separator: Separator between segments

        Returns:
            Complete transcript text
        """
        return separator.join(segment.text for segment in self.segments)

    def get_text_at_time(self, timestamp: float, window: float = 5.0) -> str:
        """Get transcript text around a specific timestamp.

        Args:
            timestamp: Time in seconds
            window: Time window in seconds (before and after)

        Returns:
            Text around the specified timestamp
        """
        relevant_segments = [
            seg for seg in self.segments
            if seg.start <= timestamp <= seg.start + seg.duration + window
            or timestamp - window <= seg.start <= timestamp + window
        ]
        return " ".join(seg.text for seg in relevant_segments)


class SearchResult(BaseModel):
    """Represents a single YouTube search result."""
    video_id: str = Field(description="YouTube video ID")
    title: str = Field(description="Video title")
    channel: str = Field(description="Channel name")
    duration: str = Field(default="", description="Video duration")
    view_count: str = Field(default="", description="View count as string")
    publish_time: str = Field(default="", description="Time since published")
    thumbnail_url: str = Field(default="", description="Thumbnail URL")
    url: str = Field(description="Full YouTube URL")
    description: str = Field(default="", description="Video description snippet")


class SearchResults(BaseModel):
    """Container for multiple YouTube search results."""
    query: str = Field(description="The search query")
    results: list[SearchResult] = Field(description="List of search results")
    total_results: int = Field(description="Total number of results returned")

    def __len__(self) -> int:
        """Return the number of results."""
        return len(self.results)

    def __iter__(self) -> Any:
        """Allow iteration over results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> SearchResult:
        """Allow indexing of results."""
        return self.results[index]


def _format_transcript_unavailable_message(
    video_url_or_id: str,
    error: Exception,
) -> str:
    """Build a model-friendly result for transcript fetch failures."""
    reason = _extract_transcript_error_reason(error)
    return (
        "TRANSCRIPT_UNAVAILABLE\n"
        f"Video: {video_url_or_id}\n"
        f"Reason: {reason}\n"
        "Do not stop the task. Continue with metadata and other search results, "
        "and prefer videos whose transcripts you can verify. If this looks like "
        "throttling, increase the delay with set_transcript_delay before retrying."
    )


def _extract_transcript_error_reason(error: Exception) -> str:
    """Extract the most useful human-readable reason from a transcript error."""
    candidates = [
        getattr(error, "cause", None),
        str(error),
    ]
    for candidate in candidates:
        text = str(candidate or "").strip()
        if text:
            return text
    return error.__class__.__name__


def _is_transcript_rate_limited(error: Exception) -> bool:
    """Return whether a transcript error indicates YouTube-side throttling."""
    haystack = _extract_transcript_error_reason(error).lower()
    return "429" in haystack or "too many requests" in haystack


def _format_transcript_rate_limited_message(
    video_url_or_id: str,
    error: Exception,
) -> str:
    """Build a model-friendly result for active transcript rate limiting."""
    reason = _extract_transcript_error_reason(error)
    return (
        "TRANSCRIPT_RATE_LIMITED\n"
        f"Video: {video_url_or_id}\n"
        f"Reason: {reason}\n"
        "YouTube is currently blocking transcript requests from this environment. "
        "Do not retry transcript fetches again in this run. Continue with metadata "
        "and other search results instead. Increasing set_transcript_delay can help "
        "avoid future throttling, but an active 429 block usually requires waiting "
        "15-30 minutes or using proxy rotation before transcripts work again."
    )


def _coerce_text_field(value: Any) -> str:
    """Normalize optional text fields returned by YouTube search libraries."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return ""


class YouTubeSearcher:
    """YouTube search and transcript searcher."""

    def __init__(
        self,
        language: str = "en",
        region: str = "US",
        **kwargs: Any
    ) -> None:
        """Initialize the YouTube searcher.

        Args:
            language: Preferred language for searches and transcripts
            region: Region for search results
            **kwargs: Additional configuration
        """
        self.language = language
        self.region = region
        self.config = kwargs

    @staticmethod
    def extract_video_id(url_or_id: str) -> str:
        """Extract video ID from a YouTube URL or return the ID if already extracted.

        Args:
            url_or_id: YouTube URL or video ID

        Returns:
            Video ID

        Raises:
            ValueError: If URL is invalid or ID cannot be extracted
        """
        # If it's already a video ID (11 characters, alphanumeric with _ and -)
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url_or_id):
            return url_or_id

        # Parse various YouTube URL formats
        parsed_url = urlparse(url_or_id)

        # youtube.com/watch?v=VIDEO_ID
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com'):
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                if 'v' in query_params:
                    return query_params['v'][0]
            # youtube.com/embed/VIDEO_ID
            elif parsed_url.path.startswith('/embed/') or parsed_url.path.startswith('/v/'):
                return parsed_url.path.split('/')[2]

        # youtu.be/VIDEO_ID
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path.lstrip('/')

        raise ValueError(f"Could not extract video ID from: {url_or_id}")

    @staticmethod
    def _build_search_preferences(
        duration: str | None = None,
        upload_date: str | None = None,
        sort_by: str | None = None,
    ) -> str:
        """Build a search preferences filter string for CustomSearch.

        Args:
            duration: "short" (< 4 min) or "long" (> 20 min)
            upload_date: "hour", "today", "week", "month", "year"
            sort_by: "relevance", "date", "views", "rating"

        Returns:
            Combined filter string, or empty string if no filters
        """
        parts = []

        if duration:
            duration_map = {
                "short": VideoDurationFilter.short,
                "long": VideoDurationFilter.long,
            }
            if duration.lower() in duration_map:
                parts.append(duration_map[duration.lower()])

        if upload_date:
            date_map = {
                "hour": VideoUploadDateFilter.lastHour,
                "today": VideoUploadDateFilter.today,
                "week": VideoUploadDateFilter.thisWeek,
                "month": VideoUploadDateFilter.thisMonth,
                "year": VideoUploadDateFilter.thisYear,
            }
            if upload_date.lower() in date_map:
                parts.append(date_map[upload_date.lower()])

        if sort_by:
            sort_map = {
                "relevance": VideoSortOrder.relevance,
                "date": VideoSortOrder.uploadDate,
                "views": VideoSortOrder.viewCount,
                "rating": VideoSortOrder.rating,
            }
            sort_key = sort_by.lower()
            if sort_key == "relevance":
                # Relevance is the default ordering for VideosSearch, so avoid
                # sending it through CustomSearch unless other filters require it.
                pass
            elif sort_key in sort_map:
                parts.append(sort_map[sort_key])

        return "".join(parts)

    def search(
        self,
        query: str,
        max_results: int = 10,
        duration: str | None = None,
        upload_date: str | None = None,
        sort_by: str | None = None,
        **kwargs: Any,
    ) -> SearchResults:
        """Search for YouTube videos.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            duration: Filter by duration - "short" (< 4 min) or "long" (> 20 min)
            upload_date: Filter by upload date - "hour", "today", "week", "month", "year"
            sort_by: Sort order - "relevance", "date", "views", "rating"
            **kwargs: Additional search parameters

        Returns:
            SearchResults object containing the results
        """
        try:
            # Build search preferences filter string if any filters specified
            search_preferences = self._build_search_preferences(
                duration=duration, upload_date=upload_date, sort_by=sort_by
            )

            if search_preferences:
                # Use CustomSearch when filters are specified
                videos_search = CustomSearch(
                    query, search_preferences, limit=max_results
                )
            else:
                # Use simple VideosSearch when no filters
                videos_search = VideosSearch(query, limit=max_results)

            raw_results = videos_search.result()

            # Convert to SearchResult objects
            results = []
            if 'result' in raw_results:
                for video in raw_results['result']:
                    search_result = SearchResult(
                        video_id=_coerce_text_field(video.get('id', '')),
                        title=_coerce_text_field(video.get('title', '')),
                        channel=_coerce_text_field(
                            video.get('channel', {}).get('name', '')
                        ),
                        duration=_coerce_text_field(video.get('duration', '')),
                        view_count=_coerce_text_field(
                            video.get('viewCount', {}).get('short', '')
                        ),
                        publish_time=_coerce_text_field(
                            video.get('publishedTime', '')
                        ),
                        thumbnail_url=(
                            _coerce_text_field(
                                video.get('thumbnails', [{}])[0].get('url', '')
                            )
                            if video.get('thumbnails')
                            else ''
                        ),
                        url=_coerce_text_field(video.get('link', '')),
                        description=(
                            _coerce_text_field(
                                video.get('descriptionSnippet', [{}])[0].get(
                                    'text', ''
                                )
                            )
                            if video.get('descriptionSnippet')
                            else ''
                        ),
                    )
                    results.append(search_result)

            return SearchResults(
                query=query,
                results=results,
                total_results=len(results)
            )

        except Exception as e:
            raise LLMError(f"YouTube search failed: {e}") from e

    async def search_async(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> SearchResults:
        """Async version of search.

        Args:
            query: Search query string
            max_results: Maximum number of results
            **kwargs: Additional parameters

        Returns:
            SearchResults object
        """
        # Run synchronous search in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.search(query=query, max_results=max_results, **kwargs)
        )

    def get_video_metadata(
        self,
        video_url_or_id: str,
        **kwargs: Any,
    ) -> VideoMetadata:
        """Get detailed metadata for a YouTube video.

        Args:
            video_url_or_id: YouTube URL or video ID
            **kwargs: Additional parameters

        Returns:
            VideoMetadata object
        """
        try:
            video_id = self.extract_video_id(video_url_or_id)

            # Filter kwargs to only include supported parameters for Video.getInfo
            # Video.getInfo accepts: mode, timeout
            video_kwargs = {}
            supported_params = {'mode', 'timeout'}
            for key, value in kwargs.items():
                if key in supported_params:
                    video_kwargs[key] = value

            # Use Video class to get detailed info
            video_info = Video.getInfo(f"https://www.youtube.com/watch?v={video_id}", **video_kwargs)

            return VideoMetadata(
                video_id=video_id,
                title=video_info.get('title', ''),
                channel=video_info.get('channel', {}).get('name', ''),
                channel_id=video_info.get('channel', {}).get('id', ''),
                description=video_info.get('description', ''),
                duration=video_info.get('duration', {}).get('secondsText', ''),
                view_count=int(video_info.get('viewCount', {}).get('text', '0').replace(',', '').replace(' views', '').split()[0]) if video_info.get('viewCount') else 0,
                publish_date=video_info.get('publishDate', ''),
                thumbnail_url=video_info.get('thumbnails', [{}])[-1].get('url', '') if video_info.get('thumbnails') else '',
                url=f"https://www.youtube.com/watch?v={video_id}"
            )

        except Exception as e:
            raise LLMError(f"Failed to get video metadata: {e}") from e

    async def get_video_metadata_async(
        self,
        video_url_or_id: str,
        **kwargs: Any,
    ) -> VideoMetadata:
        """Async version of get_video_metadata.

        Args:
            video_url_or_id: YouTube URL or video ID
            **kwargs: Additional parameters

        Returns:
            VideoMetadata object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_video_metadata(video_url_or_id, **kwargs)
        )

    def get_transcript(
        self,
        video_url_or_id: str,
        languages: list[str] | None = None,
        preserve_formatting: bool = False,
        **kwargs: Any,
    ) -> VideoTranscript:
        """Get transcript for a YouTube video.

        Args:
            video_url_or_id: YouTube URL or video ID
            languages: Preferred languages (e.g., ['en', 'es'])
            preserve_formatting: Preserve line breaks and formatting
            **kwargs: Additional parameters

        Returns:
            VideoTranscript object

        Raises:
            LLMError: If transcript cannot be fetched
        """
        try:
            video_id = self.extract_video_id(video_url_or_id)

            # Default to instance language if not specified
            if languages is None:
                languages = [self.language]

            # Create API instance and fetch transcript
            api = YouTubeTranscriptApi()

            # Try to fetch transcript (tries all available transcripts)
            fetched_transcript = api.fetch(video_id, **kwargs)

            # Convert snippets to TranscriptSegment objects
            segments = []
            for snippet in fetched_transcript.snippets:
                segments.append(TranscriptSegment(
                    text=snippet.text,
                    start=snippet.start,
                    duration=snippet.duration
                ))

            return VideoTranscript(
                video_id=video_id,
                language=fetched_transcript.language_code,
                is_generated=fetched_transcript.is_generated,
                segments=segments
            )

        except Exception as e:
            raise LLMError(f"Failed to fetch transcript: {e}") from e

    async def get_transcript_async(
        self,
        video_url_or_id: str,
        languages: list[str] | None = None,
        preserve_formatting: bool = False,
        **kwargs: Any,
    ) -> VideoTranscript:
        """Async version of get_transcript.

        Args:
            video_url_or_id: YouTube URL or video ID
            languages: Preferred languages
            preserve_formatting: Preserve formatting
            **kwargs: Additional parameters

        Returns:
            VideoTranscript object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_transcript(
                video_url_or_id,
                languages=languages,
                preserve_formatting=preserve_formatting,
                **kwargs
            )
        )

    def get_video_with_transcript(
        self,
        video_url_or_id: str,
        languages: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[VideoMetadata, VideoTranscript]:
        """Get both metadata and transcript for a video.

        Args:
            video_url_or_id: YouTube URL or video ID
            languages: Preferred languages for transcript
            **kwargs: Additional parameters

        Returns:
            Tuple of (VideoMetadata, VideoTranscript)
        """
        metadata = self.get_video_metadata(video_url_or_id, **kwargs)
        transcript = self.get_transcript(video_url_or_id, languages=languages, **kwargs)
        return metadata, transcript

    async def get_video_with_transcript_async(
        self,
        video_url_or_id: str,
        languages: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[VideoMetadata, VideoTranscript]:
        """Async version of get_video_with_transcript.

        Args:
            video_url_or_id: YouTube URL or video ID
            languages: Preferred languages
            **kwargs: Additional parameters

        Returns:
            Tuple of (VideoMetadata, VideoTranscript)
        """
        # Run both operations concurrently
        metadata_task = self.get_video_metadata_async(video_url_or_id, **kwargs)
        transcript_task = self.get_transcript_async(video_url_or_id, languages=languages, **kwargs)

        metadata, transcript = await asyncio.gather(metadata_task, transcript_task)
        return metadata, transcript

    def search_and_get_transcripts(
        self,
        query: str,
        max_results: int = 5,
        languages: list[str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[SearchResult, VideoTranscript]]:
        """Search for videos and get their transcripts.

        Args:
            query: Search query
            max_results: Maximum number of results
            languages: Preferred languages for transcripts
            **kwargs: Additional parameters

        Returns:
            List of tuples (SearchResult, VideoTranscript)
        """
        search_results = self.search(query, max_results=max_results, **kwargs)

        results_with_transcripts = []
        for result in search_results.results:
            try:
                transcript = self.get_transcript(result.video_id, languages=languages)
                results_with_transcripts.append((result, transcript))
            except Exception:
                # Skip videos without transcripts
                continue

        return results_with_transcripts

    async def search_and_get_transcripts_async(
        self,
        query: str,
        max_results: int = 5,
        languages: list[str] | None = None,
        max_concurrent: int = 3,
        **kwargs: Any,
    ) -> list[tuple[SearchResult, VideoTranscript]]:
        """Async version of search_and_get_transcripts.

        Args:
            query: Search query
            max_results: Maximum number of results
            languages: Preferred languages
            max_concurrent: Maximum concurrent transcript fetches
            **kwargs: Additional parameters

        Returns:
            List of tuples (SearchResult, VideoTranscript)
        """
        search_results = await self.search_async(query, max_results=max_results, **kwargs)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_transcript_safe(result: SearchResult) -> tuple[SearchResult, VideoTranscript] | None:
            async with semaphore:
                try:
                    transcript = await self.get_transcript_async(result.video_id, languages=languages)
                    return (result, transcript)
                except Exception:
                    return None

        tasks = [fetch_transcript_safe(result) for result in search_results.results]
        results = await asyncio.gather(*tasks)

        # Filter out None results (videos without transcripts)
        return [r for r in results if r is not None]


class RateLimitedYouTubeSearcher(YouTubeSearcher):
    """YouTube searcher with rate limiting, retry logic, and proxy support.

    This class extends YouTubeSearcher to prevent IP blocking by:
    1. Adding configurable delays between transcript fetches
    2. Implementing retry logic with exponential backoff
    3. Supporting proxy configuration
    4. Optional transcript caching

    Example:
        >>> # Basic rate limiting (2 second delay between requests)
        >>> searcher = RateLimitedYouTubeSearcher(
        ...     rate_limit_delay=2.0,
        ...     max_retries=3
        ... )
        >>>
        >>> # With proxy support (requires youtube-transcript-api with proxy support)
        >>> from youtube_transcript_api.proxies import WebshareProxyConfig
        >>> proxy_config = WebshareProxyConfig(
        ...     proxy_username="user",
        ...     proxy_password="pass"
        ... )
        >>> searcher = RateLimitedYouTubeSearcher(
        ...     rate_limit_delay=2.0,
        ...     proxy_config=proxy_config
        ... )
    """

    def __init__(
        self,
        language: str = "en",
        region: str = "US",
        rate_limit_delay: float = 1.5,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        rate_limit_cooldown_seconds: float = 900.0,
        enable_cache: bool = True,
        cache_size: int = 128,
        proxy_config: Any | None = None,
        progress_callback: Callable[[str], None] | None = None,
        **kwargs: Any
    ) -> None:
        """Initialize the rate-limited YouTube searcher.

        Args:
            language: Preferred language for searches and transcripts
            region: Region for search results
            rate_limit_delay: Delay in seconds between transcript requests (default: 1.5s)
            max_retries: Maximum number of retry attempts (default: 3)
            retry_backoff_factor: Exponential backoff multiplier (default: 2.0)
            rate_limit_cooldown_seconds: How long to stop retrying after YouTube returns
                a transcript 429 block (default: 900s / 15 minutes)
            enable_cache: Enable transcript caching to avoid redundant requests
            cache_size: Maximum number of cached transcripts (default: 128)
            proxy_config: Optional proxy configuration (WebshareProxyConfig or GenericProxyConfig)
            progress_callback: Optional callback for progress updates (e.g., rate limit waits)
            **kwargs: Additional configuration
        """
        super().__init__(language=language, region=region, **kwargs)

        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.rate_limit_cooldown_seconds = max(0.0, rate_limit_cooldown_seconds)
        self.enable_cache = enable_cache
        self.proxy_config = proxy_config
        self.progress_callback = progress_callback

        # Track last request time for rate limiting
        self._last_transcript_request = 0.0
        self._transcript_rate_limited_until = 0.0
        self._transcript_rate_limit_reason: str | None = None

        # Create cached version of get_transcript if caching enabled
        if enable_cache:
            self._cached_get_transcript = lru_cache(maxsize=cache_size)(
                self._get_transcript_impl
            )

    def _get_active_transcript_rate_limit_error(self) -> LLMError | None:
        """Return the active transcript rate-limit error, if still in cooldown."""
        if self._transcript_rate_limited_until <= 0:
            return None

        now = time.time()
        if now >= self._transcript_rate_limited_until:
            self._transcript_rate_limited_until = 0.0
            self._transcript_rate_limit_reason = None
            return None

        remaining_seconds = max(
            1,
            int(round(self._transcript_rate_limited_until - now)),
        )
        reason = self._transcript_rate_limit_reason or "HTTP 429 Too Many Requests"
        return LLMError(
            "Transcript requests are temporarily blocked by YouTube for this "
            f"environment. Wait about {remaining_seconds}s before retrying, or "
            f"use proxy rotation. Last error: {reason}"
        )

    def _mark_transcript_rate_limited(self, error: Exception) -> None:
        """Record an active transcript 429 block for the current environment."""
        if self.rate_limit_cooldown_seconds <= 0:
            return
        self._transcript_rate_limited_until = (
            time.time() + self.rate_limit_cooldown_seconds
        )
        self._transcript_rate_limit_reason = _extract_transcript_error_reason(error)

    def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limiting."""
        if self.rate_limit_delay <= 0:
            return

        time_since_last = time.time() - self._last_transcript_request
        if time_since_last < self.rate_limit_delay:
            wait_time = self.rate_limit_delay - time_since_last

            # Notify user of rate limit wait
            if self.progress_callback:
                self.progress_callback(f"⏳ Rate limit: waiting {wait_time:.1f}s before next request...")

            time.sleep(wait_time)

        self._last_transcript_request = time.time()

    def _get_transcript_impl(
        self,
        video_id: str,
        languages: tuple[str, ...] | None = None,
        preserve_formatting: bool = False,
    ) -> VideoTranscript:
        """Internal implementation for getting transcripts.

        This method is used for caching. Note: languages must be tuple for hashability.

        Args:
            video_id: YouTube video ID
            languages: Preferred languages as tuple
            preserve_formatting: Preserve line breaks

        Returns:
            VideoTranscript object
        """
        active_rate_limit_error = self._get_active_transcript_rate_limit_error()
        if active_rate_limit_error is not None:
            raise active_rate_limit_error

        # Wait for rate limit before making request
        self._wait_for_rate_limit()

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                # Create API instance with optional proxy
                if self.proxy_config:
                    api = YouTubeTranscriptApi(proxy_config=self.proxy_config)
                else:
                    api = YouTubeTranscriptApi()

                # Try to fetch transcript
                fetched_transcript = api.fetch(video_id)

                # Convert snippets to TranscriptSegment objects
                segments = []
                for snippet in fetched_transcript.snippets:
                    segments.append(TranscriptSegment(
                        text=snippet.text,
                        start=snippet.start,
                        duration=snippet.duration
                    ))

                self._transcript_rate_limited_until = 0.0
                self._transcript_rate_limit_reason = None
                return VideoTranscript(
                    video_id=video_id,
                    language=fetched_transcript.language_code,
                    is_generated=fetched_transcript.is_generated,
                    segments=segments
                )

            except Exception as e:
                last_exception = e

                if _is_transcript_rate_limited(e):
                    self._mark_transcript_rate_limited(e)

                    if self.progress_callback:
                        cooldown_seconds = max(
                            1,
                            int(round(self.rate_limit_cooldown_seconds)),
                        )
                        self.progress_callback(
                            "⚠️  YouTube is rate limiting transcript requests "
                            f"(HTTP 429). Skipping further transcript retries for "
                            f"about {cooldown_seconds}s."
                        )
                    break

                # If this is not the last attempt, wait before retrying
                if attempt < self.max_retries - 1:
                    # Exponential backoff: wait longer after each failure
                    backoff_time = self.rate_limit_delay * (self.retry_backoff_factor ** attempt)

                    # Notify user of retry
                    if self.progress_callback:
                        self.progress_callback(
                            f"⚠️  Retry attempt {attempt + 1}/{self.max_retries} failed. "
                            f"Waiting {backoff_time:.1f}s before retry {attempt + 2}..."
                        )

                    time.sleep(backoff_time)

        active_rate_limit_error = self._get_active_transcript_rate_limit_error()
        if active_rate_limit_error is not None:
            raise active_rate_limit_error from last_exception

        # All retries exhausted
        raise LLMError(
            f"Failed to fetch transcript after {self.max_retries} attempts: {last_exception}"
        ) from last_exception

    def get_transcript(
        self,
        video_url_or_id: str,
        languages: list[str] | None = None,
        preserve_formatting: bool = False,
        **kwargs: Any,
    ) -> VideoTranscript:
        """Get transcript for a YouTube video with rate limiting and retry logic.

        This method includes:
        - Rate limiting to prevent IP blocking
        - Exponential backoff retry on failures
        - Optional caching to avoid redundant requests
        - Optional proxy support

        Args:
            video_url_or_id: YouTube URL or video ID
            languages: Preferred languages (e.g., ['en', 'es'])
            preserve_formatting: Preserve line breaks and formatting
            **kwargs: Additional parameters

        Returns:
            VideoTranscript object

        Raises:
            LLMError: If transcript cannot be fetched after all retries
        """
        try:
            video_id = self.extract_video_id(video_url_or_id)

            # Default to instance language if not specified
            if languages is None:
                languages_tuple: tuple[str, ...] = (self.language,)
            else:
                # Convert to tuple for hashability (needed for caching)
                languages_tuple = tuple(languages)

            # Use cached version if caching enabled
            if self.enable_cache:
                return self._cached_get_transcript(
                    video_id=video_id,
                    languages=languages_tuple,
                    preserve_formatting=preserve_formatting
                )
            else:
                return self._get_transcript_impl(
                    video_id=video_id,
                    languages=languages_tuple,
                    preserve_formatting=preserve_formatting
                )

        except Exception as e:
            # Re-raise LLMError as-is, wrap others
            if isinstance(e, LLMError):
                raise
            raise LLMError(f"Failed to fetch transcript: {e}") from e

    async def get_transcript_async(
        self,
        video_url_or_id: str,
        languages: list[str] | None = None,
        preserve_formatting: bool = False,
        **kwargs: Any,
    ) -> VideoTranscript:
        """Async version of get_transcript with rate limiting.

        Args:
            video_url_or_id: YouTube URL or video ID
            languages: Preferred languages
            preserve_formatting: Preserve formatting
            **kwargs: Additional parameters

        Returns:
            VideoTranscript object
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.get_transcript(
                video_url_or_id,
                languages=languages,
                preserve_formatting=preserve_formatting,
                **kwargs
            )
        )

    def clear_cache(self) -> None:
        """Clear the transcript cache if caching is enabled."""
        if self.enable_cache and hasattr(self, '_cached_get_transcript'):
            self._cached_get_transcript.cache_clear()


# Agent Tools
# ---------------------------------------------------------------------------

def youtube_search_tools(
    text_processor: TextProcessor | None = None,
    max_transcript_tokens: int | None = None,
    truncation_strategy: str = "simple",
    rate_limit_delay: float = 1.5,
    max_retries: int = 3,
    enable_cache: bool = True,
    proxy_config: Any | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> ToolRegistry:
    """Create YouTube search and transcript tools for agents.

    These tools are returned as plain callables. Framework-specific backends
    adapt them when needed.

    By default, uses RateLimitedYouTubeSearcher to prevent IP blocking with:
    - 1.5 second delay between transcript requests
    - 3 retry attempts with exponential backoff
    - Transcript caching to avoid redundant requests

    Args:
        text_processor: Optional TextProcessor for handling long transcripts.
                       If not provided but max_transcript_tokens is set,
                       a default processor will be created.
        max_transcript_tokens: Maximum tokens for transcripts. If None, no truncation.
        truncation_strategy: Strategy to use ("simple", "map_reduce", "sequential")
        rate_limit_delay: Delay in seconds between transcript requests (default: 1.5).
                         Set to 0 to disable rate limiting.
        max_retries: Maximum retry attempts on failures (default: 3)
        enable_cache: Enable transcript caching (default: True)
        proxy_config: Optional proxy configuration (WebshareProxyConfig or GenericProxyConfig)

    Returns:
        Dict of plain tool callables ready for adaptation by agent frameworks

    Example:
        >>> # Default with rate limiting (recommended)
        >>> tools = youtube_search_tools(max_transcript_tokens=4000)
        >>>
        >>> # With custom rate limiting
        >>> tools = youtube_search_tools(
        ...     max_transcript_tokens=4000,
        ...     rate_limit_delay=2.0,  # 2 seconds between requests
        ...     max_retries=5
        ... )
        >>>
        >>> # With proxy support
        >>> from youtube_transcript_api.proxies import WebshareProxyConfig
        >>> proxy_config = WebshareProxyConfig(
        ...     proxy_username="user",
        ...     proxy_password="pass"
        ... )
        >>> tools = youtube_search_tools(
        ...     proxy_config=proxy_config,
        ...     rate_limit_delay=1.0
        ... )
        >>>
        >>> # Disable rate limiting (not recommended, may cause IP blocking)
        >>> tools = youtube_search_tools(rate_limit_delay=0)
    """

    # Use rate-limited searcher by default to prevent IP blocking
    searcher = RateLimitedYouTubeSearcher(
        rate_limit_delay=rate_limit_delay,
        max_retries=max_retries,
        enable_cache=enable_cache,
        proxy_config=proxy_config,
        progress_callback=progress_callback,
    )

    # Create text processor if needed
    if text_processor is None and max_transcript_tokens is not None:
        text_processor = TextProcessor(
            max_tokens=max_transcript_tokens,
            strategy=cast(Literal["simple", "map_reduce", "sequential"], truncation_strategy),
        )
    def search_youtube(
        query: str,
        max_results: int = 10,
        duration: str | None = None,
        upload_date: str | None = None,
        sort_by: str | None = None,
    ) -> SearchResults:
        """Search YouTube for videos matching the query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            duration: Filter by video duration - "short" (< 4 min) or "long" (> 20 min). Omit for any duration.
            upload_date: Filter by upload date - "hour", "today", "week", "month", or "year". Omit for any date.
            sort_by: Sort results by - "relevance" (default), "date", "views", or "rating". Omit for relevance.

        Returns:
            SearchResults object containing video information
        """
        return searcher.search(
            query,
            max_results=max_results,
            duration=duration,
            upload_date=upload_date,
            sort_by=sort_by,
        )
    def get_video_transcript(video_url_or_id: str) -> str:
        """Get the transcript for a YouTube video.

        This function returns the transcript text, automatically handling
        long transcripts according to the configured truncation strategy.

        If you get throttling or rate-limit errors, call set_transcript_delay
        to increase the wait time between requests before retrying.

        Args:
            video_url_or_id: YouTube video URL or video ID

        Returns:
            Transcript text (truncated if configured). If the transcript
            cannot be fetched, returns either a `TRANSCRIPT_RATE_LIMITED`
            or `TRANSCRIPT_UNAVAILABLE` message describing the failure so
            the agent can continue.
        """
        try:
            transcript_obj = searcher.get_transcript(video_url_or_id)
        except Exception as exc:
            if _is_transcript_rate_limited(exc):
                return _format_transcript_rate_limited_message(
                    video_url_or_id,
                    exc,
                )
            return _format_transcript_unavailable_message(video_url_or_id, exc)

        full_text = transcript_obj.get_full_text()

        # Apply text processing if configured
        if text_processor is not None:
            full_text = text_processor.process(full_text)

        return full_text
    def get_video_metadata(video_url_or_id: str) -> VideoMetadata:
        """Get metadata for a YouTube video.

        Args:
            video_url_or_id: YouTube video URL or video ID

        Returns:
            VideoMetadata object with video information
        """
        return searcher.get_video_metadata(video_url_or_id)
    def set_transcript_delay(delay_seconds: float) -> str:
        """Set the delay between transcript requests to avoid throttling.

        Call this tool to adjust the wait time between transcript fetches.
        Start with a low value (e.g. 0.5) and increase if you encounter
        rate-limit or throttling errors (e.g. to 3.0, 5.0, or 10.0).

        Args:
            delay_seconds: Seconds to wait between transcript requests (0.0 to 30.0)

        Returns:
            Confirmation message with the new delay
        """
        clamped = max(0.0, min(30.0, delay_seconds))
        searcher.rate_limit_delay = clamped
        active_rate_limit_error = searcher._get_active_transcript_rate_limit_error()
        if active_rate_limit_error is not None:
            return (
                f"Transcript delay set to {clamped:.1f}s between requests. "
                f"Current transcript block still active: {active_rate_limit_error}"
            )
        return f"Transcript delay set to {clamped:.1f}s between requests."

    return ToolRegistry.from_mapping({
        "search_youtube": search_youtube,
        "get_video_transcript": get_video_transcript,
        "get_video_metadata": get_video_metadata,
        "set_transcript_delay": set_transcript_delay,
    })
