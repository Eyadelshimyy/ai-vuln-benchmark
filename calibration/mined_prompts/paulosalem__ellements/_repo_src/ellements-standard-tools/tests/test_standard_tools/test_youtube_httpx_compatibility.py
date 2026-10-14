"""Tests to ensure youtube-search-python httpx compatibility is maintained.

These tests verify that our monkey-patch for the youtube-search-python library's
httpx compatibility issues continues to work correctly. The library has a bug where
it tries to pass 'proxies' parameter to httpx.post() and httpx.get(), which is not
supported in modern httpx versions.

Our fix monkey-patches the RequestCore class to use httpx.Client with proxies instead.
"""

from unittest.mock import MagicMock, patch

import httpx


class TestYouTubeHttpxCompatibility:
    """Test that our httpx compatibility patches work correctly."""

    def test_monkey_patch_is_applied(self):
        """Verify that the monkey-patch is applied to RequestCore."""
        # Importing the labs youtube module installs the monkey-patch at import time.
        import inspect

        import ellements.standard_tools.web.youtube  # noqa: F401
        from youtubesearchpython.core.requests import RequestCore

        # Get the source of syncPostRequest
        source = inspect.getsource(RequestCore.syncPostRequest)

        # Our patch should contain "Patched" in the docstring or handle proxy differently
        assert "Patched" in source or "httpx.Client" in source, \
            "RequestCore.syncPostRequest should be our patched version"

    def test_search_works_without_proxy(self):
        """Test that search works without proxy (normal case)."""
        from ellements.standard_tools.web.youtube import YouTubeSearcher

        searcher = YouTubeSearcher()

        # This should not raise the 'proxies' error
        # Use a simple query that's likely to work
        result = searcher.search(query="test", max_results=1)

        # Should get results without errors
        assert result is not None
        assert result.query == "test"

    def test_search_handles_proxy_parameter_gracefully(self):
        """Test that if somehow proxy config gets passed, it's handled correctly."""
        from ellements.standard_tools.web.youtube import YouTubeSearcher

        searcher = YouTubeSearcher()

        # Even if proxy-related kwargs are passed, they should be filtered out
        # and not cause the 'proxies' parameter error
        try:
            result = searcher.search(
                query="test",
                max_results=1,
                proxies={"http": "http://proxy.example.com"}  # Should be filtered out
            )
            # Should succeed (proxy param ignored)
            assert result is not None
        except TypeError as e:
            # If it fails, it should NOT be about 'proxies' parameter
            assert "proxies" not in str(e).lower(), \
                f"Should not have httpx 'proxies' parameter error, got: {e}"

    def test_patched_methods_use_client_for_proxy(self):
        """Test that patched methods use httpx.Client when proxy is set."""
        from youtubesearchpython.core.requests import RequestCore

        # Create a RequestCore instance
        request_core = RequestCore()
        request_core.url = "https://www.youtube.com/youtubei/v1/search"
        request_core.data = {"test": "data"}
        request_core.timeout = 5

        # Set a proxy (simulating the library's proxy setup)
        request_core.proxy = {"http://": "http://proxy.example.com"}

        # Mock httpx.Client to verify it's used correctly
        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock(spec=httpx.Response)
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            # Call the patched syncPostRequest
            result = request_core.syncPostRequest()

            # Verify Client was created with proxy
            mock_client_class.assert_called_once_with(proxy=request_core.proxy)

            # Verify post was called on the client
            mock_client.post.assert_called_once()

            # Verify it returns the response
            assert result == mock_response

    def test_patched_methods_use_direct_call_without_proxy(self):
        """Test that patched methods use direct httpx.post when no proxy is set."""
        from youtubesearchpython.core.requests import RequestCore

        # Create a RequestCore instance without proxy
        request_core = RequestCore()
        request_core.url = "https://www.youtube.com/youtubei/v1/search"
        request_core.data = {"test": "data"}
        request_core.timeout = 5
        request_core.proxy = {}  # Empty proxy dict (no proxy)

        # Mock httpx.post to verify it's called correctly
        with patch('httpx.post') as mock_post:
            mock_response = MagicMock(spec=httpx.Response)
            mock_post.return_value = mock_response

            # Call the patched syncPostRequest
            result = request_core.syncPostRequest()

            # Verify direct post was called (not Client)
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args[1]

            # Verify 'proxies' is NOT in the parameters
            assert 'proxies' not in call_kwargs, \
                "Direct httpx.post should not have 'proxies' parameter"

            # Verify it returns the response
            assert result == mock_response

    def test_patched_get_request_works(self):
        """Test that patched syncGetRequest also works correctly."""
        from youtubesearchpython.core.requests import RequestCore

        # Create a RequestCore instance without proxy
        request_core = RequestCore()
        request_core.url = "https://www.youtube.com/watch?v=test123"
        request_core.timeout = 5
        request_core.proxy = {}  # No proxy

        # Mock httpx.get to verify it's called correctly
        with patch('httpx.get') as mock_get:
            mock_response = MagicMock(spec=httpx.Response)
            mock_get.return_value = mock_response

            # Call the patched syncGetRequest
            result = request_core.syncGetRequest()

            # Verify direct get was called
            mock_get.assert_called_once()
            call_kwargs = mock_get.call_args[1]

            # Verify 'proxies' is NOT in the parameters
            assert 'proxies' not in call_kwargs, \
                "Direct httpx.get should not have 'proxies' parameter"

            # Verify it returns the response
            assert result == mock_response

    def test_original_httpx_compatibility_issue_is_prevented(self):
        """Verify the patched ``syncPostRequest`` no longer crashes when a
        proxy dict is supplied, which is the regression the monkey-patch
        guards against.
        """
        # Make sure the labs monkey-patch is in place.
        import ellements.standard_tools.web.youtube  # noqa: F401
        from youtubesearchpython.core.requests import RequestCore

        request_core = RequestCore()
        request_core.url = "https://www.youtube.com/youtubei/v1/search"
        request_core.data = {"test": "data"}
        request_core.timeout = 5
        request_core.proxy = {"http://": "http://proxy.example.com"}

        with patch('httpx.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock(spec=httpx.Response)
            mock_client.post.return_value = mock_response
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = None
            mock_client_class.return_value = mock_client

            # This must not raise TypeError about 'proxies'.
            result = request_core.syncPostRequest()
            assert result is not None

    def test_kwargs_filtering_in_search(self):
        """Test that search() filters out unsupported kwargs like 'proxies'."""
        from ellements.standard_tools.web.youtube import YouTubeSearcher

        searcher = YouTubeSearcher()

        # Pass unsupported kwargs - should be filtered out
        with patch('ellements.standard_tools.web.youtube.VideosSearch') as mock_search:
            mock_instance = MagicMock()
            mock_instance.result.return_value = {'result': []}
            mock_search.return_value = mock_instance

            searcher.search(
                query="test",
                max_results=5,
                proxies={"http": "http://proxy.example.com"},  # Unsupported
                some_random_param="value"  # Unsupported
            )

            # Verify VideosSearch was called without unsupported params
            call_kwargs = mock_search.call_args[1]
            assert 'proxies' not in call_kwargs, "proxies should be filtered out"
            assert 'some_random_param' not in call_kwargs, "random params should be filtered out"

    def test_kwargs_filtering_in_get_video_metadata(self):
        """Test that get_video_metadata() filters out unsupported kwargs."""
        from ellements.standard_tools.web.youtube import YouTubeSearcher

        searcher = YouTubeSearcher()

        # Pass unsupported kwargs - should be filtered out
        with patch('ellements.standard_tools.web.youtube.Video.getInfo') as mock_get_info:
            mock_get_info.return_value = {
                'title': 'Test Video',
                'channel': {'name': 'Test Channel', 'id': 'test123'}
            }

            searcher.get_video_metadata(
                "dQw4w9WgXcQ",
                proxies={"http": "http://proxy.example.com"},  # Unsupported
                some_random_param="value"  # Unsupported
            )

            # Verify Video.getInfo was called without unsupported params
            call_kwargs = mock_get_info.call_args[1]
            assert 'proxies' not in call_kwargs, "proxies should be filtered out"
            assert 'some_random_param' not in call_kwargs, "random params should be filtered out"
