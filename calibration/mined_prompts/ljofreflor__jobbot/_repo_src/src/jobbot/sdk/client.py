"""JobBot SDK client: unified entry point for all SDK operations."""

from __future__ import annotations

from pathlib import Path

from jobbot.config import JobbotConfig, load_config
from jobbot.db.engine import make_engine, make_session_factory
from jobbot.profile.loader import load_profile
from jobbot.sdk.cv_api import CvApi
from jobbot.sdk.jobs_api import JobsApi


class JobBotClient:
    """Main SDK client providing access to JobBot functionality.

    This client manages configuration, database sessions, and provides
    access to domain-specific APIs (jobs, CV building, matching, etc.).

    Example:
        >>> client = JobBotClient()
        >>> jobs = client.jobs.search("python developer", remote=True)
        >>> client.cv.build(job=jobs[0], style="modern")
    """

    def __init__(
        self,
        config_path: Path | str | None = None,
        workspace_root: Path | str | None = None,
    ) -> None:
        """Initialize the JobBot client.

        Args:
            config_path: Path to jobbot.toml config file.
                        If None, discovers from workspace or defaults.
            workspace_root: Root directory of jobbot workspace.
                           If None, discovers from current directory or defaults.
        """
        self.config = self._load_config(config_path, workspace_root)
        self._engine = make_engine(self.config.database_path)
        self._session_factory = make_session_factory(self._engine)

        # Load candidate profile
        self._candidate = load_profile(self.config.profile_path)

        # Initialize API namespaces
        self.jobs = JobsApi(self.config, self._session_factory, self._candidate)
        self.cv = CvApi(self.config, self._candidate)

    def _load_config(
        self,
        config_path: Path | str | None,
        workspace_root: Path | str | None,
    ) -> JobbotConfig:
        """Load configuration from file or defaults."""
        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                msg = f"Config file not found: {config_path}"
                raise FileNotFoundError(msg)
            return load_config(config_path)

        if workspace_root:
            workspace_root = Path(workspace_root)
            config_file = workspace_root / "jobbot.toml"
            if config_file.exists():
                return load_config(config_file)
            # Use workspace root with default config
            return JobbotConfig(root=workspace_root)

        # Discover from current directory
        try:
            return load_config()
        except FileNotFoundError:
            # Use current directory as workspace
            return JobbotConfig()

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        self._engine.dispose()

    def __enter__(self) -> JobBotClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.close()
