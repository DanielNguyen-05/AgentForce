"""Project-specific exceptions with actionable error messages."""


class AgentForceError(Exception):
    """Base class for expected pipeline failures."""


class ConfigurationError(AgentForceError):
    """Raised when configuration is missing or invalid."""


class DataValidationError(AgentForceError):
    """Raised when canonical dataset invariants are violated."""


class OptionalDependencyError(AgentForceError):
    """Raised when a requested optional backend is not installed."""


class ExternalServiceError(AgentForceError):
    """Raised when an external inference service fails safely."""

