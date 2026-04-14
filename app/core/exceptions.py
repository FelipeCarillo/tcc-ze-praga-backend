from app.shared.enums import FeatureTypeEnum


class NotFoundError(Exception):
    def __init__(self, resource: str, identifier: str | None = None) -> None:
        detail = f"{resource} not found"
        if identifier:
            detail = f"{resource} '{identifier}' not found"
        super().__init__(detail)
        self.detail = detail


class UnauthorizedError(Exception):
    def __init__(self, detail: str = "Unauthorized") -> None:
        super().__init__(detail)
        self.detail = detail


class ForbiddenError(Exception):
    def __init__(self, detail: str = "Access forbidden") -> None:
        super().__init__(detail)
        self.detail = detail


class ConflictError(Exception):
    def __init__(self, detail: str = "Resource already exists") -> None:
        super().__init__(detail)
        self.detail = detail


class QuotaExceededError(Exception):
    def __init__(self, feature: FeatureTypeEnum, limit: int, used: int) -> None:
        detail = f"Daily quota exceeded for '{feature}': {used}/{limit} used"
        super().__init__(detail)
        self.detail = detail
        self.feature = feature
        self.limit = limit
        self.used = used
