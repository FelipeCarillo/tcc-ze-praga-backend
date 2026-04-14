from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):  # type: ignore[misc]  # pydantic+generic requires old syntax
    items: list[T]
    total: int
    page: int
    limit: int

    @property
    def pages(self) -> int:
        if self.limit == 0:
            return 0
        return (self.total + self.limit - 1) // self.limit
