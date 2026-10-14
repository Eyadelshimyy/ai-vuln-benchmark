from enum import StrEnum


class SenderRole(StrEnum):
    OWNER = "owner"
    TEAMMATE = "teammate"
    SYSTEM = "system"


class MediaKind(StrEnum):
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    VIDEO = "video"
    OTHER = "other"


class DeliveryStatus(StrEnum):
    ACCEPTED = "accepted"
    DELIVERED = "delivered"
    FAILED = "failed"


class RecipientAddressKind(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    LINK = "link"


class WorkflowRunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class OutboxStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEAD = "dead"


class QuoteStatus(StrEnum):
    DRAFTING = "drafting"
    AWAITING_CUSTOMER_SELECTION = "awaiting_customer_selection"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"
    APPROVED = "approved"
    DELIVERY_PENDING = "delivery_pending"
    DELIVERED = "delivered"


class CustomerQuoteStatus(StrEnum):
    """What the customer did with a delivered quote, as a separate progression
    so owner-side transitions stay untouched."""

    VIEWED = "viewed"
    ACCEPTED = "accepted"
    PAID = "paid"
    DECLINED = "declined"


class QuoteBilling(StrEnum):
    """How a quote is charged: once, or on a recurring interval."""

    ONE_TIME = "one_time"
    RECURRING = "recurring"


class BillingInterval(StrEnum):
    MONTH = "month"
    YEAR = "year"


class QuotePaymentStatus(StrEnum):
    """One hosted checkout attempt: open while its session stands, then a
    terminal state."""

    OPEN = "open"
    PENDING = "pending"  # checkout completed; a delayed method is still settling
    PAID = "paid"
    FAILED = "failed"
    EXPIRED = "expired"


class QuoteSendAction(StrEnum):
    REQUIRE_APPROVAL = "require_approval"
    AUTO_SEND = "auto_send"
    ESCALATE = "escalate"


class HostedLinkKind(StrEnum):
    SIGNUP = "signup"
    PAYMENT = "payment"
