"""EasyBillsPay — Pydantic Models"""
from pydantic import BaseModel, Field, ConfigDict, EmailStr
from typing import Optional
from datetime import datetime, timezone
import uuid

# Sane upper bound for any single monetary amount accepted from a client.
# Prevents e.g. a negative bill amount (which would silently reduce a
# customer's smoothed payment plan or inflate a wallet balance when a
# negative "paid" amount is subtracted) and absurdly large values.
MAX_MONEY_AMOUNT = 100_000.0


class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: str
    phone: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: EmailStr
    full_name: str
    phone: Optional[str] = None
    wallet_balance: float = 0.0
    subscription_fee: float = 0.0
    is_admin: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Bill(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    category: str
    provider: str
    account_number: str
    biller_code: Optional[str] = None
    reference_number: Optional[str] = None
    bpay_code: Optional[str] = None
    amount: float = Field(gt=0, le=MAX_MONEY_AMOUNT)
    due_date: str
    frequency: str
    status: str = "pending"
    paid_by: Optional[str] = None
    paid_at: Optional[str] = None
    payment_reference: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class BillCreate(BaseModel):
    category: str
    provider: str
    account_number: str
    biller_code: Optional[str] = None
    reference_number: Optional[str] = None
    bpay_code: Optional[str] = None
    amount: float = Field(gt=0, le=MAX_MONEY_AMOUNT)
    due_date: str
    frequency: str = "monthly"

class BankDetails(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    account_holder_name: str
    bank_name: str
    account_number: str
    routing_number: str
    account_type: str
    is_primary: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class BankDetailsCreate(BaseModel):
    account_holder_name: str
    bank_name: str
    account_number: str
    routing_number: str
    account_type: str = "checking"

class PaymentStructure(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    payment_frequency: str
    total_yearly_bills: float
    total_monthly_bills: float
    contribution_amount: float
    next_deduction_date: str
    auto_deduct_enabled: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class PaymentStructureCreate(BaseModel):
    payment_frequency: str
    auto_deduct_enabled: bool = True

class BillUpload(BaseModel):
    file_data: str
    file_name: str
    file_type: str

class ParsedBillData(BaseModel):
    category: Optional[str] = None
    provider: Optional[str] = None
    account_number: Optional[str] = None
    amount: Optional[float] = None
    due_date: Optional[str] = None
    extracted_text: str

class DirectDebitRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    mandate_reference: str
    bank_name: str
    bsb: str
    account_number: str
    account_holder_name: str
    account_type: str
    provider: str
    provider_type: str
    provider_account_number: str
    payment_frequency: str
    max_payment_amount: float = Field(gt=0, le=MAX_MONEY_AMOUNT)
    start_date: str
    status: str = "active"
    authorization_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    signature: str
    terms_accepted: bool = True
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class DirectDebitRequestCreate(BaseModel):
    bank_name: str
    bsb: str
    account_number: str
    account_holder_name: str
    account_type: str
    provider: str
    provider_type: str
    provider_account_number: str
    payment_frequency: str
    max_payment_amount: float = Field(gt=0, le=MAX_MONEY_AMOUNT)
    start_date: str
    signature: str

class ProviderConnection(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    provider_name: str
    provider_type: str
    api_endpoint: Optional[str] = None
    account_number: str
    customer_id: Optional[str] = None
    api_key: Optional[str] = None
    status: str = "connected"
    last_sync: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class ProviderConnectionCreate(BaseModel):
    provider_name: str
    provider_type: str
    account_number: str
    customer_id: Optional[str] = None
    api_key: Optional[str] = None

class Transaction(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    type: str
    amount: float
    description: str
    status: str = "completed"
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class MockPayment(BaseModel):
    amount: float = Field(gt=0, le=MAX_MONEY_AMOUNT)
    payment_method: str = "card"

class PaymentMethod(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    type: str
    label: str
    bank_name: Optional[str] = None
    bsb: Optional[str] = None
    account_number_masked: Optional[str] = None
    card_last4: Optional[str] = None
    card_brand: Optional[str] = None
    # Set only via stripe_collections.confirm_setup_intent_and_save() after
    # real Stripe tokenization (SetupIntent). Rows created through the
    # legacy POST /payment-methods raw-entry form leave this null and are
    # correctly treated as not chargeable — see
    # stripe_collections.get_chargeable_payment_method().
    stripe_payment_method_id: Optional[str] = None
    is_primary: bool = False
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class PaymentMethodCreate(BaseModel):
    type: str
    label: str
    bank_name: Optional[str] = None
    bsb: Optional[str] = None
    account_number: Optional[str] = None
    card_number: Optional[str] = None
    card_brand: Optional[str] = None
    is_primary: bool = False

class AdminPayBillRequest(BaseModel):
    bill_id: str
    payment_reference: Optional[str] = None
