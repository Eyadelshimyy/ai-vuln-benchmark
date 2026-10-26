"""OtterWorks enterprise-drive taxonomy.

A data-driven description of the OtterWorks company shared drive — an
enterprise retailer of products for otters (SalmonSnax, shrimp treats,
fish-oil supplements, kelp snacks, otter apparel and grooming textiles).
Departments are top-level folders with a deep nested subfolder structure, and
file templates expand over axes (years, quarters, regions, stores, suppliers,
campaigns, skus, ...) to produce a realistic, high-volume, multimodal corpus.

Suppliers, categories and SKUs come from the shared product catalog
(``catalog.py``, backed by the OTD-15 ``testdata/market-series/`` contract) so
every artifact draws figures from one canonical source.

``DEPARTMENTS`` maps a department name to a list of *file specs*. Each spec:

    {
      "folder": "Financial Statements/Quarterly",  # nested path under the dept
      "name":   "{q} {year} Income Statement",     # template (see AXES)
      "type":   "xlsx",                            # file extension
      "expand": ["year", "quarter"],               # axes to cartesian-expand
      "kind":   "chart",                           # optional builder variant
    }

The generator (`generate_drive.py`) creates every folder in every spec's path
and one file per expansion tuple. ``SCALE`` multiplies list-axis breadth.
"""
from __future__ import annotations

import catalog

YEARS = [2023, 2024, 2025, 2026]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REGIONS = ["Northeast", "Southeast", "Midwest", "Southwest", "West", "Pacific-NW"]
STORES = [f"Store-{1000 + i}" for i in range(24)]

# Suppliers and categories come from the shared catalog (OTD-15 contract).
SUPPLIERS = catalog.suppliers()
CATEGORIES = catalog.categories()
SKUS = [s.name.replace(" ", "-").replace("/", "-") for s in catalog.skus()]

CAMPAIGNS = [
    "River-Days", "Salmon-Run-Sale", "Kelp-Forest-Fest", "Tidepool-Treats",
    "Winter-Coat-Drive", "Pup-Season-Picks", "Streamside-Clearance", "New-Rafts",
]

# Axis name -> value list (used for {placeholder} substitution + expansion).
AXES = {
    "year": YEARS,
    "quarter": QUARTERS,
    "month": MONTHS,
    "region": REGIONS,
    "store": STORES,
    "vendor": SUPPLIERS,
    "campaign": CAMPAIGNS,
    "category": CATEGORIES,
    "sku": SKUS,
}


def _s(folder, name, type, expand=None, kind=None):
    spec = {"folder": folder, "name": name, "type": type, "expand": expand or []}
    if kind:
        spec["kind"] = kind
    return spec


DEPARTMENTS: dict[str, list[dict]] = {
    "Finance": [
        _s("Financial Statements/Quarterly", "{quarter} {year} Income Statement", "xlsx", ["year", "quarter"]),
        _s("Financial Statements/Quarterly", "{quarter} {year} Balance Sheet", "xlsx", ["year", "quarter"]),
        _s("Financial Statements/Annual", "{year} Annual Report", "pdf", ["year"]),
        _s("Budgeting/Department Budgets", "{year} {category} Budget", "xlsx", ["year", "category"]),
        _s("Budgeting/Forecasts", "{quarter} {year} Revenue Forecast", "xlsx", ["year", "quarter"]),
        _s("Accounts Payable/Invoices", "Invoice {vendor} {quarter} {year}", "pdf", ["vendor", "quarter"], kind="invoice"),
        _s("Market Charts", "{quarter} {year} Margin Trend", "png", ["year", "quarter"], kind="chart"),
        _s("Tax/Filings", "{year} Corporate Tax Filing", "pdf", ["year"]),
        _s("Policies", "Expense Reimbursement Policy", "docx"),
        _s("Board Reports", "{quarter} {year} Board Financial Review", "pptx", ["year", "quarter"]),
    ],
    "Human Resources": [
        _s("Policies", "Employee Handbook {year}", "pdf", ["year"]),
        _s("Policies", "Code of Conduct", "docx"),
        _s("Recruiting/Job Descriptions", "{category} Store Associate JD", "docx", ["category"]),
        _s("Recruiting/Pipeline", "{quarter} {year} Hiring Pipeline", "xlsx", ["year", "quarter"]),
        _s("Compensation", "{year} Compensation Bands", "xlsx", ["year"]),
        _s("Benefits", "{year} Benefits Guide", "pdf", ["year"]),
        _s("Training/Onboarding", "New Hire Onboarding Checklist", "md"),
        _s("Training/Compliance", "Anti-Harassment Training {year}", "pptx", ["year"]),
        _s("Headcount", "{region} Headcount {quarter} {year}", "xlsx", ["region", "quarter"]),
    ],
    "Legal": [
        _s("Contracts/Suppliers", "MSA {vendor}", "pdf", ["vendor"], kind="contract"),
        _s("Contracts/Leases", "Lease {store}", "pdf", ["store"]),
        _s("Compliance", "{year} Compliance Audit", "pdf", ["year"]),
        _s("Policies", "Data Privacy Policy", "docx"),
        _s("Policies", "Terms of Service", "docx"),
        _s("Litigation/Matters", "Matter Summary {region} {year}", "docx", ["region", "year"]),
        _s("Intellectual Property", "Trademark Portfolio {year}", "xlsx", ["year"]),
    ],
    "Marketing": [
        _s("Campaigns/Briefs", "{campaign} {year} Brief", "docx", ["campaign", "year"]),
        _s("Campaigns/Assets", "{campaign} Hero Banner", "png", ["campaign"]),
        _s("Campaigns/Assets", "{campaign} Social Tile", "jpg", ["campaign"]),
        _s("Campaigns/Performance", "{campaign} {year} Results", "xlsx", ["campaign", "year"]),
        _s("Product Art", "{sku} Product Art", "png", ["sku"], kind="product_art"),
        _s("Product Art", "{sku} Logo Sticker", "svg", ["sku"]),
        _s("Brand", "Brand Guidelines {year}", "pdf", ["year"]),
        _s("Email", "{quarter} {year} Email Calendar", "xlsx", ["year", "quarter"]),
        _s("Analytics", "{quarter} {year} Web Traffic", "csv", ["year", "quarter"]),
        _s("Decks", "{campaign} Strategy Deck", "pptx", ["campaign"]),
    ],
    "Merchandising": [
        _s("Assortment/{category}", "{category} Assortment Plan {year}", "xlsx", ["category", "year"]),
        _s("Planograms/{category}", "{category} Planogram {region}", "pdf", ["category", "region"]),
        _s("Pricing", "{category} Price List {quarter} {year}", "xlsx", ["category", "quarter"]),
        _s("Markdowns", "{quarter} {year} Markdown Schedule", "xlsx", ["year", "quarter"]),
        _s("Supplier Catalogs", "{vendor} Catalog {year}", "pdf", ["vendor", "year"]),
        _s("Spec Sheets", "{sku} Spec Sheet", "pdf", ["sku"], kind="spec_sheet"),
        _s("Seasonal", "{campaign} Merchandising Guide", "pptx", ["campaign"]),
    ],
    "Store Operations": [
        _s("Store Manuals", "Store Operations Manual {year}", "pdf", ["year"]),
        _s("Audits", "{store} Audit {quarter} {year}", "pdf", ["store", "quarter"]),
        _s("Scheduling", "{store} Staff Schedule {month}", "xlsx", ["store", "month"]),
        _s("Openings", "{store} New Store Opening Plan", "docx", ["store"]),
        _s("Checklists", "Daily Opening Checklist", "md"),
        _s("Checklists", "Closing Procedures", "md"),
        _s("Performance", "{region} Store KPIs {quarter} {year}", "xlsx", ["region", "quarter"]),
    ],
    "Supply Chain & Logistics": [
        _s("Inventory", "{region} Inventory Snapshot {month}", "csv", ["region", "month"]),
        _s("Distribution Centers", "DC {region} Throughput {quarter} {year}", "xlsx", ["region", "quarter"]),
        _s("Shipping/Carriers", "Carrier Rate Card {vendor} {year}", "xlsx", ["vendor", "year"]),
        _s("Demand Planning", "{category} Demand Forecast {quarter} {year}", "xlsx", ["category", "quarter"]),
        _s("Freight", "{quarter} {year} Freight Index Chart", "png", ["year", "quarter"], kind="chart"),
        _s("Returns", "{quarter} {year} Returns Analysis", "xlsx", ["year", "quarter"]),
        _s("SOPs", "Warehouse Safety SOP", "pdf"),
    ],
    "E-Commerce": [
        _s("Site Content/Product Pages", "{category} PDP Copy", "docx", ["category"]),
        _s("Merch/Homepage", "{campaign} Homepage Layout", "png", ["campaign"]),
        _s("Product Media", "{sku} Listing Image", "png", ["sku"], kind="product_art"),
        _s("Analytics/Funnel", "{quarter} {year} Conversion Funnel", "csv", ["year", "quarter"]),
        _s("Catalog", "{category} Product Feed", "json", ["category"]),
        _s("A-B Tests", "{campaign} Test Results", "xlsx", ["campaign"]),
        _s("Roadmap", "E-Commerce Roadmap {year}", "pptx", ["year"]),
    ],
    "Information Technology": [
        _s("Architecture", "System Architecture {year}", "pdf", ["year"]),
        _s("Runbooks", "Incident Response Runbook", "md"),
        _s("Runbooks", "Deployment Runbook", "md"),
        _s("Security/Audits", "{year} Security Audit", "pdf", ["year"]),
        _s("Security/Policies", "Access Control Policy", "docx"),
        _s("Inventory", "Asset Inventory {region}", "xlsx", ["region"]),
        _s("Projects", "{category} Systems Integration Plan", "docx", ["category"]),
        _s("Configs", "service-config", "json"),
    ],
    "Procurement": [
        _s("Suppliers/Profiles", "{vendor} Supplier Profile", "docx", ["vendor"]),
        _s("Suppliers/Scorecards", "{vendor} Scorecard {year}", "xlsx", ["vendor", "year"]),
        _s("Purchase Orders", "PO {vendor} {quarter} {year}", "pdf", ["vendor", "quarter"], kind="invoice"),
        _s("Contracts", "Supply Agreement {vendor} {year}", "pdf", ["vendor", "year"], kind="contract"),
        _s("RFPs", "{category} Sourcing RFP {year}", "docx", ["category", "year"]),
        _s("Savings", "{year} Cost Savings Tracker", "xlsx", ["year"]),
    ],
    "Customer Service": [
        _s("Playbooks", "Customer Support Playbook {year}", "pdf", ["year"]),
        _s("Macros", "Response Macros {category}", "md", ["category"]),
        _s("Metrics", "{quarter} {year} CSAT Report", "xlsx", ["year", "quarter"]),
        _s("Metrics", "{quarter} {year} Ticket Volume", "csv", ["year", "quarter"]),
        _s("Escalations", "{region} Escalation Log {month}", "xlsx", ["region", "month"]),
        _s("Training", "Support Onboarding", "pptx"),
    ],
    "Real Estate & Facilities": [
        _s("Leases", "{store} Lease Agreement", "pdf", ["store"]),
        _s("Floor Plans", "{store} Floor Plan", "png", ["store"]),
        _s("Maintenance", "{region} Maintenance Log {quarter} {year}", "xlsx", ["region", "quarter"]),
        _s("Construction", "{store} Buildout Plan", "pdf", ["store"]),
        _s("Utilities", "{region} Utilities {year}", "xlsx", ["region", "year"]),
    ],
    "Loss Prevention": [
        _s("Policies", "Loss Prevention Policy {year}", "pdf", ["year"]),
        _s("Incidents", "{store} Incident Report {quarter} {year}", "pdf", ["store", "quarter"]),
        _s("Shrinkage", "{region} Shrinkage Analysis {quarter} {year}", "xlsx", ["region", "quarter"]),
        _s("Training", "LP Awareness Training", "pptx"),
        _s("Audits", "{store} Cash Audit {month}", "xlsx", ["store", "month"]),
    ],
    "Analytics & Insights": [
        _s("Dashboards/Sales", "{quarter} {year} Sales Dashboard", "xlsx", ["year", "quarter"]),
        _s("Market Charts", "{quarter} {year} Salmon Price Chart", "png", ["year", "quarter"], kind="chart"),
        _s("Market Charts", "{category} Margin Trend {year}", "png", ["category", "year"], kind="chart"),
        _s("Reports/Category", "{category} Performance {quarter} {year}", "xlsx", ["category", "quarter"]),
        _s("Customer", "{year} Customer Segmentation", "csv", ["year"]),
        _s("Datasets", "{region} Transactions {month}", "csv", ["region", "month"]),
        _s("Models", "Demand Model Spec", "md"),
        _s("Executive", "{quarter} {year} KPI Pack", "pptx", ["year", "quarter"]),
    ],
    "Executive": [
        _s("Board", "{quarter} {year} Board Deck", "pptx", ["year", "quarter"]),
        _s("Strategy", "{year} Strategic Plan", "docx", ["year"]),
        _s("Investor Relations", "{quarter} {year} Earnings Script", "docx", ["year", "quarter"]),
        _s("OKRs", "{year} Company OKRs", "xlsx", ["year"]),
        _s("Communications", "All-Hands {quarter} {year}", "pptx", ["year", "quarter"]),
    ],
}


# Committed MP4 clips (testdata/generated/retail-drive/assets/) -> drive folders.
# Uploaded idempotently by generate_drive.py with mime video/mp4.
ASSET_PLACEMENTS: list[tuple[str, tuple[str, ...]]] = [
    ("salmon-run-sale-promo.mp4", ("Marketing", "Campaigns", "Videos")),
    ("river-days-teaser.mp4", ("Marketing", "Campaigns", "Videos")),
    ("kelp-forest-fest-spot.mp4", ("E-Commerce", "Product Media", "Videos")),
]


# Rich text documents (doc-service) per department: (title, body-topic).
DEPARTMENT_DOCS: dict[str, list[str]] = {
    dept: [
        f"{dept} Charter",
        f"{dept} Quarterly Goals",
        f"{dept} Process Handbook",
        f"{dept} Team Directory",
    ]
    for dept in DEPARTMENTS
}


def all_departments() -> list[str]:
    return list(DEPARTMENTS.keys())
