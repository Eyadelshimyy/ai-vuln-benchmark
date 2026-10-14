"""The demo's raw sales data: what the ``sales`` pipeline lands.

Orders and customers as an export would hand them over. The pipeline writes
them to ``orders_raw`` / ``customers_raw``, cleans them, and ``sales_model``
(see ``pipeline.py``) reads the cleaned tables.
"""

import pandas as pd

orders_df = pd.DataFrame({
    "order_id":    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "customer_id": [1, 2, 3, 4, 1, 3, 2, 4, 1, 3],
    "product":  ["Widget A", "Widget B", "Gadget X", "Widget A",
                 "Gadget X", "Widget B", "Widget A", "Gadget X",
                 "Widget B", "Widget A"],
    "qty":      [120, 85, 200, 60, 95, 150, 40, 110, 75, 180],
    "revenue":  [3598.80, 4249.15, 19998.00, 1799.40, 9499.05,
                 14998.50, 1199.60, 10998.90, 3748.25, 17997.00],
    "cost":     [2100.00, 2800.00, 14000.00, 1050.00, 6600.00,
                 10500.00, 700.00, 7700.00, 2200.00, 12600.00],
    "status":   ["shipped", "shipped", "open", "shipped", "open",
                 "shipped", "shipped", "open", "shipped", "open"],
})

customers_df = pd.DataFrame({
    "customer_id": [1, 2, 3, 4, 5],
    "name":        ["Acme Corp", "Globex", "Initech", "Umbrella", "Vandelay"],
    "region":      ["North East", "South East", "Midwest", "West", "North East"],
    "segment":     ["enterprise", "smb", "smb", "enterprise", "mid-market"],
})
