# settings.py
RANDOM_STATE = 42
VALID_SIZE = 0.15
TEST_SIZE = 0.15

ACCOUNT_FILE = "accounts.csv"
SUBSCRIPTIONS_FILE = "subscriptions.csv"
FEATURE_USAGE_FILE = "feature_usage.csv"
SUPPORT_TICKETS_FILE = "support_tickets.csv"
CHURN_EVENTS_FILE = "churn_events.csv"

TARGET_COL = "churn_flag"
ACCOUNT_KEY = "account_id"
SUBSCRIPTION_KEY = "subscription_id"

LOW_CARDINALITY_THRESHOLD = 25
TOP_K_COUNTRIES_FOR_DISPLAY = 10

MEDIAN_IMPUTE_COLUMNS = [
    "avg_resolution_time_hours",
    "avg_first_response_time_minutes",
    "avg_satisfaction_score",
]

MISSING_FLAG_COLUMNS = [
    "avg_resolution_time_hours",
    "avg_first_response_time_minutes",
    "avg_satisfaction_score",
]

KEY_NUMERIC_FEATURES = [
    "account_age_days",
    "total_subscriptions",
    "active_subscriptions",
    "avg_sub_seats",
    "avg_mrr_amount",
    "total_arr_amount",
    "total_usage_count",
    "total_usage_duration_secs",
    "total_error_count",
    "unique_feature_count",
    "days_since_last_usage",
    "error_rate",
    "total_tickets",
    "avg_resolution_time_hours",
    "avg_first_response_time_minutes",
    "avg_satisfaction_score",
    "escalation_ratio",
    "usage_per_subscription",
    "ticket_per_subscription",
    "error_per_subscription",
    "health_score",
]
