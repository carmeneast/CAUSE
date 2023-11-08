from pyspark.sql.functions import *
from pyspark.sql import DataFrame, SparkSession, Window

spark = SparkSession.builder.getOrCreate()


class Account:
    tenant_id: str
    account_id: str
    domain: str
    billingCountry___demandbase: str
    billingCountry___salesforce: str
    billingCountry___hubspot: str
    billingCountry___dynamics: str
    industry___demandbase: str
    industry___salesforce: str
    industry___hubspot: str
    industry___dynamics: str
    numberOfEmployees___demandbase: str
    numberOfEmployees___salesforce: str
    numberOfEmployees___hubspot: str
    numberOfEmployees___dynamics: str
    revenueRange___demandbase: str
    start_dt: str


class Event:
    tenant_id: str
    account_id: str
    activity_date: str
    event_type: str
    weight: float


class Metrics:
    tenant_id: str
    accounts: int
    activities: int
    oppEvents: int
    intentEvents: int
    accountsWithActivity: int
    accountsWithOppEvent: int
    accountsWithIntentEvent: int
    accountsWithNoEvents: int
    pctAccountsNoEvents: float


class RawEvents:
    def __init__(self, tenant_id: str, run_date: str, activity_months: int, intent_months: int, bucket: str):
        self.tenant_id = tenant_id
        self.run_date = run_date
        self.activity_months = activity_months
        self.intent_months = intent_months
        self.bucket = bucket

    def get_accounts(self) -> DataFrame:
        account_query = f"""
        SELECT tenant_id
            , id AS account_id
            , domain___e AS domain
            , fields_map['billingCountry___Demandbase'] AS billingCountry___demandbase
            , fields_map['billingCountry___Salesforce'] AS billingCountry___salesforce
            , fields_map['billingCountry___Hubspot'] AS billingCountry___hubspot
            , fields_map['billingCountry___Dynamics'] AS billingCountry___dynamics
            , fields_map['industry___Demandbase'] AS industry___demandbase
            , fields_map['industry___Salesforce'] AS industry___salesforce
            , fields_map['industry___Hubspot'] AS industry___hubspot
            , fields_map['industry___Dynamics'] AS industry___dynamics
            , fields_map['numberOfEmployees___Demandbase'] AS numberOfEmployees___demandbase
            , fields_map['numberOfEmployees___Salesforce'] AS numberOfEmployees___salesforce
            , fields_map['numberOfEmployees___Hubspot'] AS numberOfEmployees___hubspot
            , fields_map['numberOfEmployees___Dynamics'] AS numberOfEmployees___dynamics
            , fields_map['revenueRange___d'] AS revenueRange___demandbase
        FROM db1_data_warehouse.tenant.account
        WHERE tenant_id = '{self.tenant_id}'
        AND domain___e IS NOT NULL
        """
        accounts = spark.sql(account_query)

        # Get account's first journey stage
        journey_query = f"""
        WITH first_stage_dt AS (
            -- First date we started collecting data on the account
            SELECT tenant_id
                , account_id
                , MIN(enteredAt) AS enteredAt
            FROM db1_data_warehouse.tenant.journey
            WHERE tenant_id = '{self.tenant_id}'
            AND enteredAt <= TIMESTAMP('{self.run_date}')
            GROUP BY 1, 2
        )
        , latest_stage_dt AS (
            -- Last journey stage change before data collection began (if exists)
            SELECT tenant_id
                , account_id
                , MAX(enteredAt) AS enteredAt
            FROM db1_data_warehouse.tenant.journey
            WHERE tenant_id = '{self.tenant_id}'
            AND enteredAt <= TIMESTAMP(ADD_MONTHS('{self.run_date}', -{self.activity_months}))
            GROUP BY 1, 2
        )
        , final_dts AS (
            SELECT tenant_id
                , account_id
                , CASE WHEN DATE(f.enteredAt) > DATE(ADD_MONTHS('{self.run_date}', -{self.activity_months}))
                    THEN DATE(f.enteredAt)
                    ELSE DATE(ADD_MONTHS('{self.run_date}', -{self.activity_months}))
                    END AS start_dt
                , COALESCE(l.enteredAt, f.enteredAt) AS initial_stage_entered_at
            FROM first_stage_dt f
            LEFT JOIN latest_stage_dt l
            USING (tenant_id, account_id)
        )
        SELECT f.tenant_id
            , f.account_id
            , f.start_dt
            , TRIM(j.stageName) AS initial_journey_stage
        FROM final_dts f
        INNER JOIN db1_data_warehouse.tenant.journey j
        ON f.tenant_id = j.tenant_id
        AND f.account_id = j.account_id
        AND f.initial_stage_entered_at = j.enteredAt
        WHERE j.tenant_id = '{self.tenant_id}'
        """
        journeys = spark.sql(journey_query)
        return accounts.join(journeys, on=['tenant_id', 'account_id'])

    def get_opportunities(self) -> DataFrame:
        opportunity_query = f"""
        WITH all_opps AS (
            SELECT tenant_id
              , account_id
              , id AS opp_id
              , DATE(createdDate) AS createdDate
              , DATE(closeDate) AS closeDate
              , isWon
              , CASE WHEN DATE(createdDate) > DATE(closeDate) THEN 1 ELSE 0 END AS backdated
              , ADD_MONTHS('{self.run_date}', -{self.activity_months}) AS start_dt
              , '{self.run_date}' AS end_dt
            FROM db1_data_warehouse.tenant.opportunity
            WHERE tenant_id = '{self.tenant_id}'
            ORDER BY 1, 2, 3, 4
        )
        SELECT *
        FROM all_opps
        WHERE (
            -- Opened during the observation period
            (start_dt <= createdDate AND createdDate <= end_dt)
            -- Closed during the observation period
            OR (start_dt <= closeDate AND closeDate <= end_dt)
            )
        """
        opportunities = spark.sql(opportunity_query)

        # Get journey stages to check if account was a customer when opp was opened
        journey_query = f"""
        SELECT tenant_id
            , account_id
            , DATE(enteredAt) AS enteredAt
            , stageName
            , CASE WHEN LOWER(stageName) RLIKE 'customer|expansion|closed won|existing|upsell'
              AND LOWER(stageName) NOT RLIKE 'lost' THEN 1 ELSE 0 END AS customer_stage
        FROM db1_data_warehouse.tenant.journey
        WHERE tenant_id = '{self.tenant_id}'
        AND enteredAt <= TIMESTAMP('{self.run_date}')
        """
        journeys = spark.sql(journey_query)

        opp_with_journey_stage = opportunities \
            .join(journeys, on=['tenant_id', 'account_id']) \
            .filter(col('enteredAt') <= col('createdDate')) \
            .withColumn(
                'rank',
                row_number().over(
                    Window.partitionBy('tenant_id', 'account_id', 'opp_id').orderBy(col('enteredAt').desc())
                )
            ) \
            .filter(col('rank') == 1) \
            .drop('rank') \
            .orderBy('tenant_id', 'account_id', 'opp_id', 'enteredAt')

        opened = opp_with_journey_stage \
            .filter(col('createdDate') >= col('start_dt')) \
            .filter(col('createdDate') <= col('end_dt')) \
            .withColumn(
                'event_type',
                when(col('customer_stage') == 1, lit('opened_post_customer_opportunity'))
                .otherwise(lit('opened_new_business_opportunity'))
            ) \
            .select('tenant_id', 'account_id', 'createdDate', 'event_type') \
            .withColumnRenamed('createdDate', 'activity_date')

        closed = opp_with_journey_stage \
            .filter(col('closeDate') >= col('start_dt')) \
            .filter(col('closeDate') <= col('end_dt')) \
            .withColumn(
                'event_type',
                when((col('customer_stage') == 1) & col('isWon'), lit('closed_won_post_customer_opportunity'))
                .when((col('customer_stage') == 1) & ~col('isWon'), lit('closed_lost_post_customer_opportunity'))
                .when(col('isWon'), lit('closed_won_new_business_opportunity'))
                .when(~col('isWon'), lit('closed_lost_new_business_opportunity'))
            ) \
            .select('tenant_id', 'account_id', 'createdDate', 'event_type') \
            .withColumnRenamed('createdDate', 'activity_date')

        return opened \
            .unionByName(closed) \
            .groupBy('tenant_id', 'account_id', 'activity_date', 'event_type') \
            .agg(count('*').alias('weight'))

    def get_activities(self) -> DataFrame:
        anonymous_activity_query = f"""
        -- Page Visits (Anonymous)
        SELECT tenant_id
            , account_id
            , DATE(activity_date) AS activity_date
            , CONCAT(
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(activityType)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
                '__',
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(_per_role)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
                '__',
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(category___e)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
            ) AS event_type
            , COUNT(*) AS weight
        FROM db1_data_warehouse.tenant.activity_bundle
        WHERE tenant_id = '{self.tenant_id}'
        AND activity_date <= TIMESTAMP('{self.run_date}')
        AND activity_date >= TIMESTAMP(ADD_MONTHS('{self.run_date}', -{self.activity_months}))
        AND activity_source_type = 'web'
        AND engagement > 0
        GROUP BY 1, 2, 3, 4
        """

        identified_activity_query = f"""
        -- Activities like clicks, form fills, page visits (non-anonymous)
        SELECT tenant_id
            , account_id
            , DATE(activity_date) AS activity_date
            , CONCAT(
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(activityType)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
                '__',
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(_per_role)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
                '__',
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(category___e)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
            ) AS event_type
            , COUNT(*) AS weight
        FROM db1_data_warehouse.tenant.activity_bundle
        WHERE tenant_id = '{self.tenant_id}'
        AND activity_date <= TIMESTAMP('{self.run_date}')
        AND activity_date >= TIMESTAMP(ADD_MONTHS('{self.run_date}', -{self.activity_months}))
        AND activity_source_type = 'db1-platform'
        AND activityType NOT IN ('Intent Surge')
        AND engagement > 0
        GROUP BY 1, 2, 3, 4
        """
        return spark.sql(anonymous_activity_query)\
            .unionByName(spark.sql(identified_activity_query))

    def get_intent_events(self) -> DataFrame:
        # TODO: Add activity_source_type = 'trending_intent'
        # TODO: Check if we can get the specific keyword for intent surge events
        intent_surge_query = f"""
        SELECT tenant_id
            , account_id
            , DATE(activity_date) AS activity_date
            , CONCAT(
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(activityType)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
                '__',
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(_per_role)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
                '__',
                REGEXP_REPLACE(REGEXP_REPLACE(TRIM(LOWER(category___e)), '[^a-zA-Z0-9_ ]+', ''), ' ', '_'),
            ) AS event_type
            , COUNT(*) AS weight
        FROM db1_data_warehouse.tenant.activity_bundle
        WHERE tenant_id = '{self.tenant_id}'
        AND activity_date <= TIMESTAMP('{self.run_date}')
        AND activity_date >= TIMESTAMP(ADD_MONTHS('{self.run_date}', -{self.intent_months}))
        AND activity_source_type = 'db1-platform'
        AND activityType = 'Intent Surge'
        AND engagement > 0
        GROUP BY 1, 2, 3, 4
        """
        keyword_intent_query = f"""
        SELECT tenant_id
            , account_id
            , DATE(activity_date) AS activity_date
            , low_intent_keywords
            , medium_intent_keywords
            , high_intent_keywords
        FROM db1_data_warehouse.tenant.activity_bundle
        WHERE tenant_id = '{self.tenant_id}'
        AND activity_date <= TIMESTAMP('{self.run_date}')
        AND activity_date >= TIMESTAMP(ADD_MONTHS('{self.run_date}', -{self.intent_months}))
        AND activity_source_type = 'intent'
        """

        intent_surge_df = spark.sql(intent_surge_query)
        keyword_intent_df = spark.sql(keyword_intent_query)

        def unpack_keywords(_strength: str) -> DataFrame:
            # TODO: Instead of creating events like db_keyword_intent__other__high (activity, role, category),
            #     do db_keyword_intent__high__<keyword> (activity, category, keyword)
            return keyword_intent_df \
                .select('tenant_id', 'account_id', 'activity_date', f'${_strength}_intent_keywords') \
                .withColumn('keyword', explode(col(f'${_strength}_intent_keywords'))) \
                .drop(f'${_strength}_intent_keyword') \
                .distinct() \
                .withColumn('event_type', lit(f'db_keyword_intent__other__${_strength}')) \
                .groupBy('tenant_id', 'account_id', 'activity_date', 'event_type') \
                .agg(count('keyword').alias('weight'))

        for strength in ['low', 'medium', 'high']:
            intent_surge_df = intent_surge_df.unionByName(unpack_keywords(strength))

        return intent_surge_df

    @staticmethod
    def calculate_metrics(accounts: DataFrame, activities: DataFrame, opp_events: DataFrame,
                          intent_events: DataFrame) -> DataFrame:
        t = accounts.select('tenant_id').distinct()
        n = accounts.agg(count('*').alias('accounts'))
        n_activities = activities.agg(sum('weight').alias('activities')).na.fill(0)
        n_opp_events = opp_events.agg(sum('weight').alias('oppEvents')).na.fill(0)
        n_intent_events = intent_events.agg(sum('weight').alias('intentEvents')).na.fill(0)

        n_acct_w_activity = activities.agg(countDistinct('account_id').alias('accountsWithActivity'))
        n_acct_w_opp = opp_events.agg(countDistinct('account_id').alias('accountsWithOppEvent'))
        n_acct_w_intent = intent_events.agg(countDistinct('account_id').alias('accountsWithIntentEvent'))

        n_acct_no_events = accounts \
            .join(activities, on=['tenant_id', 'account_id'], how='left') \
            .filter(col('weight').isNull()) \
            .select('tenant_id', 'account_id') \
            .join(opp_events, on=['tenant_id', 'account_id'], how='left') \
            .filter(col('weight').isNull()) \
            .select('tenant_id', 'account_id') \
            .join(intent_events, on=['tenant_id', 'account_id'], how='left') \
            .filter(col('weight').isNull()) \
            .agg(count('*').alias('accountsWithNoEvents'))

        return t.join(n) \
            .join(n_activities) \
            .join(n_opp_events) \
            .join(n_intent_events) \
            .join(n_acct_w_activity) \
            .join(n_acct_w_opp) \
            .join(n_acct_w_intent) \
            .join(n_acct_no_events) \
            .withColumn('pctAccountsWActivity', col('accountsWithActivity') / col('accounts')) \
            .withColumn('pctAccountsWOppEvent', col('accountsWithOppEvent') / col('accounts')) \
            .withColumn('pctAccountsWIntentEvent', col('accountsWithIntentEvent') / col('accounts')) \
            .withColumn('pctAccountsNoEvents', col('accountsWithNoEvents') / col('accounts'))

    def save_parquet(self, df: DataFrame, name: str) -> None:
        df.write \
            .mode('overwrite') \
            .parquet(f's3://{self.bucket}/upsell/{self.tenant_id}/{self.run_date}/{name}/')

    def save_json(self, df: DataFrame, name: str) -> None:
        df.coalesce(1) \
            .write \
            .mode('overwrite') \
            .json(f's3://{self.bucket}/upsell/{self.tenant_id}/{self.run_date}/{name}/')

    def main(self) -> None:
        accounts = self.get_accounts().cache()
        opp_events = self.get_opportunities().cache()
        activities = self.get_activities().cache()
        intent_events = self.get_intent_events().cache()

        metrics = self.calculate_metrics(accounts, activities, opp_events, intent_events).cache()

        self.save_parquet(accounts, "accounts")
        self.save_parquet(opp_events, "oppEvents")
        self.save_parquet(activities, "activities")
        self.save_parquet(intent_events, "intentEvents")
        self.save_json(metrics, "metrics/events/")


if __name__ == '__main__':
    TENANT_IDS = ['1309', '1619', '1681', '2874', '5715', '6114', '11640', '12636', '13279', '13574']
    RUN_DATE = '2023-07-01'
    ACTIVITY_MONTHS = 12
    INTENT_MONTHS = 3
    BUCKET = 'ceasterwood'

    # Create datasets for each tenant and save to S3
    all_metrics = None
    for TENANT_ID in TENANT_IDS:
        raw_events = RawEvents(TENANT_ID, RUN_DATE, ACTIVITY_MONTHS, INTENT_MONTHS, BUCKET)
        raw_events.main()

        # Combine metrics datasets
        tenant_metrics = spark.read.json(f's3://{BUCKET}/upsell/{TENANT_ID}/{RUN_DATE}/metrics/events/')

        all_metrics = all_metrics.unionByName(tenant_metrics) if all_metrics else tenant_metrics

    all_metrics.coalesce(1) \
        .write \
        .mode('overwrite') \
        .json(f's3://{BUCKET}/upsell/all-tenant-metrics/{RUN_DATE}/events/')
