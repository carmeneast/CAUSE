from upsell.utils.s3 import s3_key

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()


class TruePrecisionOpps:

    def __init__(self, tenant_id: str, model_id: str, run_date: str, opportunity_selector: str = '',
                 days_of_opps: int = 30, bucket: str = 'ceasterwood'):
        self.tenant_id = tenant_id
        self.model_id = model_id
        self.run_date = run_date
        self.days_of_opps = days_of_opps
        self.opportunity_selector = opportunity_selector
        self.bucket = bucket

        if len(self.opportunity_selector.strip()) == 0:
            self.opportunity_selector = 'true'

    def get_opportunities(self) -> DataFrame:
        # Get opportunities created soon after the run date
        opportunity_query = f"""
            WITH all_opps AS (
                -- All opps opened since the run date
                SELECT tenant_id
                    , account_id
                    , id AS opp_id
                    , DATE(createdDate) AS created_date
                    , CASE WHEN {self.opportunity_selector} THEN 1 ELSE 0 END AS meets_selector_criteria
                    , DATEDIFF(DATE(createdDate), DATE('{self.run_date}')) AS days_since_run_date
                FROM db1_data_warehouse.tenant.opportunity
                WHERE tenant_id = '{self.tenant_id}'
                AND DATE(createdDate) > DATE('{self.run_date}')
                ORDER BY 1, 2, 3, 4
            )
            , ordered_opps AS (
                -- For each account, get the first opp that meets the selector criteria.
                -- If none meet the criteria, get the first opp of any kind.
                SELECT *
                    , ROW_NUMBER() OVER (PARTITION BY tenant_id, account_id
                        ORDER BY meets_selector_criteria DESC, created_date) AS opp_number
                FROM all_opps
            )
            SELECT tenant_id
                , account_id
                , opp_id
                , created_date
                , meets_selector_criteria
                , days_since_run_date
            FROM ordered_opps
            WHERE opp_number = 1
            AND days_since_run_date <= {self.days_of_opps}
        """
        return spark.sql(opportunity_query)

    def get_journey_stage(self) -> DataFrame:
        # Get journey stage as of run_date to check if account was a customer when opp was opened
        journey_query = f"""
            SELECT tenant_id
                , account_id
                , DATE(enteredAt) AS entered_at
                , stageName AS stage_name
                , CASE WHEN LOWER(stageName) RLIKE 'customer|expansion|closed won|existing|upsell'
                  AND LOWER(stageName) NOT RLIKE 'lost' THEN 1 ELSE 0 END AS customer_stage
            FROM db1_data_warehouse.tenant.journey
            WHERE tenant_id = '{self.tenant_id}'
            AND enteredAt <= TIMESTAMP('{self.run_date}')
        """
        return spark.sql(journey_query)\
            .withColumn('rank', row_number().over(
                Window.partitionBy('tenant_id', 'account_id').orderBy(desc('entered_at'))))\
            .filter(col('rank') == lit(1))\
            .withColumn('dt', lit(self.run_date))\
            .drop('entered_at', 'rank')

    def get_precision_opps(self) -> DataFrame:
        opportunities = self.get_opportunities()
        journeys = self.get_journey_stage()

        return journeys.join(opportunities, ['tenant_id', 'account_id'], 'left')\
            .withColumn('label', when(col('created_date').isNotNull(), lit(1)).otherwise(lit(0)))\
            .withColumn('opp_type', when(col('created_date').isNotNull() & (col('customer_stage') == lit(1)),
                                         lit('Post-Customer'))
                        .when(col('created_date').isNotNull() & (col('customer_stage') == lit(0)), lit('New Business'))
                        )\
            .select('tenant_id', 'account_id', 'dt', 'label', 'customer_stage', 'opp_type', 'days_since_run_date')\
            .orderBy('tenant_id', 'account_id')

    def get_save_opps(self) -> None:
        opps = self.get_precision_opps()

        prefix = s3_key(self.tenant_id, self.run_date, self.model_id)
        opps.coalesce(1)\
            .write\
            .option('header', 'true')\
            .mode('overwrite')\
            .csv(f's3://{self.bucket}/{prefix}truePrecision/')


if __name__ == '__main__':
    _tenant_ids = ['1681']
    _model_id = '2'
    _run_date = '2023-11-01'
    _days_of_opps = 28
    _opportunity_selector = ''
    _bucket = 'ceasterwood'

    for _tenant_id in _tenant_ids:
        tp = TruePrecisionOpps(_tenant_id, _model_id, _run_date, _opportunity_selector, _days_of_opps, _bucket)
        tp.get_save_opps()
