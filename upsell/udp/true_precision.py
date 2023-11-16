from upsell.utils.s3 import s3_key
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import *

spark = SparkSession.builder.getOrCreate()


class TruePrecisionOpps:

    def __init__(self, tenant_id: str, model_id: Optional[str], run_date: str, opportunity_selector: str = '',
                 days_of_opps: int = 30, bucket: str = 'ceasterwood'):
        self.tenant_id = tenant_id
        self.model_id = model_id
        self.run_date = run_date
        self.days_of_opps = days_of_opps
        self.opportunity_selector = opportunity_selector
        self.bucket = bucket
        self.spark = spark

    def get_opportunities(self) -> DataFrame:
        # Get opportunities created soon after the run date
        opportunity_query = f"""
            WITH all_opps AS (
                SELECT tenant_id
                    , account_id
                    , DATE(createdDate) AS createdDate
                    -- , type AS opp_type
                    , ROW_NUMBER() OVER (PARTITION BY tenant_id, account_id ORDER BY createdDate) AS opp_number
                    , DATEDIFF(DATE(createdDate), DATE('{self.run_date}')) AS days_since_run_date
                FROM db1_data_warehouse.tenant.opportunity
                WHERE tenant_id = '{self.tenant_id}'
                AND DATE(createdDate) > DATE('{self.run_date}')
                {self.opportunity_selector}
                ORDER BY 1, 2, 3, 4
            )
            SELECT tenant_id
                , account_id
                , createdDate AS created_date
                -- , opp_type
                , days_since_run_date
            FROM all_opps
            WHERE opp_number = 1
            AND days_since_run_date <= {self.days_of_opps}
        """
        return self.spark.sql(opportunity_query)

    def get_journey_stage(self) -> DataFrame:
        # Get journey stage as of run_date to check if account was a customer when opp was opened
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
        return self.spark.sql(journey_query)\
            .withColumn('rank', row_number().over(
                Window.partitionBy('tenant_id', 'account_id').orderBy(desc('enteredAt'))))\
            .filter(col('rank') == lit(1))\
            .withColumn('dt', lit(self.run_date))\
            .drop('enteredAt', 'rank')

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
    _tenant_ids = ['1681']  # ['1309', '1619', '1681', '2874', '5715', '6114', '11640', '12636', '13279', '13574']
    _model_id = 'ads'
    _run_date = '2023-07-01'
    _days_of_opps = 35
    _opportunity_selector = ''
    _bucket = 'ceasterwood'

    for _tenant_id in _tenant_ids:
        tp = TruePrecisionOpps(_tenant_id, _model_id, _run_date, _opportunity_selector, _days_of_opps, _bucket)
        tp.get_save_opps()
