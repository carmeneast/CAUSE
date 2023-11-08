from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import StructType


def validate_event_df_schema(schema: StructType) -> None:
    for input_col in ['tenant_id', 'account_id', 'dt', 'event_type', 'weight']:
        assert input_col in schema.names, f'{input_col} column must be present in X'


def roll_up_event_type(event_type_col: Column, level: int) -> Column:
    if level == 0:
        return event_type_col
    elif level == 1:
        return concat(split(event_type_col, '__').getItem(0), lit('__'),
                      split(event_type_col, '__').getItem(1), lit('__RARE_CATEGORIES'))
    elif level == 2:
        return concat(split(event_type_col, '__').getItem(0), lit('__RARE_ROLE_CATEGORIES'))
    elif level == 3:
        return lit(f'RARE_ACTIVITY_ROLE_CATEGORIES')
    else:
        raise ValueError(f'Invalid rollup level: {level}')


class HierarchicalEventTypeRollUp(Estimator, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, min_accounts: int = 50, min_occurrences: int = 100):
        super().__init__()
        self.min_accounts = min_accounts
        self.min_occurrences = min_occurrences

    def _fit(self, event_df: DataFrame):
        validate_event_df_schema(event_df.schema)

        # Get activity__role__category events large enough to not roll up
        roll_up_0 = event_df.groupBy('event_type') \
            .agg(
                countDistinct('account_id').alias('accounts'),
                sum('weight').alias('weight')
            ) \
            .filter((col('accounts') >= self.min_accounts) & (col('weight') >= self.min_occurrences))

        # Roll up small activity__role__category events to activity__role__RARE_CATEGORIES
        remainder = event_df.join(roll_up_0.drop('weight'), on=['event_type'], how='left') \
            .filter(col('accounts').isNull()) \
            .withColumn('event_type', roll_up_event_type(col('event_type'), level=1)) \
            .drop('accounts')

        roll_up_1 = remainder.groupBy('event_type') \
            .agg(
                countDistinct('account_id').alias('accounts'),
                sum('weight').alias('weight')
            ) \
            .filter((col('accounts') >= self.min_accounts) & (col('weight') >= self.min_occurrences))

        # Roll up small activity__role__RARE_CATEGORIES combinations to activity__RARE_ROLE_CATEGORIES
        remainder = remainder.join(roll_up_1.drop('weight'), on=['event_type'], how='left') \
            .filter(col('accounts').isNull()) \
            .withColumn('event_type', roll_up_event_type(col('event_type'), level=2)) \
            .drop('accounts')

        roll_up_2 = remainder.groupBy('event_type') \
            .agg(
                countDistinct('account_id').alias('accounts'),
                sum('weight').alias('weight')
            ) \
            .filter((col('accounts') >= self.min_accounts) & (col('weight') >= self.min_occurrences))

        # If combination is still too small, roll up to RARE_ACTIVITY_ROLE_CATEGORIES
        roll_up_3 = remainder.join(roll_up_2.drop('weight'), on=['event_type'], how='left') \
            .filter(col('accounts').isNull()) \
            .withColumn('event_type', roll_up_event_type(col('event_type'), level=3)) \
            .groupBy('event_type') \
            .agg(
                sum('accounts').alias('accounts'),
                sum('weight').alias('weight')
            )

        event_types = roll_up_0 \
            .unionByName(roll_up_1) \
            .unionByName(roll_up_2) \
            .unionByName(roll_up_3) \
            .orderBy('event_type') \
            .rdd \
            .map(lambda x: x['event_type']) \
            .collect()

        return HierarchicalEventTypeRollUpModel(self.uid, event_types)

    def __repr__(self):
        return f'HierarchicalEventTypeRollUp(uid={self.uid}, min_occurrences={self.min_occurrences}, ' + \
            f'min_accounts={self.min_accounts},'


class HierarchicalEventTypeRollUpModel(Model, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, uid, event_types: list):
        super().__init__()
        self.uid = uid
        self.event_types = event_types

    def _transform(self, df: DataFrame) -> DataFrame:
        validate_event_df_schema(df.schema)

        return df\
            .withColumn('rollup0', roll_up_event_type(col('event_type'), level=0))\
            .withColumn('rollup1', roll_up_event_type(col('event_type'), level=1))\
            .withColumn('rollup2', roll_up_event_type(col('event_type'), level=2))\
            .withColumn('rollup3', roll_up_event_type(col('event_type'), level=3))\
            .withColumn(
                'event_type',
                when(col('rollup0').isin(self.event_types), col('rollup0'))
                .when(col('rollup1').isin(self.event_types), col('rollup1'))
                .when(col('rollup2').isin(self.event_types), col('rollup2'))
                .when(col('rollup3').isin(self.event_types), col('rollup3'))
            )\
            .groupBy('tenant_id', 'account_id', 'dt', 'event_type')\
            .agg(sum('weight').alias('weight'))

    def __repr__(self):
        return f'HierarchicalEventTypeRollUpModel(uid={self.uid}, n_event_types={len(self.event_types)}'
