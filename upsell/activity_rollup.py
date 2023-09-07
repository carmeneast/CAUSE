from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import StructType


def activity_validate_schema(schema: StructType) -> None:
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


class ActivityRoleCategoryRollUp(Estimator, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, min_occurrences: int = 100, min_accounts: int = 50):
        super().__init__()
        self.min_occurrences = min_occurrences
        self.min_accounts = min_accounts

    def _fit(self, df: DataFrame):
        activity_validate_schema(df.schema)

        # Clean up event type names
        df = df.withColumn('event_type', roll_up_event_type(col('event_type'), level=0))

        # Get activity + role + category combinations large enough to not roll up
        act_role_cat = df.groupBy('event_type') \
            .agg(
                countDistinct('account_id').alias('accounts'),
                sum('weight').alias('weight')
            ) \
            .filter((col('accounts') >= self.min_accounts) & (col('weight') >= self.min_occurrences))

        # Roll up small 'activity|role|category' combinations to 'activity|role|<RARE CATEGORIES>'
        remainder = df.join(act_role_cat.drop('weight'), on=['event_type'], how='left') \
            .filter(col('accounts').isNull()) \
            .withColumn('event_type', roll_up_event_type(col('event_type'), level=1)) \
            .drop('accounts')

        act_role = remainder.groupBy('event_type') \
            .agg(
                countDistinct('account_id').alias('accounts'),
                sum('weight').alias('weight')
            ) \
            .filter((col('accounts') >= self.min_accounts) & (col('weight') >= self.min_occurrences))

        # Roll up small 'activity|role|<RARE CATEGORIES>' combinations to 'activity|<RARE ROLE + CATEGORIES>'
        # If combination is still too small, roll up to '<RARE ACTIVITY + ROLE + CATEGORIES>'
        act = remainder.join(act_role.drop('weight'), on=['event_type'], how='left') \
            .filter(col('accounts').isNull()) \
            .withColumn('event_type', roll_up_event_type(col('event_type'), level=2)) \
            .groupBy('event_type') \
            .agg(
                countDistinct('account_id').alias('accounts'),
                sum('weight').alias('weight')
            ) \
            .withColumn(
                'event_type',
                when((col('accounts') >= self.min_accounts) & (col('weight') >= self.min_occurrences), col('event_type'))
                .otherwise(roll_up_event_type(col('event_type'), level=3))
            ) \
            .groupBy('event_type') \
            .agg(
                sum('accounts').alias('accounts'),
                sum('weight').alias('weight')
            )

        activities = act_role_cat\
            .unionByName(act_role)\
            .unionByName(act)\
            .orderBy('event_type')\
            .rdd\
            .map(lambda x: x['event_type'])\
            .collect()

        return ActivityRoleCategoryRollUpModel(self.uid, activities)

    def __repr__(self):
        return f'ActivityRoleCategoryRollUp(uid={self.uid}, min_occurrences={self.min_occurrences}, ' + \
            f'min_accounts={self.min_accounts},'


class ActivityRoleCategoryRollUpModel(Model, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, uid, activities: list):
        super().__init__()
        self.uid = uid
        self.activities = activities

    def _transform(self, df: DataFrame) -> DataFrame:
        activity_validate_schema(df.schema)

        return df\
            .withColumn('rollup0', roll_up_event_type(col('event_type'), level=0))\
            .withColumn('rollup1', roll_up_event_type(col('event_type'), level=1))\
            .withColumn('rollup2', roll_up_event_type(col('event_type'), level=2))\
            .withColumn('rollup3', roll_up_event_type(col('event_type'), level=3))\
            .withColumn(
                'event_type',
                when(col('rollup0').isin(self.activities), col('rollup0'))
                .when(col('rollup1').isin(self.activities), col('rollup1'))
                .when(col('rollup2').isin(self.activities), col('rollup2'))
                .otherwise(col('rollup3'))
            )\
            .groupBy('tenant_id', 'account_id', 'dt', 'event_type')\
            .agg(sum('weight').alias('weight'))

    def __repr__(self):
        return f'ActivityRoleCategoryRollUpModel(uid={self.uid}, n_activities={len(self.activities)}'
