from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import *
from pyspark.sql.types import StructType
from typing import Optional


def validate_account_df_schema(input_col: str, schema: StructType) -> None:
    for _col in [input_col, 'account_id']:
        assert _col in schema.names, f'{_col} column must be present in X'


class FlatEventTypeRollUp(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    """ Keep only events that meet certain conditions and ignore the rest """

    def __init__(self, input_col: str, output_col: str, min_accounts: int = 50,
                 min_pct_accounts: Optional[float] = None):
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.min_accounts = min_accounts
        self.min_pct_accounts = min_pct_accounts

    def _fit(self, account_df: DataFrame):
        validate_account_df_schema(self.input_col, account_df.schema)

        # Filter out events that don't meet the minimum account threshold
        all_input_vals = account_df.filter(col(self.input_col).isNotNull()) \
            .groupBy(self.input_col) \
            .agg(count('account_id').alias('accounts')) \
            .filter(col('accounts') >= self.min_accounts)

        if self.min_pct_accounts is not None:
            # Filter out events that don't meet the minimum % of accounts threshold
            all_input_vals = all_input_vals \
                .withColumn('pct_accounts', col('accounts') / sum('accounts').over(Window.partitionBy())) \
                .filter(col('pct_accounts') >= self.min_pct_accounts)

        event_types = all_input_vals.orderBy(self.input_col) \
            .rdd \
            .map(lambda x: x[self.input_col]) \
            .collect()

        return FlatEventTypeRollUpModel(
            uid=self.uid,
            input_col=self.input_col,
            output_col=self.output_col,
            event_types=event_types
        )

    def __repr__(self):
        return f'FlatEventTypeRollUp(uid={self.uid}, input_col={self.input_col}, min_accounts={self.min_accounts}, ' + \
            f'min_pct_accounts={self.min_pct_accounts},'


class FlatEventTypeRollUpModel(Model, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, uid, input_col, output_col, event_types: list):
        super().__init__()
        self.uid = uid
        self.input_col = input_col
        self.output_col = output_col
        self.event_types = event_types

    def _transform(self, account_df: DataFrame) -> DataFrame:
        validate_account_df_schema(self.input_col, account_df.schema)

        return account_df.withColumn(self.output_col,
                                     when(col(self.input_col).isin(self.event_types), col(self.input_col)))

    def __repr__(self):
        return f'FlatEventTypeRollUpModel(uid={self.uid}, input_col={self.input_col}, event_types={self.event_types}'
