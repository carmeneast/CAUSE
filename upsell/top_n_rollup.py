from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import StructType
from typing import List


def top_n_validate_schema(input_col: str, schema: StructType) -> None:
    assert input_col in schema.names, f'{input_col} column must be present in X'


class TopNCategoryRollUp(Estimator, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self,
                 input_col: str,
                 output_col: str,
                 n_categories: int = 10):
        super().__init__()
        self.input_col = input_col
        self.output_col = output_col
        self.n_categories = n_categories

    def _fit(self, df: DataFrame):
        top_n_validate_schema(self.input_col, df.schema)

        categories = df.groupBy(self.input_col) \
            .count() \
            .orderBy(desc('count')) \
            .limit(self.n_categories) \
            .rdd \
            .map(lambda x: x[self.input_col]) \
            .collect()
        return TopNCategoryRollUpModel(
            uid=self.uid,
            input_col=self.input_col,
            output_col=self.output_col,
            categories=categories
        )

    def __repr__(self):
        return f'TopNCategoryRollUp(uid={self.uid}, input_col={self.input_col}, n_categories={self.n_categories})'


class TopNCategoryRollUpModel(Model, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self,
                 uid: str,
                 input_col: str,
                 output_col: str,
                 categories: List[str]):
        super().__init__()
        self.uid = uid
        self.input_col = input_col
        self.output_col = output_col
        self.categories = categories

    def _transform(self, df: DataFrame) -> DataFrame:
        top_n_validate_schema(self.input_col, df.schema)

        return df.withColumn(self.output_col, when(col(self.input_col).isin(self.categories), col(self.input_col)))

    def __repr__(self):
        return f'TopNCategoryRollUpModel(uid={self.uid}, input_col={self.input_col}, categories={self.categories})'
