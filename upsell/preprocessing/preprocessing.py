from itertools import chain
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import DoubleType, FloatType, MapType, StringType
from pyspark.ml.feature import Imputer, VectorAssembler
# from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.ml.pipeline import Pipeline, PipelineModel
from typing import Dict, List, Optional, Tuple

from upsell.configs import load_yaml_config
from upsell.preprocessing.country_map import COUNTRY_MAP
from upsell.preprocessing.top_n_rollup import TopNCategoryRollUp, TopNCategoryRollUpModel
from upsell.preprocessing.activity_rollup import ActivityRoleCategoryRollUp, ActivityRoleCategoryRollUpModel


class CausePreprocessing:

    def __init__(self,
                 spark: SparkSession,
                 tenant_id: int,
                 run_date: str,
                 # sampling: str,
                 bucket: str = 'ceasterwood',
                 weekly: bool = True
                 ):
        self.spark = spark
        self.tenant_id = tenant_id
        self.run_date = run_date
        # self.sampling = sampling
        self.bucket = bucket
        self.weekly = weekly

        self.CONFIG = load_yaml_config('upsell/config.yml').preprocessing

        self.firmo_rollup: Optional[PipelineModel] = None
        self.firmo_categories: Optional[Dict[str, List[str]]] = None
        self.activity_rollup: Optional[PipelineModel] = None
        self.n_event_types = None
        self.event_type_names = None

    def run(self, dataset: str = 'train'):
        # Load raw event data
        print('Loading raw event data...')
        raw_data_path = f's3://{self.bucket}/upsell/{self.tenant_id}/{self.run_date}'
        transformed_data_path = f'{raw_data_path}/'  # {self.sampling}'

        accounts = self.spark.read.parquet(f'{raw_data_path}/accounts')
        activities = self.spark.read.parquet(f'{raw_data_path}/activities')
        opps = self.spark.read.parquet(f'{raw_data_path}/oppEvents')
        intent = self.spark.read.parquet(f'{raw_data_path}/intentEvents')

        # Create timeline events
        print('Creating timeline events...')
        activity_events = self.create_timeline_events(accounts, activities)
        opp_events = self.create_timeline_events(accounts, opps)
        intent_events = self.create_timeline_events(accounts, intent)

        if dataset == 'train':
            print('Sampling accounts...')
            account_sample = self.sample_accounts(accounts, activity_events, opp_events, intent_events)
            self.save_parquet(account_sample, f'{transformed_data_path}/account_sample')

            print('Splitting into train/test...')
            train_accounts, test_accounts = self.train_test_split(account_sample)
            self.save_parquet(train_accounts, f'{transformed_data_path}/train_accounts')
            self.save_parquet(test_accounts, f'{transformed_data_path}/test_accounts')

            accounts = train_accounts

        elif dataset == 'test':
            accounts = self.spark.read.parquet(f'{transformed_data_path}/test_accounts')

        print(f'Preprocessing {dataset} data...')
        sparse_matrices = self.apply_preprocessing(accounts, activity_events, opp_events, intent_events,
                                                   dataset=dataset)
        self.save_parquet(sparse_matrices, f'{transformed_data_path}/{dataset}_sparse_matrices')

    def apply_preprocessing(self, accounts: DataFrame, activity_events: DataFrame,
                            opp_events: DataFrame, intent_events: DataFrame, dataset: str = 'train'):
        model_path = f'upsell/{self.tenant_id}/{self.run_date}/'  # {self.sampling}'
        transformed_data_path = f's3://{self.bucket}/{model_path}'

        # Clean firmographics data
        print('Cleaning firmographics data...')
        accounts_cleaned = self.clean_firmographics(accounts)

        if dataset == 'train':
            # Fit firmographic roll-up
            print('Fitting firmographic roll-up...')
            self.firmo_rollup, self.firmo_categories = self.fit_firmographic_roll_up(accounts_cleaned)
            self.firmo_rollup.write().overwrite().save(f'{transformed_data_path}/firmo_rollup_pipeline')
            self.save_json(self.firmo_categories, f'{transformed_data_path}/firmo_categories')

            # Fit activity roll-up
            print('Fitting activity roll-up...')
            self.activity_rollup = self.fit_activity_roll_up(
                activity_events.join(accounts_cleaned.select('tenant_id', 'account_id'),
                                     on=['tenant_id', 'account_id'], how='inner')
            )
            self.save_json(self.activity_rollup, f'{transformed_data_path}/activity_rollup_pipeline')
        else:
            # Load roll-ups
            print('Loading preprocessors...')
            self.firmo_rollup = PipelineModel.load(f'{transformed_data_path}/firmo_rollup_pipeline')
            self.firmo_categories = self.spark.read.json(f'{transformed_data_path}/firmo_categories')
            self.activity_rollup = self.spark.read.json(f'{transformed_data_path}/activity_rollup_pipeline')

        # Roll up firmographics and activities
        print('Rolling up firmographics...')
        accounts_transformed = self.firmo_rollup.transform(accounts_cleaned)
        categories = self.firmo_categories.rdd\
            .map(lambda x: {x['feature']: x['categories']})\
            .reduce(lambda x, y: {**x, **y})
        print(categories)
        for feat in categories.keys():
            model = TopNCategoryRollUpModel(self.CONFIG.seed, feat, feat, categories[feat])
            accounts_transformed = model.transform(accounts_transformed)

        print('Rolling up activities...')
        categories = self.activity_rollup.rdd.map(lambda x: x['categories']).collect()[0]
        print(len(categories), categories[:5])
        model = ActivityRoleCategoryRollUpModel(self.CONFIG.seed, categories)
        activities_transformed = model.transform(
            activity_events.join(accounts_cleaned.select('tenant_id', 'account_id'),
                                 on=['tenant_id', 'account_id'], how='inner')
        )

        # Create firmographic events
        print('Creating firmographic events...')
        firmo_events = self.create_firmographic_events(accounts_transformed)

        # Combine events
        print('Combining events...')
        all_events = self.combine_events(accounts_transformed, firmo_events, opp_events, intent_events,
                                         activities_transformed)
        self.save_parquet(all_events, f'{transformed_data_path}/{dataset}_all_events')

        if dataset == 'train':
            # Get event type names
            print('Getting event type names...')
            self.event_type_names = all_events.select('event_type').distinct().orderBy('event_type').coalesce(1).cache()
            self.save_parquet(self.event_type_names, f'{transformed_data_path}/event_type_names')
        else:
            self.event_type_names = self.spark.read.parquet(f'{transformed_data_path}/event_type_names').cache()

        self.n_event_types = self.event_type_names.count()
        print(f'Event types: {self.n_event_types}')
        self.event_type_names.show(5, False)

        # Merge events at each timestamp into dict
        print('Merging events at each timestamp into dict...')
        event_agg = self.merge_timestamp_events_to_dict(all_events)

        # Pad event sequences
        print('Padding event sequences...')
        events_padded = self.pad_final_event_date(accounts_transformed, event_agg)

        # Convert to sparse matrices
        print('Converting to sparse matrices...')
        sparse_matrices = self.convert_to_sparse_matrices(events_padded)
        return sparse_matrices

    def sample_accounts(self, accounts: DataFrame, activity_events: DataFrame,
                        opp_events: DataFrame, intent_events: DataFrame):
        def has_enough_accounts(df, n_required, account_description):
            n = df.count()
            print(f'{account_description}: {n}')
            if n < n_required:
                raise ValueError(f'Tenant needs {n_required} {account_description}; only has {n}')

        # Check which accounts have activities, intent, and opp open events
        account_types = accounts.select('tenant_id', 'account_id') \
            .join(
                opp_events.filter(col('event_type').rlike('^opened_'))
                .select('tenant_id', 'account_id')
                .distinct()
                .withColumn('open_opp', lit(1)),
                on=['tenant_id', 'account_id'],
                how='left'
            )\
            .join(
                activity_events.select('tenant_id', 'account_id').distinct().withColumn('has_activity', lit(1)),
                on=['tenant_id', 'account_id'],
                how='left'
            )\
            .join(
                intent_events.select('tenant_id', 'account_id').distinct().withColumn('has_intent', lit(1)),
                on=['tenant_id', 'account_id'],
                how='left'
            )\
            .na.fill(0, ['open_opp', 'has_activity', 'has_intent'])\
            .withColumn('rand', rand(seed=self.CONFIG.seed))

        # Get positive accounts
        pos = account_types.filter(col('open_opp') == 1)\
            .orderBy('rand')\
            .limit(self.CONFIG.sampling.max_positives)
        n_pos = pos.count()
        has_enough_accounts(pos, self.CONFIG.sampling.min_positives, 'positive accounts')

        # Get negative accounts
        # Negatives with activity
        n_neg_w_act = n_pos * self.CONFIG.sampling.neg_w_activity_prop
        neg_w_act = account_types.filter((col('open_opp') == 0) & (col('has_activity') == 1))\
            .orderBy('rand')\
            .limit(n_neg_w_act)
        has_enough_accounts(neg_w_act, n_neg_w_act, 'negative accounts w/ activity')

        # Negatives with intent only
        n_neg_w_int = n_pos * self.CONFIG.sampling.neg_w_intent_prop
        neg_w_int = account_types.filter((col('open_opp') == 0) & (col('has_activity') == 0) &
                                         (col('has_intent') == 1))\
            .orderBy('rand')\
            .limit(n_neg_w_int)
        has_enough_accounts(neg_w_int, n_neg_w_int, 'negative accounts w/ intent only')

        # Negatives with no activity or intent
        n_neg_w_none = n_pos * self.CONFIG.sampling.neg_w_none_prop
        neg_w_none = account_types.filter((col('open_opp') == 0) & (col('has_activity') == 0) &
                                          (col('has_intent') == 0))\
            .orderBy('rand')\
            .limit(n_neg_w_none)
        has_enough_accounts(neg_w_none, n_neg_w_none, 'negative accounts w/o activity or intent')

        account_sample = pos.union(neg_w_act)\
            .union(neg_w_int)\
            .union(neg_w_none)\
            .select('tenant_id', 'account_id')

        # Join back to accounts, which has account-level fields like firmographics
        return account_sample.join(accounts, on=['tenant_id', 'account_id'], how='inner')

    def train_test_split(self, account_sample: DataFrame) -> Tuple[DataFrame, DataFrame]:
        # Not using df.randomSplit() because it can be non-deterministic (even when a seed is set) if the dataframe
        # is not cached beforehand
        x = account_sample.withColumn('rand', rand(seed=self.CONFIG.seed))

        train_size = self.CONFIG.train_size
        train = x.filter(col('rand') <= train_size)
        test = x.filter(col('rand') > train_size)
        print(f'Train accounts: {train.count()}')
        print(f'Test accounts: {test.count()}')
        return train, test

    @staticmethod
    def clean_event_type_names(event_type_col: Column) -> Column:
        return regexp_replace(
            regexp_replace(
                regexp_replace(trim(lower(event_type_col)), '[^a-zA-Z0-9\\|_ ]+', ''),  # remove punctuation
                ' ', '_'  # replace spaces with underscores
            ), '\\|', '__'  # replace pipes with double underscores
        )

    def create_timeline_events(self, accounts: DataFrame, raw_events: DataFrame):
        # 1. Filter to events that occurred between the account's start date and the run date
        # 2. Convert dates to a value on a number line, where the account's start date is 0
        # 3. Clean up event type names, since VectorAssembler can't handle columns with some types of punctuation
        timeline_event = raw_events\
            .join(accounts.select('tenant_id', 'account_id', 'start_dt'), on=['tenant_id', 'account_id'])\
            .filter((col('start_dt') <= col('activity_date')) & (col('activity_date') <= lit(self.run_date)))\
            .withColumn('dt', datediff(col('activity_date'), col('start_dt')) + lit(1))\
            .withColumn('event_type', self.clean_event_type_names(col('event_type')))\
            .select('tenant_id', 'account_id', 'dt', 'event_type', 'weight')

        if self.weekly:
            # Convert days values to week values and re-aggregate
            timeline_event = timeline_event.withColumn('dt', ceil(col('dt') / 7))\
                .groupBy('tenant_id', 'account_id', 'dt', 'event_type')\
                .agg(sum('weight').alias('weight'))

        return timeline_event

    @staticmethod
    def clean_firmographics(accounts: DataFrame) -> DataFrame:
        country_map = create_map([lit(x) for x in chain(*COUNTRY_MAP.items())])

        def normalize_country(c: Column) -> Column:
            # remove punctuation
            cleaned_col = regexp_replace(lower(trim(c)), '[^\\w\\s]', ' ')
            # map variations of country names to 2-digit codes
            return when(~cleaned_col.isin(['', 'unknown']), coalesce(lower(country_map[cleaned_col]), cleaned_col))

        def normalize_industry(c: Column) -> Column:
            cleaned_col = regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(trim(c)), '&', 'and'  # replace ampersand
                    ), '[^\\w\\s]', ' '  # replace punctuation
                ), '\\s+', '_'  # replace whitespace
            )
            return when(~cleaned_col.isin(['', 'unclassified']), cleaned_col)

        def normalize_employee_count(c: Column) -> Column:
            return when(trim(c) != '', log(c.cast(FloatType())))

        def normalize_revenue_range(c: Column) -> Column:
            return when(trim(c).isin('$1 - $1M', '$1M - $5M', '$5M - $10M', '$10M - $25M'), lit('sme'))\
                .when(trim(c).isin('$25M - $50M', '$50M - $100M', '$100M - $250M', '$250M - $500M', '$500M - $1B'),
                      lit('mid_market'))\
                .when(trim(c).isin('$1B - $2.5B', '$2.5B - $5B', 'Over $5B+'), lit('enterprise'))

        def normalize_journey_stage(c: Column) -> Column:
            return regexp_replace(
                regexp_replace(
                    lower(trim(c)), '[^\\w\\s]', ' '  # replace punctuation
                ), '\\s+', '_'  # replace whitespace
            )

        normalized = accounts \
            .withColumn('country_db', normalize_country(col('billingCountry___demandbase')))\
            .withColumn('country_sf', normalize_country(col('billingCountry___salesforce')))\
            .withColumn('country_hs', normalize_country(col('billingCountry___hubspot')))\
            .withColumn('country_dy', normalize_country(col('billingCountry___dynamics')))\
            .withColumn('country', coalesce(col('country_sf'), col('country_hs'),
                                            col('country_dy'), col('country_db'))) \
            .withColumn('industry_db', normalize_industry(col('industry___demandbase')))\
            .withColumn('industry_sf', normalize_industry(col('industry___salesforce')))\
            .withColumn('industry_hs', normalize_industry(col('industry___hubspot')))\
            .withColumn('industry_dy', normalize_industry(col('industry___dynamics')))\
            .withColumn('industry', coalesce(col('industry_db'), col('industry_sf'),
                                             col('industry_hs'), col('industry_dy')))\
            .withColumn('ln_employees_db', normalize_employee_count(col('numberOfEmployees___demandbase')))\
            .withColumn('ln_employees_sf', normalize_employee_count(col('numberOfEmployees___salesforce')))\
            .withColumn('ln_employees_hs', normalize_employee_count(col('numberOfEmployees___hubspot')))\
            .withColumn('ln_employees_dy', normalize_employee_count(col('numberOfEmployees___dynamics')))\
            .withColumn('ln_employees', coalesce(col('ln_employees_db'), col('ln_employees_sf'),
                                                 col('ln_employees_hs'), col('ln_employees_dy'))) \
            .withColumn('revenue_range', normalize_revenue_range(col('revenueRange___demandbase')))\
            .withColumn('initial_journey_stage', normalize_journey_stage(col('initial_journey_stage')))\
            .select('tenant_id', 'account_id', 'domain', 'country', 'industry', 'ln_employees', 'revenue_range',
                    'initial_journey_stage', 'start_dt')
        return normalized

    @staticmethod
    def create_firmographic_events(accounts: DataFrame) -> DataFrame:
        # All firmographic features occur at time 0
        # Stack categorical features as events at start_dt
        # | account_id | country | --> BECOMES --> | account_id | dt | event_type | weight |
        # |        123 |      US | --------------> |        123 |  0 | country=US |      1 |
        cat_events = None
        for feat in ['country', 'industry', 'revenue_range', 'initial_journey_stage']:
            feat_df = accounts.select('tenant_id', 'account_id', feat)\
                .withColumn('dt', lit(0))\
                .withColumn('event_type', concat(lit(f'{feat}__'), col(feat)))\
                .withColumn('weight', lit(1))\
                .drop(feat)
            cat_events = cat_events.unionByName(feat_df) if cat_events is not None else feat_df

        # Set numeric feature values as the weight
        # | account_id | employees | --> BECOMES --> | account_id | dt | event_type | weight |
        # |        123 |       100 | --------------> |        123 |  0 | employees  |    100 |
        num_events = None
        for feat in ['ln_employees']:
            feat_df = accounts.select('tenant_id', 'account_id', feat)\
                .withColumn('dt', lit(0))\
                .withColumn('event_type', lit(feat))\
                .withColumnRenamed(feat, 'weight')
            num_events = num_events.union(feat_df) if num_events is not None else feat_df

        return cat_events.unionByName(num_events).select('tenant_id', 'account_id', 'dt', 'event_type', 'weight')

    def fit_firmographic_roll_up(self, train_accounts: DataFrame) -> Tuple[PipelineModel, DataFrame]:
        transformers = []
        feat_categories = []

        # Keep the top categories for categorical features
        for feat in ['country', 'industry', 'revenue_range', 'initial_journey_stage']:
            if feat in ['country', 'industry']:
                # Keep top N categories
                n = self.CONFIG.firmo_rollup.top_n_categories
            else:
                # Keep all categories
                n = train_accounts.select(feat).filter(col(feat).isNotNull()).distinct().count()
            transformer = TopNCategoryRollUp(input_col=feat, output_col=feat, n_categories=n)
            # transformers.append(transformer)
            categories = transformer.fit(train_accounts).categories
            feat_categories.append((feat, categories))

        # Impute missing values for numeric features
        for feat in ['ln_employees']:
            transformer = Imputer(strategy='median', inputCol=feat, outputCol=feat)
            transformers.append(transformer)

        firmo_rollup = Pipeline(stages=transformers)
        return (firmo_rollup.fit(train_accounts),
                self.spark.createDataFrame(feat_categories, schema=['feature', 'categories']))

    def fit_activity_roll_up(self, train_activities: DataFrame) -> DataFrame:
        transformer = ActivityRoleCategoryRollUp(
            min_occurrences=self.CONFIG.activity_rollup.min_occurrences,
            min_accounts=self.CONFIG.activity_rollup.min_accounts,
        )
        # pipeline = Pipeline(stages=[transformer])
        # return pipeline.fit(train_activities)
        activities = transformer.fit(train_activities).activities
        return self.spark.createDataFrame([('activities', activities)], schema=['feature', 'categories'])

    @staticmethod
    def combine_events(accounts: DataFrame, firmo_events: DataFrame, opp_events: DataFrame,
                       intent_events: DataFrame, activities_rollup: DataFrame) -> DataFrame:
        # Gather all events for the account list into one dataframe
        all_events = None
        for df in [firmo_events, opp_events, intent_events, activities_rollup]:
            df = df.join(accounts.select('tenant_id', 'account_id'), on=['tenant_id', 'account_id'], how='inner')
            all_events = all_events.unionByName(df) if all_events else df
        return all_events.filter(col('event_type').isNotNull())\
            .orderBy('tenant_id', 'account_id', 'dt')

    @staticmethod
    def merge_timestamp_events_to_dict(all_events: DataFrame) -> DataFrame:
        # Convert events to dictionary at each date
        event_agg = all_events.groupBy('tenant_id', 'account_id', 'dt')\
            .agg(map_from_arrays(collect_list(col('event_type')), collect_list(col('weight'))).alias('events'))
        return event_agg

    def pad_final_event_date(self, accounts: DataFrame, event_agg: DataFrame) -> DataFrame:
        divisor = 7 if self.weekly else 1
        # 1. Get the max possible date (length of observation period) for each account
        # 2. Join to event_agg and drop accounts that already have an event vector on the max date
        # 3. Add an empty event vector on the max date for remaining accounts
        # 4. Union back to event_agg to get the full set of padded sequences for each account
        return accounts\
            .withColumn('max_dt', ceil((datediff(lit(self.run_date), col('start_dt')) + lit(1)) / lit(divisor))) \
            .select('tenant_id', 'account_id', 'max_dt')\
            .join(event_agg, on=['tenant_id', 'account_id'], how='inner')\
            .groupBy('tenant_id', 'account_id', 'max_dt')\
            .agg(max('dt').alias('last_event_dt'))\
            .filter(col('last_event_dt') != col('max_dt'))\
            .withColumnRenamed('max_dt', 'dt')\
            .withColumn('events', create_map().cast(MapType(StringType(), DoubleType(), False)))\
            .select('tenant_id', 'account_id', 'dt', 'events')\
            .unionByName(event_agg)\
            .orderBy('tenant_id', 'account_id', 'dt')

    def convert_to_sparse_matrices(self, padded_event_agg: DataFrame) -> DataFrame:
        event_type_names_list = self.event_type_names.coalesce(1).rdd.map(lambda x: x['event_type']).collect()

        # The EMR notebook never works correctly with any UDF, so we have to do this the hard way below
        # def sparse_vector(dt, event_type_map):
        #     # get (event_type index, weight) pairs for event_types that occurred on this date
        #     idx_weight_tuples = [(i+1, event_type_map[event]) for i, event in enumerate(event_type_names_list)
        #                          if event in event_type_map]
        #     # insert the date at index 0, then follow with event_type indices and weights
        #     return SparseVector(
        #         self.n_event_types + 1,
        #         [(0, dt)] + idx_weight_tuples
        #     )
        # sparse_vector_udf = udf(sparse_vector, VectorUDT())

        # Explode each feature into its own column
        for feat in event_type_names_list:
            padded_event_agg = padded_event_agg.withColumn(feat, col('events').getItem(feat))
        padded_event_agg = padded_event_agg.na.fill(0)

        # Convert to sparse vector
        # It doesn't matter if we're training or not because the input columns are created
        # using event_type_names, which is created from the training data.
        # Since event_type_names is saved during training, it's not necessary to save the VectorAssembler
        vec_assembler = VectorAssembler(
            inputCols=['dt'] + event_type_names_list,
            outputCol='event_sparse_vector'
        )

        return vec_assembler.transform(padded_event_agg)\
            .select('tenant_id', 'account_id', 'dt', 'event_sparse_vector')

    @staticmethod
    def save_parquet(df: DataFrame, path: str) -> None:
        df.write.mode('overwrite').parquet(path)

    @staticmethod
    def save_json(df: DataFrame, path: str) -> None:
        df.coalesce(1).write.mode('overwrite').json(path)
