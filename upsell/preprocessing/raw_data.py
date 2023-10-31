from pyspark.sql.functions import *
from pyspark.sql import DataFrame, SparkSession, Window


class Account:
  tenant_id: str,
  account_id: str,
  domain: str,
  billingCountry___demandbase: str,
  billingCountry___salesforce: str,
  billingCountry___hubspot: str,
  billingCountry___dynamics: str,
  industry___demandbase: str,
  industry___salesforce: str,
  industry___hubspot: str,
  industry___dynamics: str,
  numberOfEmployees___demandbase: str,
  numberOfEmployees___salesforce: str,
  numberOfEmployees___hubspot: str,
  numberOfEmployees___dynamics: str,
  revenueRange___demandbase: str,
  start_dt: str


class Event:
  tenant_id: str,
  account_id: str,
  activity_date: str,
  event_type: str,
  weight: float


class Metrics:
  tenant_id: str,
  accounts: int,
  activities: int,
  oppEvents: int,
  intentEvents: int,
  accountsWithActivity: int,
  accountsWithOppEvent: int,
  accountsWithIntentEvent: int,
  accountsWithNoEvents: int,
  pctAccountsNoEvents: float


class RawEvents:
  def getAccounts(self, spark: SparkSession, tenant_id: str, runDate: str, activityMonths: int) -> Dataset[Account]:
    accountQuery = f"""
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
      WHERE tenant_id = '{tenant_id}'
      AND domain___e IS NOT NULL
    """
    accounts = spark.sql(accountQuery)

    # Get account's first journey stage
    journeyQuery = f"""
      WITH first_stage_dt AS (
        -- First date we started collecting data on the account
        SELECT tenant_id
          , account_id
          , MIN(enteredAt) AS enteredAt
        FROM db1_data_warehouse.tenant.journey
        WHERE tenant_id = '{tenant_id}'
        AND enteredAt <= TIMESTAMP('{runDate}')
        GROUP BY 1, 2
      )
      , latest_stage_dt AS (
        -- Last journey stage change before data collection began (if exists)
        SELECT tenant_id
          , account_id
          , MAX(enteredAt) AS enteredAt
        FROM db1_data_warehouse.tenant.journey
        WHERE tenant_id = '{tenant_id}'
        AND enteredAt <= TIMESTAMP(ADD_MONTHS('{runDate}', -{activityMonths}))
        GROUP BY 1, 2
      )
      , final_dts AS (
        SELECT tenant_id
          , account_id
          , CASE WHEN DATE(f.enteredAt) > DATE(ADD_MONTHS('{runDate}', -{activityMonths})) THEN DATE(f.enteredAt)
            ELSE DATE(ADD_MONTHS('{runDate}', -{activityMonths})) END AS start_dt
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
      WHERE j.tenant_id = '{tenant_id}'
    """
    journeys = spark.sql(journeyQuery)

    return accounts.join(journeys, on=["tenant_id", "account_id"]).as[Account]

  def getOpportunities(spark: SparkSession, tenant_id: String, runDate: String, activityMonths: Int): Dataset[Event] = {
    val opportunityQuery =
      s"""
      WITH all_opps AS (
        SELECT tenant_id
          , account_id
          , id AS opp_id
          , DATE(createdDate) AS createdDate
          , DATE(closeDate) AS closeDate
          , isWon
          , CASE WHEN DATE(createdDate) > DATE(closeDate) THEN 1 ELSE 0 END AS backdated
          , ADD_MONTHS('$runDate', -$activityMonths) AS start_dt
          , '$runDate' AS end_dt
        FROM db1_data_warehouse.tenant.opportunity
        WHERE tenant_id = '$tenant_id'
        ORDER BY 1, 2, 3, 4
      )
      SELECT *
      FROM all_opps
      WHERE (
        -- Opened during the customer period
        (start_dt <= createdDate AND createdDate <= end_dt)
        -- Closed during the customer period
        OR (start_dt <= closeDate AND closeDate <= end_dt)
        )
    """
    val opportunities = spark.sql(opportunityQuery)

    // Get journey stages to check if account was a customer when opp was opened
    val journeyQuery =
      s"""
      SELECT tenant_id
        , account_id
        , DATE(enteredAt) AS enteredAt
        , stageName
        , CASE WHEN LOWER(stageName) RLIKE 'customer|expansion|closed won|existing|upsell'
          AND LOWER(stageName) NOT RLIKE 'lost' THEN 1 ELSE 0 END AS customer_stage
      FROM db1_data_warehouse.tenant.journey
      WHERE tenant_id = '$tenant_id'
      AND enteredAt <= TIMESTAMP('$runDate')
    """
    val journeys = spark.sql(journeyQuery)

    val oppWithJourneyStage = opportunities
      .join(journeys, Seq("tenant_id", "account_id"))
      .filter(col("enteredAt") <= col("createdDate"))
      .withColumn(
        "rank",
        row_number() over Window.partitionBy("tenant_id", "account_id", "opp_id").orderBy(col("enteredAt").desc)
      )
      .filter(col("rank") === 1)
      .drop("rank")
      .orderBy("tenant_id", "account_id", "opp_id", "enteredAt")

    val opened = oppWithJourneyStage
      .filter(col("createdDate") >= col("start_dt"))
      .filter(col("createdDate") <= col("end_dt"))
      .withColumn(
        "event_type",
        when(col("customer_stage") === 1, lit("Opened Post-Customer Opportunity"))
          .otherwise(lit("Opened New Business Opportunity"))
      )
      .select("tenant_id", "account_id", "createdDate", "event_type")
      .withColumnRenamed("createdDate", "activity_date")

    val closed = oppWithJourneyStage
      .filter(col("closeDate") >= col("start_dt"))
      .filter(col("closeDate") <= col("end_dt"))
      .withColumn(
        "event_type",
        when((col("customer_stage") === 1) && col("isWon"), lit("Closed/Won Post-Customer Opportunity"))
          .when((col("customer_stage") === 1) && !col("isWon"), lit("Closed/Lost Post-Customer Opportunity"))
          .when(col("isWon"), lit("Closed/Won New Business Opportunity"))
          .when(!col("isWon"), lit("Closed/Lost New Business Opportunity"))
      )
      .select("tenant_id", "account_id", "createdDate", "event_type")
      .withColumnRenamed("createdDate", "activity_date")

    opened
      .unionByName(closed)
      .groupBy("tenant_id", "account_id", "activity_date", "event_type")
      .agg(count("*").as("weight"))
      .as[Event]
  }

  def getActivities(spark: SparkSession, tenant_id: String, runDate: String, activityMonths: Int): Dataset[Event] = {
    val activityQuery =
      s"""
      -- Page Visits (Anonymous)
      SELECT tenant_id
        , account_id
        , DATE(activity_date) AS activity_date
        , CONCAT(activityType, '|', _per_role, '|', category___e) AS event_type
        , COUNT(*) AS weight
      FROM db1_data_warehouse.tenant.activity_bundle
      WHERE tenant_id = '$tenant_id'
      AND activity_date <= TIMESTAMP('$runDate')
      AND activity_date >= TIMESTAMP(ADD_MONTHS('$runDate', -$activityMonths))
      AND activity_source_type = 'web'
      AND engagement > 0
      GROUP BY 1, 2, 3, 4

      UNION ALL

      -- Activities like clicks, form fills, page visits (non-anonymous)
      SELECT tenant_id
        , account_id
        , DATE(activity_date) AS activity_date
        , CONCAT(activityType, '|', _per_role, '|', category___e) AS event_type
        , COUNT(*) AS weight
      FROM db1_data_warehouse.tenant.activity_bundle
      WHERE tenant_id = '$tenant_id'
      AND activity_date <= TIMESTAMP('$runDate')
      AND activity_date >= TIMESTAMP(ADD_MONTHS('$runDate', -$activityMonths))
      AND activity_source_type = 'db1-platform'
      AND activityType NOT IN ('Intent Surge')
      AND engagement > 0
      GROUP BY 1, 2, 3, 4
    """
    spark.sql(activityQuery).as[Event]
  }

  def getIntentEvents(spark: SparkSession, tenant_id: String, runDate: String, intentMonths: Int): Dataset[Event] = {
    // TODO: Add activity_source_type = 'trending_intent'
    val intentActivitiesQuery =
      s"""
      SELECT tenant_id
        , account_id
        , DATE(activity_date) AS activity_date
        , CONCAT(activityType, '|', _per_role, '|', category___e) AS event_type
        , COUNT(*) AS weight
      FROM db1_data_warehouse.tenant.activity_bundle
      WHERE tenant_id = '$tenant_id'
      AND activity_date <= TIMESTAMP('$runDate')
      AND activity_date >= TIMESTAMP(ADD_MONTHS('$runDate', -$intentMonths))
      AND activity_source_type = 'db1-platform'
      AND activityType IN ('Intent Surge')
      AND engagement > 0
      GROUP BY 1, 2, 3, 4
    """
    val keywordIntentQuery =
      s"""
      SELECT tenant_id
        , account_id
        , DATE(activity_date) AS activity_date
        , low_intent_keywords
        , medium_intent_keywords
        , high_intent_keywords
      FROM db1_data_warehouse.tenant.activity_bundle
      WHERE tenant_id = '$tenant_id'
      AND activity_date <= TIMESTAMP('$runDate')
      AND activity_date >= TIMESTAMP(ADD_MONTHS('$runDate', -$intentMonths))
      AND activity_source_type = 'intent'
    """
    val keywordIntentDf = spark.sql(keywordIntentQuery)

    def unpack_keywords(strength: String): Dataset[Event] = keywordIntentDf
      .select("tenant_id", "account_id", "activity_date", s"${strength}_intent_keywords")
      .withColumn("keyword", explode(col(s"${strength}_intent_keywords")))
      .drop(s"${strength}_intent_keywords")
      .distinct
      .withColumn("event_type", lit(s"DB Keyword Intent|${strength.toUpperCase}"))
      .groupBy("tenant_id", "account_id", "activity_date", "event_type")
      .agg(count("keyword").as("weight"))
      .as[Event]

    val intentActivitiesDf = spark.sql(intentActivitiesQuery).as[Event]

    Array("high", "medium", "low")
      .map(strength => unpack_keywords(strength))
      .reduce(_.unionByName(_))
      .unionByName(intentActivitiesDf)
      .as[Event]
  }

  def calculateMetrics(
                        accounts: Dataset[Account],
                        activities: Dataset[Event],
                        oppEvents: Dataset[Event],
                        intentEvents: Dataset[Event]
                      ): Dataset[Metrics] = {
    val t = accounts.select("tenant_id").distinct
    val n = accounts.agg(count("*").as("accounts"))
    val n_activities = activities.agg(sum("weight").as("activities")).na.fill(0)
    val n_opp_events = oppEvents.agg(sum("weight").as("oppEvents")).na.fill(0)
    val n_intent_events = intentEvents.agg(sum("weight").as("intentEvents")).na.fill(0)

    val n_acct_w_activity = activities.agg(countDistinct("account_id").as("accountsWithActivity"))
    val n_acct_w_opp = oppEvents.agg(countDistinct("account_id").as("accountsWithOppEvent"))
    val n_acct_w_intent = intentEvents.agg(countDistinct("account_id").as("accountsWithIntentEvent"))

    val n_acct_no_events = accounts
      .join(activities, Seq("tenant_id", "account_id"), "left")
      .filter(col("weight").isNull)
      .select("tenant_id", "account_id")
      .join(oppEvents, Seq("tenant_id", "account_id"), "left")
      .filter(col("weight").isNull)
      .select("tenant_id", "account_id")
      .join(intentEvents, Seq("tenant_id", "account_id"), "left")
      .filter(col("weight").isNull)
      .agg(count("*").as("accountsWithNoEvents"))

    t.join(n)
      .join(n_activities)
      .join(n_opp_events)
      .join(n_intent_events)
      .join(n_acct_w_activity)
      .join(n_acct_w_opp)
      .join(n_acct_w_intent)
      .join(n_acct_no_events)
      .withColumn("pctAccountsWActivity", col("accountsWithActivity") / col("accounts"))
      .withColumn("pctAccountsWOppEvent", col("accountsWithOppEvent") / col("accounts"))
      .withColumn("pctAccountsWIntentEvent", col("accountsWithIntentEvent") / col("accounts"))
      .withColumn("pctAccountsNoEvents", col("accountsWithNoEvents") / col("accounts"))
      .as[Metrics]
  }

  def saveParquet(df: Dataset[_], bucket: String, tenant_id: String, runDate: String, name: String): Unit = df.write
    .mode("overwrite")
    .parquet(s"s3://$bucket/upsell/$tenant_id/$runDate/$name/")

  def saveJson(df: Dataset[_], bucket: String, tenant_id: String, runDate: String, name: String): Unit = df
    .coalesce(1)
    .write
    .mode("overwrite")
    .json(s"s3://$bucket/upsell/$tenant_id/$runDate/$name/")

  def main(
            spark: SparkSession,
            bucket: String,
            tenant_id: String,
            runDate: String,
            activityMonths: Int,
            intentMonths: Int,
            includeIntent: Boolean = false
          ): Unit = {

    val accounts = getAccounts(spark, tenant_id, runDate, activityMonths).cache
    val oppEvents = getOpportunities(spark, tenant_id, runDate, activityMonths).cache
    val activities = getActivities(spark, tenant_id, runDate, activityMonths).cache

    val intentEvents =
      if (includeIntent) getIntentEvents(spark, tenant_id, runDate, intentMonths).cache
      else spark.emptyDataset[Event]

    val metrics = calculateMetrics(accounts, activities, oppEvents, intentEvents).cache

    saveParquet(accounts, bucket, tenant_id, runDate, "accounts")
    saveParquet(oppEvents, bucket, tenant_id, runDate, "oppEvents")
    saveParquet(activities, bucket, tenant_id, runDate, "activities")
    saveParquet(intentEvents, bucket, tenant_id, runDate, "intentEvents")
    saveJson(metrics, bucket, tenant_id, runDate, "metrics/events/")
  }

  def loop(spark: SparkSession): Unit = {
    val tenant_ids = List("1309", "1619", "1681", "2874", "5715", "6114", "11640", "12636", "13279", "13574")
    val runDate = "2023-07-01"
    val activityMonths = 12
    val intentMonths = 3
    val bucket = "ceasterwood"
    val includeIntent = true

    // Create datasets for each tenant and save to S3
    tenant_ids.foreach(tenant_id =>
      main(spark, bucket, tenant_id, runDate, activityMonths, intentMonths, includeIntent)
    )

    // Combine metrics datasets and save to S3
    tenant_ids
      .map(tenant_id => spark.read.json(s"s3://$bucket/upsell/$tenant_id/$runDate/metrics/events/"))
      .reduce(_.unionByName(_))
      .coalesce(1)
      .write
      .mode("overwrite")
      .json(s"s3://$bucket/upsell/all-tenant-metrics/$runDate/events/")
  }

