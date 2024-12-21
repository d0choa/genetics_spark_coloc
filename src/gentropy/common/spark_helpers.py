"""Common utilities in Spark that can be used across the project."""

from __future__ import annotations

import re
import sys
from functools import reduce, wraps
from itertools import chain
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

import pyspark.sql.functions as f
import pyspark.sql.types as t
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Column, Row, Window
from scipy.stats import norm

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, WindowSpec


def convert_from_wide_to_long(
    df: DataFrame,
    id_vars: Iterable[str],
    var_name: str,
    value_name: str,
    value_vars: Optional[Iterable[str]] = None,
) -> DataFrame:
    """Converts a dataframe from wide to long format.

    Args:
        df (DataFrame): Dataframe to melt
        id_vars (Iterable[str]): List of fixed columns to keep
        var_name (str): Name of the column containing the variable names
        value_name (str): Name of the column containing the values
        value_vars (Optional[Iterable[str]]): List of columns to melt. Defaults to None.

    Returns:
        DataFrame: Melted dataframe

    Examples:
    >>> df = spark.createDataFrame([("a", 1, 2)], ["id", "feature_1", "feature_2"])
    >>> convert_from_wide_to_long(df, ["id"], "feature", "value").show()
    +---+---------+-----+
    | id|  feature|value|
    +---+---------+-----+
    |  a|feature_1|  1.0|
    |  a|feature_2|  2.0|
    +---+---------+-----+
    <BLANKLINE>
    """
    if not value_vars:
        value_vars = [c for c in df.columns if c not in id_vars]
    _vars_and_vals = f.array(
        *(
            f.struct(
                f.lit(c).alias(var_name), f.col(c).cast(t.FloatType()).alias(value_name)
            )
            for c in value_vars
        )
    )

    # Add to the DataFrame and explode to convert into rows
    _tmp = df.withColumn("_vars_and_vals", f.explode(_vars_and_vals))

    cols = list(id_vars) + [
        f.col("_vars_and_vals")[x].alias(x) for x in [var_name, value_name]
    ]
    return _tmp.select(*cols)


def convert_from_long_to_wide(
    df: DataFrame, id_vars: list[str], var_name: str, value_name: str
) -> DataFrame:
    """Converts a dataframe from long to wide format using Spark pivot built-in function.

    Args:
        df (DataFrame): Dataframe to pivot
        id_vars (list[str]): List of fixed columns to keep
        var_name (str): Name of the column to pivot on
        value_name (str): Name of the column containing the values

    Returns:
        DataFrame: Pivoted dataframe

    Examples:
    >>> df = spark.createDataFrame([("a", "feature_1", 1), ("a", "feature_2", 2)], ["id", "featureName", "featureValue"])
    >>> convert_from_long_to_wide(df, ["id"], "featureName", "featureValue").show()
    +---+---------+---------+
    | id|feature_1|feature_2|
    +---+---------+---------+
    |  a|        1|        2|
    +---+---------+---------+
    <BLANKLINE>
    """
    return df.groupBy(id_vars).pivot(var_name).agg(f.first(value_name))


def pvalue_to_zscore(pval_col: Column) -> Column:
    """Convert p-value column to z-score column.

    Args:
        pval_col (Column): pvalues to be casted to floats.

    Returns:
        Column: p-values transformed to z-scores

    Examples:
        >>> d = [{"id": "t1", "pval": "1"}, {"id": "t2", "pval": "0.9"}, {"id": "t3", "pval": "0.05"}, {"id": "t4", "pval": "1e-300"}, {"id": "t5", "pval": "1e-1000"}, {"id": "t6", "pval": "NA"}]
        >>> df = spark.createDataFrame(d)
        >>> df.withColumn("zscore", pvalue_to_zscore(f.col("pval"))).show()
        +---+-------+----------+
        | id|   pval|    zscore|
        +---+-------+----------+
        | t1|      1|       0.0|
        | t2|    0.9|0.12566137|
        | t3|   0.05|  1.959964|
        | t4| 1e-300| 37.537838|
        | t5|1e-1000| 37.537838|
        | t6|     NA|      null|
        +---+-------+----------+
        <BLANKLINE>

    """
    pvalue_float = pval_col.cast(t.FloatType())
    pvalue_nozero = f.when(pvalue_float == 0, sys.float_info.min).otherwise(
        pvalue_float
    )
    return f.udf(
        lambda pv: float(abs(norm.ppf((float(pv)) / 2))) if pv else None,
        t.FloatType(),
    )(pvalue_nozero)


def nullify_empty_array(column: Column) -> Column:
    """Returns null when a Spark Column has an array of size 0, otherwise return the array.

    Args:
        column (Column): The Spark Column to be processed.

    Returns:
        Column: Nullified column when the array is empty.

    Examples:
    >>> df = spark.createDataFrame([[], [1, 2, 3]], "array<int>")
    >>> df.withColumn("new", nullify_empty_array(df.value)).show()
    +---------+---------+
    |    value|      new|
    +---------+---------+
    |       []|     null|
    |[1, 2, 3]|[1, 2, 3]|
    +---------+---------+
    <BLANKLINE>
    """
    return f.when(f.size(column) != 0, column)


def get_top_ranked_in_window(df: DataFrame, w: WindowSpec) -> DataFrame:
    """Returns the record with the top rank within each group of the window.

    Args:
        df (DataFrame): The DataFrame to be processed.
        w (WindowSpec): The window to be used for ranking.

    Returns:
        DataFrame: The DataFrame with the record with the top rank within each group of the window.
    """
    return (
        df.withColumn("row_number", f.row_number().over(w))
        .filter(f.col("row_number") == 1)
        .drop("row_number")
    )


def get_record_with_minimum_value(
    df: DataFrame,
    grouping_col: Column | str | list[Column | str],
    sorting_col: str,
) -> DataFrame:
    """Returns the record with the minimum value of the sorting column within each group of the grouping column.

    Args:
        df (DataFrame): The DataFrame to be processed.
        grouping_col (Column | str | list[Column | str]): The column(s) to group the DataFrame by.
        sorting_col (str): The column name to sort the DataFrame by.

    Returns:
        DataFrame: The DataFrame with the record with the minimum value of the sorting column within each group of the grouping column.
    """
    w = Window.partitionBy(grouping_col).orderBy(sorting_col)
    return get_top_ranked_in_window(df, w)


def get_record_with_maximum_value(
    df: DataFrame,
    grouping_col: str | list[str],
    sorting_col: str,
) -> DataFrame:
    """Returns the record with the maximum value of the sorting column within each group of the grouping column.

    Args:
        df (DataFrame): The DataFrame to be processed.
        grouping_col (str | list[str]): The column(s) to group the DataFrame by.
        sorting_col (str): The column name to sort the DataFrame by.

    Returns:
        DataFrame: The DataFrame with the record with the maximum value of the sorting column within each group of the grouping column.
    """
    w = Window.partitionBy(grouping_col).orderBy(f.col(sorting_col).desc())
    return get_top_ranked_in_window(df, w)


def normalise_column(
    df: DataFrame, input_col_name: str, output_col_name: str
) -> DataFrame:
    """Normalises a numerical column to a value between 0 and 1.

    Args:
        df (DataFrame): The DataFrame to be processed.
        input_col_name (str): The name of the column to be normalised.
        output_col_name (str): The name of the column to store the normalised values.

    Returns:
        DataFrame: The DataFrame with the normalised column.

    Examples:
    >>> df = spark.createDataFrame([5, 50, 1000], "int")
    >>> df.transform(lambda df: normalise_column(df, "value", "norm_value")).show()
    +-----+----------+
    |value|norm_value|
    +-----+----------+
    |    5|       0.0|
    |   50|      0.05|
    | 1000|       1.0|
    +-----+----------+
    <BLANKLINE>
    """
    vec_assembler = VectorAssembler(
        inputCols=[input_col_name], outputCol="feature_vector"
    )
    scaler = MinMaxScaler(inputCol="feature_vector", outputCol="norm_vector")
    unvector_score = f.round(vector_to_array(f.col("norm_vector"))[0], 2).alias(
        output_col_name
    )
    pipeline = Pipeline(stages=[vec_assembler, scaler])
    return (
        pipeline.fit(df)
        .transform(df)
        .select("*", unvector_score)
        .drop("feature_vector", "norm_vector")
    )


def neglog_pvalue_to_mantissa_and_exponent(p_value: Column) -> tuple[Column, Column]:
    """Computing p-value mantissa and exponent based on the negative 10 based logarithm of the p-value.

    Args:
        p_value (Column): Neg-log p-value (string)

    Returns:
        tuple[Column, Column]: mantissa and exponent of the p-value

    Examples:
        >>> (
        ... spark.createDataFrame([(4.56, 'a'),(2109.23, 'b')], ['negLogPv', 'label'])
        ... .select('negLogPv',*neglog_pvalue_to_mantissa_and_exponent(f.col('negLogPv')))
        ... .show()
        ... )
        +--------+--------------+--------------+
        |negLogPv|pValueMantissa|pValueExponent|
        +--------+--------------+--------------+
        |    4.56|     2.7542286|            -5|
        | 2109.23|     5.8884363|         -2110|
        +--------+--------------+--------------+
        <BLANKLINE>
    """
    exponent: Column = f.ceil(p_value)
    mantissa: Column = f.pow(f.lit(10), (exponent - p_value))

    return (
        mantissa.cast(t.FloatType()).alias("pValueMantissa"),
        (-1 * exponent).cast(t.IntegerType()).alias("pValueExponent"),
    )


def calculate_neglog_pvalue(
    p_value_mantissa: Column, p_value_exponent: Column
) -> Column:
    """Compute the negative log p-value.

    Args:
        p_value_mantissa (Column): P-value mantissa
        p_value_exponent (Column): P-value exponent

    Returns:
        Column: Negative log p-value

    Examples:
        >>> d = [(1, 1), (5, -2), (1, -1000)]
        >>> df = spark.createDataFrame(d).toDF("p_value_mantissa", "p_value_exponent")
        >>> df.withColumn("neg_log_p", calculate_neglog_pvalue(f.col("p_value_mantissa"), f.col("p_value_exponent"))).show()
        +----------------+----------------+------------------+
        |p_value_mantissa|p_value_exponent|         neg_log_p|
        +----------------+----------------+------------------+
        |               1|               1|              -1.0|
        |               5|              -2|1.3010299956639813|
        |               1|           -1000|            1000.0|
        +----------------+----------------+------------------+
        <BLANKLINE>
    """
    return -1 * (f.log10(p_value_mantissa) + p_value_exponent)


def string2camelcase(col_name: str) -> str:
    """Converting a string to camelcase.

    Args:
        col_name (str): a random string

    Returns:
        str: Camel cased string

    Examples:
        >>> string2camelcase("hello_world")
        'helloWorld'
        >>> string2camelcase("hello world")
        'helloWorld'
    """
    # Removing a bunch of unwanted characters from the column names:
    col_name_normalised = re.sub(r"[\/\(\)\-]+", " ", col_name)

    first, *rest = re.split("[ _-]", col_name_normalised)
    return "".join([first.lower(), *map(str.capitalize, rest)])


def column2camel_case(col_name: str) -> str:
    """A helper function to convert column names to camel cases.

    Args:
        col_name (str): a single column name

    Returns:
        str: spark expression to select and rename the column

    Examples:
        >>> column2camel_case("hello_world")
        '`hello_world` as helloWorld'
    """
    return f"`{col_name}` as {string2camelcase(col_name)}"


def order_array_of_structs_by_field(column_name: str, field_name: str) -> Column:
    """Sort a column of array of structs by a field in descending order, nulls last.

    Args:
        column_name (str): Column name
        field_name (str): Field name

    Returns:
        Column: Sorted column
    """
    return f.expr(
        f"""
        array_sort(
        {column_name},
        (left, right) -> case
                        when left.{field_name} is null and right.{field_name} is null then 0
                        when left.{field_name} is null then 1
                        when right.{field_name} is null then -1
                        when left.{field_name} < right.{field_name} then 1
                        when left.{field_name} > right.{field_name} then -1
                        else 0
                end)
        """
    )


def order_array_of_structs_by_two_fields(
    array_name: str, descending_column: str, ascending_column: str
) -> Column:
    """Sort array of structs by a field in descending order and by an other field in an ascending order.

    This function doesn't deal with null values, assumes the sort columns are not nullable.
    The sorting function compares the descending_column first, in case when two values from descending_column are equal
    it compares the ascending_column. When values in both columns are equal, the rows order is preserved.

    Args:
        array_name (str): Column name with array of structs
        descending_column (str): Name of the first keys sorted in descending order
        ascending_column (str): Name of the second keys sorted in ascending order

    Returns:
        Column: Sorted column

    Examples:
    >>> data = [(1.0, 45, 'First'), (0.5, 232, 'Third'), (0.5, 233, 'Fourth'), (1.0, 125, 'Second'),]
    >>> (
    ...    spark.createDataFrame(data, ['col1', 'col2', 'ranking'])
    ...    .groupBy(f.lit('c'))
    ...    .agg(f.collect_list(f.struct('col1','col2', 'ranking')).alias('list'))
    ...    .select(order_array_of_structs_by_two_fields('list', 'col1', 'col2').alias('sorted_list'))
    ...    .show(truncate=False)
    ... )
    +-----------------------------------------------------------------------------+
    |sorted_list                                                                  |
    +-----------------------------------------------------------------------------+
    |[{1.0, 45, First}, {1.0, 125, Second}, {0.5, 232, Third}, {0.5, 233, Fourth}]|
    +-----------------------------------------------------------------------------+
    <BLANKLINE>
    >>> data = [(1.0, 45, 'First'), (1.0, 45, 'Second'), (0.5, 233, 'Fourth'), (1.0, 125, 'Third'),]
    >>> (
    ...    spark.createDataFrame(data, ['col1', 'col2', 'ranking'])
    ...    .groupBy(f.lit('c'))
    ...    .agg(f.collect_list(f.struct('col1','col2', 'ranking')).alias('list'))
    ...    .select(order_array_of_structs_by_two_fields('list', 'col1', 'col2').alias('sorted_list'))
    ...    .show(truncate=False)
    ... )
    +----------------------------------------------------------------------------+
    |sorted_list                                                                 |
    +----------------------------------------------------------------------------+
    |[{1.0, 45, First}, {1.0, 45, Second}, {1.0, 125, Third}, {0.5, 233, Fourth}]|
    +----------------------------------------------------------------------------+
    <BLANKLINE>
    """
    return f.expr(
        f"""
        array_sort(
        {array_name},
        (left, right) -> case
                when left.{descending_column} is null and right.{descending_column} is null then 0
                when left.{ascending_column} is null and right.{ascending_column} is null then 0

                when left.{descending_column} is null then 1
                when right.{descending_column} is null then -1

                when left.{ascending_column} is null then 1
                when right.{ascending_column} is null then -1

                when left.{descending_column} < right.{descending_column} then 1
                when left.{descending_column} > right.{descending_column} then -1
                when left.{descending_column} == right.{descending_column} and left.{ascending_column} > right.{ascending_column} then 1
                when left.{descending_column} == right.{descending_column} and left.{ascending_column} < right.{ascending_column} then -1
                when left.{ascending_column} == right.{ascending_column} and left.{descending_column} == right.{descending_column} then 0
        end)
        """
    )


def map_column_by_dictionary(col: Column, mapping_dict: dict[str, Any]) -> Column:
    """Map column values to dictionary values by key.

    Missing consequence label will be converted to None, unmapped consequences will be mapped as None.

    Args:
        col (Column): Column containing labels to map.
        mapping_dict (dict[str, Any]): Dictionary with mapping key/value pairs.

    Returns:
        Column: Column with mapped values.

    Examples:
        >>> data = [('consequence_1',),('unmapped_consequence',),(None,)]
        >>> m = {'consequence_1': 'SO:000000'}
        >>> (
        ...    spark.createDataFrame(data, ['label'])
        ...    .select('label',map_column_by_dictionary(f.col('label'),m).alias('id'))
        ...    .show()
        ... )
        +--------------------+---------+
        |               label|       id|
        +--------------------+---------+
        |       consequence_1|SO:000000|
        |unmapped_consequence|     null|
        |                null|     null|
        +--------------------+---------+
        <BLANKLINE>
    """
    map_expr = f.create_map(*[f.lit(x) for x in chain(*mapping_dict.items())])

    return map_expr[col]


def pivot_df(
    df: DataFrame,
    pivot_col: str,
    value_col: str,
    grouping_cols: list[Column],
) -> DataFrame:
    """Pivot a dataframe.

    Args:
        df (DataFrame): Dataframe to pivot
        pivot_col (str): Column to pivot on
        value_col (str): Column to pivot
        grouping_cols (list[Column]): Columns to group by

    Returns:
        DataFrame: Pivoted dataframe
    """
    pivot_values = df.select(pivot_col).distinct().rdd.flatMap(lambda x: x).collect()
    return (
        df.groupBy(grouping_cols)
        .pivot(pivot_col)
        .agg({value_col: "first"})
        .select(
            grouping_cols
            + [
                f.when(f.col(x).isNull(), None)
                .otherwise(f.col(x))
                .alias(f"{x}_{value_col}")
                for x in pivot_values
            ],
        )
    )


def get_value_from_row(row: Row, column: str) -> Any:
    """Extract index value from a row if exists.

    Args:
        row (Row): One row from a dataframe
        column (str): column label we want to extract.

    Returns:
        Any: value of the column in the row

    Raises:
        ValueError: if the column is not in the row

    Examples:
        >>> get_value_from_row(Row(geneName="AR", chromosome="X"), "chromosome")
        'X'
        >>> get_value_from_row(Row(geneName="AR", chromosome="X"), "disease")
        Traceback (most recent call last):
        ...
        ValueError: Column disease not found in row Row(geneName='AR', chromosome='X')
    """
    if column not in row:
        raise ValueError(f"Column {column} not found in row {row}")
    return row[column]


def enforce_schema(
    expected_schema: t.ArrayType | t.StructType | Column | str,
) -> Callable[..., Any]:
    """A function to enforce the schema of a function output follows expectation.

    Behaviour:
        - Fields that are not present in the expected schema will be dropped.
        - Expected but missing fields will be added with Null values.
        - Fields with incorrect data types will be casted to the expected data type.

    This is a decorator function and expected to be used like this:

    @enforce_schema(spark_schema)
    def my_function() -> t.StructType:
        return ...

    Args:
        expected_schema (t.ArrayType | t.StructType | Column | str): The expected schema of the output.

    Returns:
        Callable[..., Any]: A decorator function.
    """
    T = TypeVar("T", str, Column)

    def decorator(function: Callable[..., T]) -> Callable[..., T]:
        """A decorator function to enforce the schema of a function output follows expectation.

        Args:
            function (Callable[..., T]): The function to be decorated.

        Returns:
            Callable[..., T]: The decorated function.
        """

        @wraps(function)
        def wrapper(*args: str, **kwargs: str) -> Any:
            return f.from_json(f.to_json(function(*args, **kwargs)), expected_schema)

        return wrapper

    return decorator


def rename_all_columns(df: DataFrame, prefix: str) -> DataFrame:
    """Given a prefix, rename all columns of a DataFrame.

    Args:
        df (DataFrame): The DataFrame to be processed.
        prefix (str): The prefix to be added to the column names.

    Returns:
        DataFrame: The DataFrame with all columns renamed.

    Examples:
        >>> data = [('a', 1.2, True),('b', 0.0, False),('c', None, None),]
        >>> prefix = 'prefix_'
        >>> rename_all_columns(spark.createDataFrame(data, ['col1', 'col2', 'col3']), prefix).show()
        +-----------+-----------+-----------+
        |prefix_col1|prefix_col2|prefix_col3|
        +-----------+-----------+-----------+
        |          a|        1.2|       true|
        |          b|        0.0|      false|
        |          c|       null|       null|
        +-----------+-----------+-----------+
        <BLANKLINE>
    """
    return reduce(
        lambda df, col: df.withColumnRenamed(col, f"{prefix}{col}"),
        df.columns,
        df,
    )


def safe_array_union(
    a: Column, b: Column, fields_order: list[str] | None = None
) -> Column:
    """Merge the content of two optional columns.

    The function assumes the array columns have the same schema.
    If the `fields_order` is passed, the function assumes that it deals with array of structs and sorts the nested
    struct fields by the provided `fields_order` before conducting array_merge.
    If the `fields_order` is not passed and both columns are <array<struct<...>> type then function assumes struct fields have the same order,
    otherwise the function will raise an AnalysisException.

    Args:
        a (Column): One optional array column.
        b (Column): The other optional array column.
        fields_order (list[str] | None): The order of the fields in the struct. Defaults to None.

    Returns:
        Column: array column with merged content.

    Examples:
        >>> data = [(['a'], ['b']), (['c'], None), (None, ['d']), (None, None)]
        >>> (
        ...    spark.createDataFrame(data, ['col1', 'col2'])
        ...    .select(
        ...        safe_array_union(f.col('col1'), f.col('col2')).alias('merged')
        ...    )
        ...    .show()
        ... )
        +------+
        |merged|
        +------+
        |[a, b]|
        |   [c]|
        |   [d]|
        |  null|
        +------+
        <BLANKLINE>
        >>> schema="arr2: array<struct<b:int,a:string>>, arr: array<struct<a:string,b:int>>"
        >>> data = [([(1,"a",), (2, "c")],[("a", 1,)]),]
        >>> df = spark.createDataFrame(data=data, schema=schema)
        >>> df.select(safe_array_union(f.col("arr"), f.col("arr2"), fields_order=["a", "b"]).alias("merged")).show()
        +----------------+
        |          merged|
        +----------------+
        |[{a, 1}, {c, 2}]|
        +----------------+
        <BLANKLINE>
        >>> schema="arr2: array<struct<b:int,a:string>>, arr: array<struct<a:string,b:int>>"
        >>> data = [([(1,"a",), (2, "c")],[("a", 1,)]),]
        >>> df = spark.createDataFrame(data=data, schema=schema)
        >>> df.select(safe_array_union(f.col("arr"), f.col("arr2")).alias("merged")).show() # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        pyspark.sql.utils.AnalysisException: ...
    """
    if fields_order:
        # sort the nested struct fields by the provided order
        a = sort_array_struct_by_columns(a, fields_order)
        b = sort_array_struct_by_columns(b, fields_order)
    return f.when(a.isNotNull() & b.isNotNull(), f.array_union(a, b)).otherwise(
        f.coalesce(a, b)
    )


def sort_array_struct_by_columns(column: Column, fields_order: list[str]) -> Column:
    """Sort nested struct fields by provided fields order.

    Args:
        column (Column): Column with array of structs.
        fields_order (list[str]): List of field names to sort by.

    Returns:
        Column: Sorted column.

    Examples:
        >>> schema="arr: array<struct<b:int,a:string>>"
        >>> data = [([(1,"a",), (2, "c")],)]
        >>> fields_order = ["a", "b"]
        >>> df = spark.createDataFrame(data=data, schema=schema)
        >>> df.select(sort_array_struct_by_columns(f.col("arr"), fields_order).alias("sorted")).show()
        +----------------+
        |          sorted|
        +----------------+
        |[{c, 2}, {a, 1}]|
        +----------------+
        <BLANKLINE>
    """
    column_name = extract_column_name(column)
    fields_order_expr = ", ".join([f"x.{field}" for field in fields_order])
    return f.expr(
        f"sort_array(transform({column_name}, x -> struct({fields_order_expr})), False)"
    ).alias(column_name)


def extract_column_name(column: Column) -> str:
    """Extract column name from a column expression.

    Args:
        column (Column): Column expression.

    Returns:
        str: Column name.

    Raises:
        ValueError: If the column name cannot be extracted.

    Examples:
        >>> extract_column_name(f.col('col1'))
        'col1'
        >>> extract_column_name(f.sort_array(f.col('col1')))
        'sort_array(col1, true)'
    """
    pattern = re.compile("^Column<'(?P<name>.*)'>?")

    _match = pattern.search(str(column))
    if not _match:
        raise ValueError(f"Cannot extract column name from {column}")
    return _match.group("name")


def create_empty_column_if_not_exists(
    col_name: str, col_schema: t.DataType = t.NullType()
) -> Column:
    """Create a column if it does not exist in the DataFrame.

    Args:
        col_name (str): The name of the column to be created.
        col_schema (t.DataType): The schema of the column to be created. Defaults to NullType.

    Returns:
        Column: The expression to create the column.

    Examples:
        >>> df = spark.createDataFrame([(1, 2),], ['col1', 'col2'])
        >>> df.select("*", create_empty_column_if_not_exists('col3', t.IntegerType())).show()
        +----+----+----+
        |col1|col2|col3|
        +----+----+----+
        |   1|   2|null|
        +----+----+----+
        <BLANKLINE>
    """
    return f.lit(None).cast(col_schema).alias(col_name)


def get_standard_error_from_confidence_interval(lower: Column, upper: Column) -> Column:
    """Compute the standard error from the confidence interval.

    Args:
        lower (Column): The lower bound of the confidence interval.
        upper (Column): The upper bound of the confidence interval.

    Returns:
        Column: The standard error.

    Examples:
        >>> data = [(0.5, 1.5), (None, 2.5), (None, None)]
        >>> (
        ...    spark.createDataFrame(data, ['lower', 'upper'])
        ...    .select(
        ...        get_standard_error_from_confidence_interval(f.col('lower'), f.col('upper')).alias('standard_error')
        ...    )
        ...    .show()
        ... )
        +-------------------+
        |     standard_error|
        +-------------------+
        |0.25510204081632654|
        |               null|
        |               null|
        +-------------------+
        <BLANKLINE>
    """
    return (upper - lower) / (2 * 1.96)


def get_nested_struct_schema(dtype: t.DataType) -> t.StructType:
    """Get the bottom StructType from a nested ArrayType type.

    Args:
        dtype (t.DataType): The nested data structure.

    Returns:
        t.StructType: The nested struct schema.

    Raises:
        TypeError: If the input data type is not a nested struct.

    Examples:
        >>> get_nested_struct_schema(t.ArrayType(t.StructType([t.StructField('a', t.StringType())])))
        StructType([StructField('a', StringType(), True)])

        >>> get_nested_struct_schema(t.ArrayType(t.ArrayType(t.StructType([t.StructField("a", t.StringType())]))))
        StructType([StructField('a', StringType(), True)])
    """
    if isinstance(dtype, t.StructField):
        dtype = dtype.dataType

    match dtype:
        case t.StructType(fields=_):
            return dtype
        case t.ArrayType(elementType=dtype):
            return get_nested_struct_schema(dtype)
        case _:
            raise TypeError("The input data type must be a nested struct.")


def get_struct_field_schema(schema: t.StructType, name: str) -> t.DataType:
    """Get schema for underlying struct field.

    Args:
        schema (t.StructType): Provided schema where the name should be looked in.
        name (str): Name of the field to look in the schema

    Returns:
        t.DataType: Data type of the StructField with provided name

    Raises:
        ValueError: If provided name is not present in the input schema

    Examples:
        >>> get_struct_field_schema(t.StructType([t.StructField("a", t.StringType())]), "a")
        StringType()

        >>> get_struct_field_schema(t.StructType([t.StructField("a", t.StringType())]), "b") # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        ValueError: Provided name b is not present in the schema

    """
    matching_fields = [f for f in schema.fields if f.name == name]
    if not matching_fields:
        raise ValueError("Provided name %s is not present in the schema.", name)
    return matching_fields[0].dataType


def calculate_harmonic_sum(input_array: Column) -> Column:
    """Calculate the harmonic sum of an array.

    Args:
        input_array (Column): input array of doubles

    Returns:
        Column: column of harmonic sums

    Examples:
        >>> from pyspark.sql import Row
        >>> df = spark.createDataFrame([
        ...     Row([0.3, 0.8, 1.0]),
        ...     Row([0.7, 0.2, 0.9]),
        ...     ], ["input_array"]
        ... )
        >>> df.select("*", calculate_harmonic_sum(f.col("input_array")).alias("harmonic_sum")).show()
        +---------------+------------------+
        |    input_array|      harmonic_sum|
        +---------------+------------------+
        |[0.3, 0.8, 1.0]|0.7502326177269538|
        |[0.7, 0.2, 0.9]|0.6674366756805108|
        +---------------+------------------+
        <BLANKLINE>
    """
    return f.aggregate(
        f.arrays_zip(
            f.sort_array(input_array, False).alias("score"),
            f.sequence(f.lit(1), f.size(input_array)).alias("pos"),
        ),
        f.lit(0.0),
        lambda acc, x: acc
        + x["score"]
        / f.pow(x["pos"], 2)
        / f.lit(sum(1 / ((i + 1) ** 2) for i in range(1000))),
    )


def convert_map_type_to_columns(df: DataFrame, map_column: Column) -> list[Column]:
    """Convert a MapType column into multiple columns, one for each key in the map.

    Args:
        df (DataFrame): A Spark DataFrame
        map_column (Column): A Spark Column of MapType

    Returns:
        list[Column]: List of columns, one for each key in the map

    Examples:
        >>> df = spark.createDataFrame([({'a': 1, 'b': 2},), ({'c':3},)], ["map_col"])
        >>> df.select(*convert_map_type_to_columns(df, f.col("map_col"))).show()
        +----+----+----+
        |   a|   b|   c|
        +----+----+----+
        |   1|   2|null|
        |null|null|   3|
        +----+----+----+
        <BLANKLINE>
    """
    # Schema is agnostic of the map keys, I have to collect them first
    keys = (
        df.select(f.explode(map_column))
        .select("key")
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    return [map_column.getItem(k).alias(k) for k in keys]
