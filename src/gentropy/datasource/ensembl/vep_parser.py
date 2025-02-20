"""Generating variant index based on Esembl's Variant Effect Predictor output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t

from gentropy.common.schemas import parse_spark_schema
from gentropy.common.spark_helpers import (
    enforce_schema,
    get_nested_struct_schema,
    map_column_by_dictionary,
    order_array_of_structs_by_field,
    order_array_of_structs_by_two_fields,
)
from gentropy.dataset.variant_index import VariantEffectNormaliser, VariantIndex

if TYPE_CHECKING:
    from pyspark.sql import Column, DataFrame

from gentropy.config import VariantIndexConfig


class VariantEffectPredictorParser:
    """Collection of methods to parse VEP output in json format."""

    # NOTE: Due to the fact that the comparison of the xrefs is done om the base of rsids
    # if the field `colocalised_variants` have multiple rsids, this extracting xrefs will result in
    # an array of xref structs, rather then the struct itself.

    DBXREF_SCHEMA = VariantIndex.get_schema()["dbXrefs"].dataType

    # Schema description of the variant effect object:
    VARIANT_EFFECT_SCHEMA = get_nested_struct_schema(
        VariantIndex.get_schema()["variantEffect"]
    )

    # Schema for the allele frequency column:
    ALLELE_FREQUENCY_SCHEMA = VariantIndex.get_schema()["alleleFrequencies"].dataType

    # Consequence to sequence ontology map:
    SEQUENCE_ONTOLOGY_MAP = {
        item["label"]: item["id"]
        for item in VariantIndexConfig.consequence_to_pathogenicity_score
    }

    # Sequence ontology to score map:
    LABEL_TO_SCORE_MAP = {
        item["label"]: item["score"]
        for item in VariantIndexConfig.consequence_to_pathogenicity_score
    }

    @staticmethod
    def get_schema() -> t.StructType:
        """Return the schema of the VEP output.

        Returns:
            t.StructType: VEP output schema.

        Examples:
            >>> type(VariantEffectPredictorParser.get_schema())
            <class 'pyspark.sql.types.StructType'>
        """
        return parse_spark_schema("vep_json_output.json")

    @classmethod
    def extract_variant_index_from_vep(
        cls: type[VariantEffectPredictorParser],
        spark: SparkSession,
        vep_output_path: str | list[str],
        hash_threshold: int,
        **kwargs: bool | float | int | str | None,
    ) -> VariantIndex:
        """Extract variant index from VEP output.

        Args:
            spark (SparkSession): Spark session.
            vep_output_path (str | list[str]): Path to the VEP output.
            hash_threshold (int): Threshold above which variant identifiers will be hashed.
            **kwargs (bool | float | int | str | None): Additional arguments to pass to spark.read.json.

        Returns:
            VariantIndex: Variant index dataset.

        Raises:
            ValueError: Failed reading file or if the dataset is empty.
        """
        # To speed things up and simplify the json structure, read data following an expected schema:
        vep_schema = cls.get_schema()

        try:
            vep_data = spark.read.json(vep_output_path, schema=vep_schema, **kwargs)
        except ValueError as e:
            raise ValueError(f"Failed reading file: {vep_output_path}.") from e

        if vep_data.isEmpty():
            raise ValueError(f"The dataset is empty: {vep_output_path}")

        # Convert to VariantAnnotation dataset:
        return VariantIndex(
            _df=VariantEffectPredictorParser.process_vep_output(
                vep_data, hash_threshold
            ),
            _schema=VariantIndex.get_schema(),
            id_threshold=hash_threshold,
        )

    @staticmethod
    def _extract_ensembl_xrefs(colocated_variants: Column) -> Column:
        """Extract rs identifiers and build cross reference to Ensembl's variant page.

        Args:
            colocated_variants (Column): Colocated variants field from VEP output.

        Returns:
            Column: List of dbXrefs for rs identifiers.
        """
        return VariantEffectPredictorParser._generate_dbxrefs(
            VariantEffectPredictorParser._colocated_variants_to_rsids(
                colocated_variants
            ),
            "ensembl_variation",
        )

    @enforce_schema(DBXREF_SCHEMA)
    @staticmethod
    def _generate_dbxrefs(ids: Column, source: str) -> Column:
        """Convert a list of variant identifiers to dbXrefs given the source label.

        Identifiers are cast to strings, then Null values are filtered out of the id list.

        Args:
            ids (Column): List of variant identifiers.
            source (str): Source label for the dbXrefs.

        Returns:
            Column: List of dbXrefs.

        Examples:
            >>> (
            ...     spark.createDataFrame([('rs12',),(None,)], ['test_id'])
            ...     .select(VariantEffectPredictorParser._generate_dbxrefs(f.array(f.col('test_id')), "ensemblVariation").alias('col'))
            ...     .show(truncate=False)
            ... )
            +--------------------------+
            |col                       |
            +--------------------------+
            |[{rs12, ensemblVariation}]|
            |[]                        |
            +--------------------------+
            <BLANKLINE>
            >>> (
            ...     spark.createDataFrame([('rs12',),(None,)], ['test_id'])
            ...     .select(VariantEffectPredictorParser._generate_dbxrefs(f.array(f.col('test_id')), "ensemblVariation").alias('col'))
            ...     .first().col[0].asDict()
            ... )
            {'id': 'rs12', 'source': 'ensemblVariation'}
        """
        ids = f.filter(ids, lambda id: id.isNotNull())
        xref_column = f.transform(
            ids,
            lambda id: f.struct(
                id.cast(t.StringType()).alias("id"), f.lit(source).alias("source")
            ),
        )

        return f.when(xref_column.isNull(), f.array()).otherwise(xref_column)

    @staticmethod
    def _colocated_variants_to_rsids(colocated_variants: Column) -> Column:
        """Extract rs identifiers from the colocated variants VEP field.

        Args:
            colocated_variants (Column): Colocated variants field from VEP output.

        Returns:
            Column: List of rs identifiers.

        Examples:
            >>> data = [('s1', 'rs1'),('s1', 'rs2'),('s1', 'rs3'),('s2', None),]
            >>> (
            ...    spark.createDataFrame(data, ['s','v'])
            ...    .groupBy('s')
            ...    .agg(f.collect_list(f.struct(f.col('v').alias('id'))).alias('cv'))
            ...    .select(VariantEffectPredictorParser._colocated_variants_to_rsids(f.col('cv')).alias('rsIds'),)
            ...    .show(truncate=False)
            ... )
            +---------------+
            |rsIds          |
            +---------------+
            |[rs1, rs2, rs3]|
            |[NULL]         |
            +---------------+
            <BLANKLINE>
        """
        return f.when(
            colocated_variants.isNotNull(),
            f.transform(
                colocated_variants, lambda variant: variant.getItem("id").alias("id")
            ),
        ).alias("rsIds")

    @staticmethod
    def _extract_omim_xrefs(colocated_variants: Column) -> Column:
        """Build xdbRefs for OMIM identifiers.

        OMIM identifiers are extracted from the colocated variants field, casted to strings and formatted as dbXrefs.

        Args:
            colocated_variants (Column): Colocated variants field from VEP output.

        Returns:
            Column: List of dbXrefs for OMIM identifiers.

        Examples:
            >>> data = [('234234.32', 'rs1', 's1',),(None, 'rs1', 's1',),]
            >>> (
            ...    spark.createDataFrame(data, ['id', 'rsid', 's'])
            ...    .groupBy('s')
            ...    .agg(f.collect_list(f.struct(f.struct(f.array(f.col('id')).alias('OMIM')).alias('var_synonyms'),f.col('rsid').alias('id'),),).alias('cv'),).select(VariantEffectPredictorParser._extract_omim_xrefs(f.col('cv')).alias('dbXrefs'))
            ...    .show(truncate=False)
            ... )
            +-------------------+
            |dbXrefs            |
            +-------------------+
            |[{234234#32, omim}]|
            +-------------------+
            <BLANKLINE>
        """
        variants_w_omim_ref = f.filter(
            colocated_variants,
            lambda variant: variant.getItem("var_synonyms").getItem("OMIM").isNotNull(),
        )

        omim_var_ids = f.transform(
            variants_w_omim_ref,
            lambda var: f.transform(
                var.getItem("var_synonyms").getItem("OMIM"),
                lambda var: f.regexp_replace(var.cast(t.StringType()), r"\.", r"#"),
            ),
        )

        return VariantEffectPredictorParser._generate_dbxrefs(
            f.flatten(omim_var_ids), "omim"
        )

    @staticmethod
    def _extract_clinvar_xrefs(colocated_variants: Column) -> Column:
        """Build xdbRefs for VCV ClinVar identifiers.

        ClinVar identifiers are extracted from the colocated variants field.
        VCV-identifiers are filtered out to generate cross-references.

        Args:
            colocated_variants (Column): Colocated variants field from VEP output.

        Returns:
            Column: List of dbXrefs for VCV ClinVar identifiers.

        Examples:
            >>> data = [('VCV2323,RCV3423', 'rs1', 's1',),(None, 'rs1', 's1',),]
            >>> (
            ...    spark.createDataFrame(data, ['id', 'rsid', 's'])
            ...    .groupBy('s')
            ...    .agg(f.collect_list(f.struct(f.struct(f.split(f.col('id'),',').alias('ClinVar')).alias('var_synonyms'),f.col('rsid').alias('id'),),).alias('cv'),).select(VariantEffectPredictorParser._extract_clinvar_xrefs(f.col('cv')).alias('dbXrefs'))
            ...    .show(truncate=False)
            ... )
            +--------------------+
            |dbXrefs             |
            +--------------------+
            |[{VCV2323, clinvar}]|
            +--------------------+
            <BLANKLINE>
        """
        variants_w_clinvar_ref = f.filter(
            colocated_variants,
            lambda variant: variant.getItem("var_synonyms")
            .getItem("ClinVar")
            .isNotNull(),
        )

        clin_var_ids = f.transform(
            variants_w_clinvar_ref,
            lambda var: f.filter(
                var.getItem("var_synonyms").getItem("ClinVar"),
                lambda x: x.startswith("VCV"),
            ),
        )

        return VariantEffectPredictorParser._generate_dbxrefs(
            f.flatten(clin_var_ids), "clinvar"
        )

    @staticmethod
    def _get_most_severe_transcript(
        transcript_column_name: str, score_field_name: str
    ) -> Column:
        """Return transcript with the highest in silico predicted score.

        This method assumes the higher the score, the more severe the consequence of the variant is.
        Hence, by selecting the transcript with the highest score, we are selecting the most severe consequence.

        Args:
            transcript_column_name (str): Name of the column containing the list of transcripts.
            score_field_name (str): Name of the field containing the severity score.

        Returns:
            Column: Most severe transcript.

        Examples:
            >>> data = [("v1", 0.2, 'transcript1'),("v1", None, 'transcript2'),("v1", 0.6, 'transcript3'),("v1", 0.4, 'transcript4'),]
            >>> (
            ...    spark.createDataFrame(data, ['v', 'score', 'transcriptId'])
            ...    .groupBy('v')
            ...    .agg(f.collect_list(f.struct(f.col('score'),f.col('transcriptId'))).alias('transcripts'))
            ...    .select(VariantEffectPredictorParser._get_most_severe_transcript('transcripts', 'score').alias('most_severe_transcript'))
            ...    .show(truncate=False)
            ... )
            +----------------------+
            |most_severe_transcript|
            +----------------------+
            |{0.6, transcript3}    |
            +----------------------+
            <BLANKLINE>

        Raises:
            AssertionError: When `transcript_column_name` is not a string.
        """
        assert isinstance(
            transcript_column_name, str
        ), "transcript_column_name must be a string and not a column."
        # Order transcripts by severity score:
        ordered_transcripts = order_array_of_structs_by_field(
            transcript_column_name, score_field_name
        )

        # Drop transcripts with no severity score and return the most severe one:
        return f.filter(
            ordered_transcripts,
            lambda transcript: transcript.getItem(score_field_name).isNotNull(),
        )[0]

    @classmethod
    @enforce_schema(VARIANT_EFFECT_SCHEMA)
    def _get_vep_prediction(cls, most_severe_consequence: Column) -> Column:
        return f.struct(
            f.lit("VEP").alias("method"),
            most_severe_consequence.alias("assessment"),
            map_column_by_dictionary(
                most_severe_consequence, cls.LABEL_TO_SCORE_MAP
            ).alias("score"),
        )

    @staticmethod
    @enforce_schema(VARIANT_EFFECT_SCHEMA)
    def _get_max_alpha_missense(transcripts: Column) -> Column:
        """Return the most severe alpha missense prediction from all transcripts.

        This function assumes one variant can fall onto only one gene with alpha-sense prediction available on the canonical transcript.

        Args:
            transcripts (Column): List of transcripts.

        Returns:
            Column: Most severe alpha missense prediction.

        Examples:
        >>> data = [('v1', 0.4, 'assessment 1'), ('v1', None, None), ('v1', None, None),('v2', None, None),]
        >>> (
        ...     spark.createDataFrame(data, ['v', 'a', 'b'])
        ...    .groupBy('v')
        ...    .agg(f.collect_list(f.struct(f.struct(
        ...        f.col('a').alias('am_pathogenicity'),
        ...        f.col('b').alias('am_class')).alias('alphamissense'),
        ...        f.lit('gene1').alias('gene_id'))).alias('transcripts')
        ...    )
        ...    .select(VariantEffectPredictorParser._get_max_alpha_missense(f.col('transcripts')).alias('am'))
        ...    .show(truncate=False)
        ... )
        +-----------------------------------------------------+
        |am                                                   |
        +-----------------------------------------------------+
        |{AlphaMissense, assessment 1, 0.4, NULL, gene1, NULL}|
        |{AlphaMissense, NULL, NULL, NULL, gene1, NULL}       |
        +-----------------------------------------------------+
        <BLANKLINE>
        """
        # Extracting transcript with alpha missense values:
        transcript = f.filter(
            transcripts,
            lambda transcript: transcript.getItem("alphamissense").isNotNull(),
        )[0]

        return f.when(
            transcript.isNotNull(),
            f.struct(
                # Adding method:
                f.lit("AlphaMissense").alias("method"),
                # Extracting assessment:
                transcript.alphamissense.am_class.alias("assessment"),
                # Extracting score:
                transcript.alphamissense.am_pathogenicity.cast(t.FloatType()).alias(
                    "score"
                ),
                # Adding assessment flag:
                f.lit(None).cast(t.StringType()).alias("assessmentFlag"),
                # Extracting target id:
                transcript.gene_id.alias("targetId"),
            ),
        )

    @classmethod
    @enforce_schema(VARIANT_EFFECT_SCHEMA)
    def _vep_variant_effect_extractor(
        cls: type[VariantEffectPredictorParser],
        transcript_column_name: str,
        method_name: str,
        score_column_name: str | None = None,
        assessment_column_name: str | None = None,
        assessment_flag_column_name: str | None = None,
    ) -> Column:
        """Extract variant effect from VEP output.

        Args:
            transcript_column_name (str): Name of the column containing the list of transcripts.
            method_name (str): Name of the variant effect.
            score_column_name (str | None): Name of the column containing the score.
            assessment_column_name (str | None): Name of the column containing the assessment.
            assessment_flag_column_name (str | None): Name of the column containing the assessment flag.

        Returns:
            Column: Variant effect.
        """
        # Get transcript with the highest score:
        most_severe_transcript: Column = (
            # Getting the most severe transcript:
            VariantEffectPredictorParser._get_most_severe_transcript(
                transcript_column_name, score_column_name
            )
            if score_column_name is not None
            # If we don't have score, just pick one of the transcript where assessment is available:
            else f.filter(
                f.col(transcript_column_name),
                lambda transcript: transcript.getItem(
                    assessment_column_name
                ).isNotNull(),
            )
        )

        # Get assessment:
        assessment = (
            f.lit(None).cast(t.StringType()).alias("assessment")
            if assessment_column_name is None
            else most_severe_transcript.getField(assessment_column_name).alias(
                "assessment"
            )
        )

        # Get score:
        score = (
            f.lit(None).cast(t.FloatType()).alias("score")
            if score_column_name is None
            else most_severe_transcript.getField(score_column_name)
            .cast(t.FloatType())
            .alias("score")
        )

        # Get assessment flag:
        assessment_flag = (
            f.lit(None).cast(t.StringType()).alias("assessmentFlag")
            if assessment_flag_column_name is None
            else most_severe_transcript.getField(assessment_flag_column_name)
            .cast(t.StringType())
            .alias("assessmentFlag")
        )

        # Extract gene id:
        gene_id = most_severe_transcript.getItem("gene_id").alias("targetId")

        return f.when(
            most_severe_transcript.isNotNull(),
            f.struct(
                f.lit(method_name).alias("method"),
                assessment,
                score,
                assessment_flag,
                gene_id,
            ),
        )

    @staticmethod
    def _parser_amino_acid_change(amino_acids: Column, protein_end: Column) -> Column:
        """Convert VEP amino acid change information to one letter code aa substitution code.

        The logic assumes that the amino acid change is given in the format "from/to" and the protein end is given also.
        If any of the information is missing, the amino acid change will be set to None.

        Args:
            amino_acids (Column): Amino acid change information.
            protein_end (Column): Protein end information.

        Returns:
            Column: Amino acid change in one letter code.

        Examples:
            >>> data = [('A/B', 1),('A/B', None),(None, 1),]
            >>> (
            ...    spark.createDataFrame(data, ['amino_acids', 'protein_end'])
            ...    .select(VariantEffectPredictorParser._parser_amino_acid_change(f.col('amino_acids'), f.col('protein_end')).alias('amino_acid_change'))
            ...    .show()
            ... )
            +-----------------+
            |amino_acid_change|
            +-----------------+
            |              A1B|
            |             NULL|
            |             NULL|
            +-----------------+
            <BLANKLINE>
        """
        return f.when(
            amino_acids.isNotNull() & protein_end.isNotNull(),
            f.concat(
                f.split(amino_acids, r"\/")[0],
                protein_end,
                f.split(amino_acids, r"\/")[1],
            ),
        ).otherwise(f.lit(None))

    @staticmethod
    def _collect_uniprot_accessions(trembl: Column, swissprot: Column) -> Column:
        """Flatten arrays containing Uniprot accessions.

        Args:
            trembl (Column): List of TrEMBL protein accessions.
            swissprot (Column): List of SwissProt protein accessions.

        Returns:
            Column: List of unique Uniprot accessions extracted from swissprot and trembl arrays, splitted from version numbers.

        Examples:
        >>> data = [('v1', ["sp_1"], ["tr_1"]), ('v1', None, None), ('v1', None, ["tr_2"]),]
        >>> (
        ...    spark.createDataFrame(data, ['v', 'sp', 'tr'])
        ...    .select(VariantEffectPredictorParser._collect_uniprot_accessions(f.col('sp'), f.col('tr')).alias('proteinIds'))
        ...    .show()
        ... )
        +------------+
        |  proteinIds|
        +------------+
        |[sp_1, tr_1]|
        |          []|
        |      [tr_2]|
        +------------+
        <BLANKLINE>
        """
        # Dropping Null values and flattening the arrays:
        return f.filter(
            f.array_distinct(
                f.transform(
                    f.flatten(
                        f.filter(
                            f.array(trembl, swissprot),
                            lambda x: x.isNotNull(),
                        )
                    ),
                    lambda x: f.split(x, r"\.")[0],
                )
            ),
            lambda x: x.isNotNull(),
        )

    @staticmethod
    def _parse_variant_location_id(vep_input_field: Column) -> list[Column]:
        r"""Parse variant identifier, chromosome, position, reference allele and alternate allele from VEP input field.

        Args:
            vep_input_field (Column): Column containing variant vcf string used as VEP input.

        Returns:
            list[Column]: List of columns containing chromosome, position, reference allele and alternate allele.
        """
        variant_fields = f.split(vep_input_field, r"\t")
        return [
            f.concat_ws(
                "_",
                f.array(
                    variant_fields[0],
                    variant_fields[1],
                    variant_fields[3],
                    variant_fields[4],
                ),
            ).alias("variantId"),
            variant_fields[0].cast(t.StringType()).alias("chromosome"),
            variant_fields[1].cast(t.IntegerType()).alias("position"),
            variant_fields[3].cast(t.StringType()).alias("referenceAllele"),
            variant_fields[4].cast(t.StringType()).alias("alternateAllele"),
        ]

    @classmethod
    def process_vep_output(
        cls, vep_output: DataFrame, hash_threshold: int = 100
    ) -> DataFrame:
        """Process and format a VEP output in JSON format.

        Args:
            vep_output (DataFrame): raw VEP output, read as spark DataFrame.
            hash_threshold (int): threshold above which variant identifiers will be hashed.

        Returns:
           DataFrame: processed data in the right shape.
        """
        # Processing VEP output:
        return (
            vep_output
            # Dropping non-canonical transcript consequences:  # TODO: parametrize this.
            .withColumn(
                "transcript_consequences",
                f.filter(
                    f.col("transcript_consequences"),
                    lambda consequence: consequence.getItem("canonical") == 1,
                ),
            )
            .select(
                # Parse id and chr:pos:alt:ref:
                *cls._parse_variant_location_id(f.col("input")),
                # Extracting corss_references from colocated variants:
                cls._extract_ensembl_xrefs(f.col("colocated_variants")).alias(
                    "ensembl_xrefs"
                ),
                cls._extract_omim_xrefs(f.col("colocated_variants")).alias(
                    "omim_xrefs"
                ),
                cls._extract_clinvar_xrefs(f.col("colocated_variants")).alias(
                    "clinvar_xrefs"
                ),
                # Extracting variant effect assessments
                f.when(
                    # The following variant effect assessments are only available for variants with transcript consequences:
                    f.col("transcript_consequences").isNotNull(),
                    f.filter(
                        f.array(
                            # Extract CADD scores:
                            cls._vep_variant_effect_extractor(
                                transcript_column_name="transcript_consequences",
                                method_name="CADD",
                                score_column_name="cadd_phred",
                            ),
                            # Extract polyphen scores:
                            cls._vep_variant_effect_extractor(
                                transcript_column_name="transcript_consequences",
                                method_name="PolyPhen",
                                score_column_name="polyphen_score",
                                assessment_column_name="polyphen_prediction",
                            ),
                            # Extract sift scores:
                            cls._vep_variant_effect_extractor(
                                transcript_column_name="transcript_consequences",
                                method_name="SIFT",
                                score_column_name="sift_score",
                                assessment_column_name="sift_prediction",
                            ),
                            # Extract loftee scores:
                            cls._vep_variant_effect_extractor(
                                method_name="LOFTEE",
                                transcript_column_name="transcript_consequences",
                                score_column_name="lof",
                                assessment_column_name="lof",
                                assessment_flag_column_name="lof_filter",
                            ),
                            # Extract GERP conservation score:
                            cls._vep_variant_effect_extractor(
                                method_name="GERP",
                                transcript_column_name="transcript_consequences",
                                score_column_name="conservation",
                            ),
                            # Extract max alpha missense:
                            cls._get_max_alpha_missense(
                                f.col("transcript_consequences")
                            ),
                            # Extract VEP prediction:
                            cls._get_vep_prediction(f.col("most_severe_consequence")),
                        ),
                        lambda predictor: predictor.isNotNull(),
                    ),
                )
                .otherwise(
                    # Extract CADD scores from intergenic object:
                    f.array(
                        cls._vep_variant_effect_extractor(
                            transcript_column_name="intergenic_consequences",
                            method_name="CADD",
                            score_column_name="cadd_phred",
                        ),
                        # Extract GERP conservation score:
                        cls._vep_variant_effect_extractor(
                            method_name="GERP",
                            transcript_column_name="intergenic_consequences",
                            score_column_name="conservation",
                        ),
                        # Extract VEP prediction:
                        cls._get_vep_prediction(f.col("most_severe_consequence")),
                    )
                )
                .alias("variantEffect"),
                # Convert consequence to SO:
                map_column_by_dictionary(
                    f.col("most_severe_consequence"), cls.SEQUENCE_ONTOLOGY_MAP
                ).alias("mostSevereConsequenceId"),
                # Propagate most severe consequence:
                "most_severe_consequence",
                # Extract HGVS identifier:
                f.when(
                    f.size("transcript_consequences") > 0,
                    f.col("transcript_consequences").getItem(0).getItem("hgvsg"),
                )
                .when(
                    f.size("intergenic_consequences") > 0,
                    f.col("intergenic_consequences").getItem(0).getItem("hgvsg"),
                )
                .otherwise(f.lit(None))
                .alias("hgvsId"),
                # Collect transcript consequence:
                f.when(
                    f.col("transcript_consequences").isNotNull(),
                    f.transform(
                        f.col("transcript_consequences"),
                        lambda transcript: f.struct(
                            # Convert consequence terms to SO identifier:
                            f.transform(
                                transcript.consequence_terms,
                                lambda y: map_column_by_dictionary(
                                    y, cls.SEQUENCE_ONTOLOGY_MAP
                                ),
                            ).alias("variantFunctionalConsequenceIds"),
                            # Convert consequence terms to consequence score:
                            f.array_max(
                                f.transform(
                                    transcript.consequence_terms,
                                    lambda term: map_column_by_dictionary(
                                        term, cls.LABEL_TO_SCORE_MAP
                                    ),
                                )
                            )
                            .cast(t.FloatType())
                            .alias("consequenceScore"),
                            # Format amino acid change:
                            cls._parser_amino_acid_change(
                                transcript.amino_acids, transcript.protein_end
                            ).alias("aminoAcidChange"),
                            # Extract and clean uniprot identifiers:
                            cls._collect_uniprot_accessions(
                                transcript.swissprot,
                                transcript.trembl,
                            ).alias("uniprotAccessions"),
                            # Add canonical flag:
                            f.when(transcript.canonical == 1, f.lit(True))
                            .otherwise(f.lit(False))
                            .alias("isEnsemblCanonical"),
                            # Extract footprint distance:
                            transcript.codons.alias("codons"),
                            f.when(transcript.distance.isNotNull(), transcript.distance)
                            .otherwise(f.lit(0))
                            .cast(t.LongType())
                            .alias("distanceFromFootprint"),
                            # Extract distance from the transcription start site:
                            transcript.tssdistance.cast(t.LongType()).alias(
                                "distanceFromTss"
                            ),
                            # Extracting APPRIS isoform annotation for this transcript:
                            transcript.appris.alias("appris"),
                            # Extracting MANE select transcript:
                            transcript.mane_select.alias("maneSelect"),
                            transcript.gene_id.alias("targetId"),
                            transcript.impact.alias("impact"),
                            transcript.lof.cast(t.StringType()).alias(
                                "lofteePrediction"
                            ),
                            transcript.lof.cast(t.FloatType()).alias("siftPrediction"),
                            transcript.lof.cast(t.FloatType()).alias(
                                "polyphenPrediction"
                            ),
                            transcript.transcript_id.alias("transcriptId"),
                            transcript.biotype.alias("biotype"),
                            transcript.gene_symbol.alias("approvedSymbol"),
                        ),
                    ),
                ).alias("transcriptConsequences"),
                # Extracting rsids:
                cls._colocated_variants_to_rsids(f.col("colocated_variants")).alias(
                    "rsIds"
                ),
                # Adding empty array for allele frequencies - now this piece of data is not coming form the VEP data:
                f.array().cast(cls.ALLELE_FREQUENCY_SCHEMA).alias("alleleFrequencies"),
            )
            # Dropping transcripts where the consequence score or the distance is null:
            .withColumn(
                "transcriptConsequences",
                f.filter(
                    f.col("transcriptConsequences"),
                    lambda x: x.getItem("consequenceScore").isNotNull()
                    & x.getItem("distanceFromFootprint").isNotNull(),
                ),
            )
            # Sort transcript consequences by consequence score and distance from footprint and add index:
            .withColumn(
                "transcriptConsequences",
                f.when(
                    f.col("transcriptConsequences").isNotNull(),
                    f.transform(
                        order_array_of_structs_by_two_fields(
                            "transcriptConsequences",
                            "consequenceScore",
                            "distanceFromFootprint",
                        ),
                        lambda x, i: x.withField("transcriptIndex", i + f.lit(1)),
                    ),
                ),
            )
            # Adding protvar xref for missense variants:  # TODO: making and extendable list of consequences
            .withColumn(
                "protvar_xrefs",
                f.when(
                    f.size(
                        f.filter(
                            f.col("transcriptConsequences"),
                            lambda x: f.array_contains(
                                x.variantFunctionalConsequenceIds, "SO_0001583"
                            ),
                        )
                    )
                    > 0,
                    cls._generate_dbxrefs(f.array(f.col("variantId")), "protvar"),
                ),
            )
            .withColumn(
                "dbXrefs",
                f.flatten(
                    f.filter(
                        f.array(
                            f.col("ensembl_xrefs"),
                            f.col("omim_xrefs"),
                            f.col("clinvar_xrefs"),
                            f.col("protvar_xrefs"),
                        ),
                        lambda x: x.isNotNull(),
                    )
                ),
            )
            # If the variantId is too long, hash it:
            .withColumn(
                "variantId",
                VariantIndex.hash_long_variant_ids(
                    f.col("variantId"),
                    f.col("chromosome"),
                    f.col("position"),
                    hash_threshold,
                ),
            )
            # Generating a temporary column with only protein coding transcripts:
            .withColumn(
                "proteinCodingTranscripts",
                f.filter(
                    f.col("transcriptConsequences"),
                    lambda x: x.getItem("biotype") == "protein_coding",
                ),
            )
            # Generate variant descrioption:
            .withColumn(
                "variantDescription",
                cls._compose_variant_description(
                    # Passing the most severe consequence:
                    f.col("most_severe_consequence"),
                    # The first transcript:
                    f.filter(
                        f.col("transcriptConsequences"),
                        lambda vep: vep.transcriptIndex == 1,
                    ).getItem(0),
                    # The first protein coding transcript:
                    order_array_of_structs_by_field(
                        "proteinCodingTranscripts", "transcriptIndex"
                    )[f.size("proteinCodingTranscripts") - 1],
                ),
            )
            # Normalising variant effect assessments:
            .withColumn(
                "variantEffect",
                VariantEffectNormaliser.normalise_variant_effect(
                    f.col("variantEffect")
                ),
            )
            # Dropping intermediate xref columns:
            .drop(
                *[
                    "ensembl_xrefs",
                    "omim_xrefs",
                    "clinvar_xrefs",
                    "protvar_xrefs",
                    "most_severe_consequence",
                    "proteinCodingTranscripts",
                ]
            )
            # Drooping rows with null position:
            .filter(f.col("position").isNotNull())
        )

    @classmethod
    def _compose_variant_description(
        cls: type[VariantEffectPredictorParser],
        most_severe_consequence: Column,
        first_transcript: Column,
        first_protein_coding: Column,
    ) -> Column:
        """Compose variant description based on the most severe consequence.

        Args:
            most_severe_consequence (Column): Most severe consequence
            first_transcript (Column): First transcript
            first_protein_coding (Column): First protein coding transcript

        Returns:
            Column: Variant description
        """
        return (
            # When there's no transcript whatsoever:
            f.when(
                first_transcript.isNull(),
                f.lit("Intergenic variant no gene in window"),
            )
            # When the biotype of the first gene is protein coding:
            .when(
                first_transcript.getItem("biotype") == "protein_coding",
                cls._process_protein_coding_transcript(
                    first_transcript, most_severe_consequence
                ),
            )
            # When the first gene is not protein coding, we also pass the first protein coding gene:
            .otherwise(
                cls._process_non_protein_coding_transcript(
                    most_severe_consequence, first_transcript, first_protein_coding
                )
            )
        )

    @staticmethod
    def _process_consequence_term(consequence_term: Column) -> Column:
        """Cleaning up consequence term: capitalizing and replacing underscores.

        Args:
            consequence_term (Column): Consequence term.

        Returns:
            Column: Cleaned up consequence term.
        """
        last = f.when(consequence_term.contains("variant"), f.lit("")).otherwise(
            " variant"
        )
        return f.concat(f.regexp_replace(f.initcap(consequence_term), "_", " "), last)

    @staticmethod
    def _process_overlap(transcript: Column) -> Column:
        """Process overlap with gene: if the variant overlaps with the gene, return the gene name or distance.

        Args:
            transcript (Column): Transcript.

        Returns:
            Column: string column with overlap description.
        """
        gene_label = f.when(
            transcript.getField("approvedSymbol").isNotNull(),
            transcript.getField("approvedSymbol"),
        ).otherwise(transcript.getField("targetId"))

        return f.when(
            transcript.getField("distanceFromFootprint") == 0,
            # "overlapping with CCDC8"
            f.concat(f.lit(" overlapping with "), gene_label),
        ).otherwise(
            # " 123 basepair away from CCDC8"
            f.concat(
                f.lit(" "),
                f.format_number(transcript.getField("distanceFromFootprint"), 0),
                f.lit(" basepair away from "),
                gene_label,
            )
        )

    @staticmethod
    def _process_aa_change(transcript: Column) -> Column:
        """Extract amino acid change information from transcript when available.

        Args:
            transcript (Column): Transcript.

        Returns:
            Column: Amino acid change information.
        """
        return f.when(
            transcript.getField("aminoAcidChange").isNotNull(),
            f.concat(
                f.lit(", causing amio-acid change: "),
                transcript.getField("aminoAcidChange"),
                f.lit(" with "),
                f.lower(transcript.getField("impact")),
                f.lit(" impact."),
            ),
        ).otherwise(f.lit("."))

    @staticmethod
    def _process_lof(transcript: Column) -> Column:
        """Process loss of function annotation from LOFTEE prediction.

        Args:
            transcript (Column): Transcript.

        Returns:
            Column: Loss of function annotation.
        """
        return f.when(
            transcript.getField("lofteePrediction").isNotNull()
            & (transcript.getField("lofteePrediction") == "HC"),
            f.lit(" A high-confidence loss-of-function variant by loftee."),
        ).otherwise(f.lit(""))

    @classmethod
    def _process_protein_coding_transcript(
        cls: type[VariantEffectPredictorParser],
        transcript: Column,
        most_severe_consequence: Column,
    ) -> Column:
        """Extract information from the first, protein coding transcript.

        Args:
            transcript (Column): Transcript.
            most_severe_consequence (Column): Most severe consequence.

        Returns:
            Column: Variant description.
        """
        # Process consequence term:
        consequence_text = cls._process_consequence_term(most_severe_consequence)

        # Does it overlap with the gene:
        overlap = cls._process_overlap(transcript)

        # Does it cause amino acid change:
        amino_acid_change = cls._process_aa_change(transcript)

        # Processing lof annotation:
        lof_assessment = cls._process_lof(transcript)

        # Concat all together:
        return f.concat(consequence_text, overlap, amino_acid_change, lof_assessment)

    @staticmethod
    def _adding_biotype(transcript: Column) -> Column:
        """Adding biotype information to the variant description.

        Args:
            transcript (Column): Transcript.

        Returns:
            Column: Biotype information.
        """
        biotype = f.when(
            transcript.getField("biotype").contains("gene"),
            f.regexp_replace(transcript.getField("biotype"), "_", " "),
        ).otherwise(
            f.concat(
                f.regexp_replace(transcript.getField("biotype"), "_", " "),
                f.lit(" gene."),
            )
        )

        return f.concat(f.lit(", a "), biotype)

    @staticmethod
    def _parse_protein_coding_transcript(transcript: Column) -> Column:
        """Parse the closest, not first protein coding transcript: extract gene symbol and distance.

        Args:
            transcript (Column): Transcript.

        Returns:
            Column: Protein coding transcript information.
        """
        gene_label = f.when(
            transcript.getField("approvedSymbol").isNotNull(),
            transcript.getField("approvedSymbol"),
        ).otherwise(transcript.getField("targetId"))

        return f.when(
            transcript.isNotNull(),
            f.concat(
                f.lit(" The closest protein-coding gene is "),
                gene_label,
                f.lit(" ("),
                f.format_number(transcript.getField("distanceFromFootprint"), 0),
                f.lit(" basepair away)."),
            ),
        ).otherwise(f.lit(""))

    @classmethod
    def _process_non_protein_coding_transcript(
        cls: type[VariantEffectPredictorParser],
        most_severe_consequence: Column,
        first_transcript: Column,
        first_protein_coding: Column,
    ) -> Column:
        """Extract information from the first, non-protein coding transcript.

        Args:
            most_severe_consequence (Column): Most severe consequence.
            first_transcript (Column): First transcript.
            first_protein_coding (Column): First protein coding transcript.

        Returns:
            Column: Variant description.
        """
        # Process consequence term:
        consequence_text = cls._process_consequence_term(most_severe_consequence)

        # Does it overlap with the gene:
        overlap = cls._process_overlap(first_transcript)

        # Adding biotype:
        biotype = cls._adding_biotype(first_transcript)

        # Adding protein coding gene:
        protein_transcript = cls._parse_protein_coding_transcript(first_protein_coding)

        # Concat all together:
        return f.concat(consequence_text, overlap, biotype, protein_transcript)
