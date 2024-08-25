import fire
import polars as pl


def generate_captions_parquet(input_tsv, output_parquet, first_utterance_only=True):
    df = (
        pl.scan_csv(input_tsv, separator="\t")
        .filter(pl.col("gender").is_not_null())
        .with_columns(
            pl.when(pl.col("gender").str.contains("f"))
            .then(pl.lit("female"))
            .otherwise(pl.lit("male"))
            .alias("gender")
        )
        .select(
            pl.col("client_id"),
            pl.col("path").str.extract(r"_(\d+)\.mp3$", 1).cast(pl.Int64).alias("id"),
            pl.col("path").alias("filename"),
            pl.concat_str(
                [
                    pl.lit("A "),
                    pl.col("gender"),
                    pl.lit(" speaker saying: "),
                    pl.col("sentence"),
                ]
            ).alias("caption"),
        )
        .group_by("client_id")
        .agg(pl.all().first())
    )
    df.collect().write_parquet(output_parquet)


def main(meta_path, output_path="captions.parquet", first_utterance_only=True):
    generate_captions_parquet(meta_path, output_path, first_utterance_only)


if __name__ == "__main__":
    fire.Fire(main)
