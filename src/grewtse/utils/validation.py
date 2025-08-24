import pandas as pd


def load_and_validate_mp_dataset(filepath: str):
    required_columns = {
        "sentence_id",
        "match_id",
        "original_text",
    }

    try:
        df = pd.read_csv(filepath)

        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(missing)}")

        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error parsing CSV file: {e}")
