#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def convert_file(file_type: str, input_path: Path, output_path: Path) -> None:
    swap_first_two = file_type == "response"

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        writer = csv.writer(dst)
        row_count = 0

        for lineno, raw_line in enumerate(src, 1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if swap_first_two:
                if len(parts) < 2:
                    raise ValueError(
                        f"Line {lineno} has fewer than two columns: {line!r}"
                    )
                parts[0], parts[1] = parts[1], parts[0]

            writer.writerow(parts)
            row_count += 1

    print(f"Wrote {row_count} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LabelMe response or gold TSV files to CSV. "
            "Response files swap the first two columns."
        )
    )
    parser.add_argument(
        "file_type",
        choices=["response", "gold"],
        help="Which file to convert.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Path to the source .txt file. Defaults to the dataset file for the given type.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the CSV. Defaults to replacing the input suffix with .csv.",
    )

    args = parser.parse_args()

    default_inputs = {
        "response": Path("datasets/real-world/labelme/LabelMe.response.txt"),
        "gold": Path("datasets/real-world/labelme/LabelMe.gold.txt"),
    }

    input_path = Path(args.input) if args.input else default_inputs[args.file_type]
    output_path = (
        Path(args.output) if args.output else input_path.with_suffix(".csv")
    )

    convert_file(args.file_type, input_path, output_path)


if __name__ == "__main__":
    main()
