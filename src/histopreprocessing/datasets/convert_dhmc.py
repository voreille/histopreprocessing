from pathlib import Path

import click
import pandas as pd
import tifftools
from tqdm import tqdm


@click.command()
@click.option("--csv-path", type=click.Path(exists=True))
@click.option("--input-dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output-dir", type=click.Path(file_okay=False), required=True)
@click.option(
    "--force", is_flag=True, help="Overwrite existing files in the output directory."
)
def update_tiff_tags(csv_path, input_dir, output_dir, force):
    """
    Update TIFF ImageDescription tags using metadata from a CSV file.

    Arguments:
    - CSV_PATH: Path to CSV file with columns 'File Name', 'Magnification', and 'Microns Per Pixel'.
    - INPUT_DIR: Directory containing original .tif files.
    - OUTPUT_DIR: Directory to save the updated .tif files.
    """
    df = pd.read_csv(csv_path).set_index("File Name")
    files = list(Path(input_dir).glob("*.tif"))
    click.echo(f"Found {len(files)} TIFF files in input directory: {input_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for file in tqdm(files, desc="Processing TIFF files"):
        file_name = file.name
        if file_name not in df.index:
            click.echo(f"Warning: {file_name} not found in CSV. Skipping.")
            continue

        appmag = int(df.at[file_name, "Magnification"])
        mpp = float(df.at[file_name, "Microns Per Pixel"])

        image_description = f"Aperio Fake |AppMag = {appmag}|MPP = {mpp}"
        in_path = Path(input_dir) / file_name
        out_path = Path(output_dir) / file_name

        if not in_path.exists():
            click.echo(f"Warning: File not found: {in_path}")
            continue

        setlist = [("ImageDescription", image_description)]
        if out_path.exists():
            if not force:
                click.echo(
                    f"Skipping existing file: {out_path}, use --force to overwrite."
                )
                continue

            out_path.unlink()

        tifftools.tiff_set(str(in_path), str(out_path), setlist=setlist)
        click.echo(f"Updated: {file_name} -> {out_path}")


if __name__ == "__main__":
    update_tiff_tags()
