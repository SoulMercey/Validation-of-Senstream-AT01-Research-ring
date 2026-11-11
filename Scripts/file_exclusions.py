from pathlib import Path


def _exclusion_file() -> Path:
    return Path(__file__).resolve().with_name("file_exclusions.csv")


def load_excluded_files() -> set[str]:
    exclusion_path = _exclusion_file()
    if not exclusion_path.exists():
        return set()

    with exclusion_path.open("r", encoding="utf-8-sig") as handle:
        return {line.strip() for line in handle if line.strip()}


def is_excluded(path_or_name) -> bool:
    name = Path(path_or_name).name if not isinstance(path_or_name, str) else Path(path_or_name).name
    excluded = load_excluded_files()
    return name.casefold() in {entry.casefold() for entry in excluded}


def filter_csv_paths(paths) -> list:
    excluded = load_excluded_files()
    excluded_norm = {entry.casefold() for entry in excluded}
    filtered = []
    for path in paths:
        name = Path(path).name
        if name.casefold() in excluded_norm:
            continue
        filtered.append(path)
    return filtered
