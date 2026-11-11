"""Validate BioPack/Senstream recording pairs using day, ID, and trial number."""
from collections import defaultdict
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DAYS = ("Dan 1", "Dan 2", "Dan 3")
PAIR_RE = re.compile(r"^(?P<participant>P\d{3})\s+(?P<trial>[12])(?:\s+.*)?$", re.IGNORECASE)


def recording_key(day, name):
    match = PAIR_RE.fullmatch(Path(name).stem.strip())
    if not match:
        return None
    return day, match.group("participant").upper(), match.group("trial")


def _collect_biopac():
    records = defaultdict(list)
    for day in DAYS:
        raw_folder = ROOT / "Meritve" / "BioPack" / day
        for raw_path in sorted(raw_folder.glob("*.acq")) if raw_folder.exists() else ():
            key = recording_key(day, raw_path.name)
            if key:
                csv_path = raw_folder / "csv" / f"{raw_path.stem}.csv"
                records[key].append(csv_path if csv_path.exists() else raw_path)
    return records


def _collect_senstream():
    records = defaultdict(dict)
    for day in DAYS:
        day_folder = ROOT / "Meritve" / "Senstream" / day
        for recording in sorted(day_folder.iterdir()) if day_folder.exists() else ():
            if not recording.is_dir() or recording.name == "Originals":
                continue
            key = recording_key(day, recording.name)
            if not key:
                continue
            for kind in ("raw", "filtered", "accel"):
                path = day_folder / "csv" / kind / f"{recording.name}.csv"
                if path.exists():
                    records[key].setdefault(kind, []).append(path)
    return records


def compare_files():
    biopac, senstream = _collect_biopac(), _collect_senstream()
    paired, unpaired, duplicates = [], [], []
    keys = set(biopac) | set(senstream)
    for key in sorted(keys):
        bp = biopac.get(key, [])
        ss = senstream.get(key, {})
        dup = len(bp) > 1 or any(len(ss.get(kind, [])) > 1 for kind in ("raw", "filtered"))
        if dup:
            duplicates.append((key, len(bp), {kind: len(paths) for kind, paths in ss.items()}))
            continue
        if len(bp) == 1 and bp[0].suffix.lower() == ".csv" and ss.get("raw") and ss.get("filtered"):
            paired.append({
                "name": bp[0].stem, "day": key[0], "participant": key[1], "trial": key[2],
                "biopac_path": str(bp[0]),
                "senstream_paths": {kind: paths[0] for kind, paths in ss.items() if paths},
                "senstream_types": sorted(ss),
            })
            continue
        issues = []
        if len(bp) != 1:
            issues.append("BioPack" if not bp else "duplicate BioPack")
        elif bp[0].suffix.lower() != ".csv":
            issues.append("BioPack trimmed CSV")
        if not ss.get("raw"):
            issues.append("Senstream raw")
        if not ss.get("filtered"):
            issues.append("Senstream filtered")
        unpaired.append({"day": key[0], "name": f"{key[1]} {key[2]}", "key": key, "missing_in": ", ".join(issues)})
    return paired, unpaired, len(biopac), len(senstream), keys, duplicates


def main():
    paired, unpaired, bp_count, ss_count, keys, duplicates = compare_files()
    print("Recording pair validation (day, participant ID, measurement suffix)")
    print(f"BioPack recordings: {bp_count}; Senstream recordings: {ss_count}; unique keys: {len(keys)}")
    print(f"Valid pairs: {len(paired)}; unmatched/incomplete: {len(unpaired)}; duplicate keys: {len(duplicates)}")
    for item in unpaired:
        print(f"UNMATCHED {item['day']} {item['name']}: missing {item['missing_in']}")
    for key, bp_count, ss_counts in duplicates:
        print(f"DUPLICATE {key[0]} {key[1]} {key[2]}: BioPack={bp_count}, Senstream={ss_counts}")
    return paired


if __name__ == "__main__":
    main()
