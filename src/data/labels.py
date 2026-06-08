def resolve_value(name: str, class_values: list) -> float:
    """Resolve a YOLO class-folder name to its BCS value.

    Accepts a value-named folder ("3.25") or an index-named folder ("0"); makes
    no assumption about which scheme the source data uses.
    """
    try:
        value = float(name)
    except ValueError:
        value = None
    if value is not None and any(abs(value - v) < 1e-6 for v in class_values):
        return float(value)

    try:
        index = int(name)
    except ValueError:
        index = None
    if index is not None and 0 <= index < len(class_values):
        return float(class_values[index])

    raise ValueError(f"cannot resolve class name {name!r} against {class_values}")


def class_index_to_value(names: dict, class_values: list) -> dict:
    """Map each YOLO class index to its BCS value, using ``model.names``."""
    return {index: resolve_value(name, class_values) for index, name in names.items()}
