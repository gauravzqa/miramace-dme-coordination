import pytest

from miramace.agents.schemas import LLM_SCHEMAS

_pyd = pytest.importorskip("openai.lib._pydantic", reason="openai strict-schema helper not available")
to_strict_json_schema = getattr(_pyd, "to_strict_json_schema", None)


def _objects(schema: dict):
    """Yield every object-typed sub-schema (root, $defs, nested)."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            yield schema
        for v in schema.values():
            yield from _objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _objects(v)


@pytest.mark.parametrize("model", LLM_SCHEMAS, ids=lambda m: m.__name__)
def test_strict_schema(model):
    if to_strict_json_schema is None:
        pytest.skip("to_strict_json_schema missing")
    schema = to_strict_json_schema(model)
    assert schema.get("properties")
    for obj in _objects(schema):
        props = set(obj.get("properties", {}))
        assert props == set(obj.get("required", [])), (model.__name__, props, obj.get("required"))
        assert obj.get("additionalProperties") is False
        for p in obj.get("properties", {}).values():
            assert "default" not in p, (model.__name__, p)


@pytest.mark.parametrize("model", LLM_SCHEMAS, ids=lambda m: m.__name__)
def test_no_defaults_in_llm_schemas(model):
    for name, f in model.model_fields.items():
        assert f.is_required(), f"{model.__name__}.{name} must be required (no default)"
