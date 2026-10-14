import importlib

import pytest


def providers():
    return importlib.import_module("replycoach.generate.providers")


@pytest.mark.parametrize("provider,role", [("v4", "C"), ("v5", "D")])
def test_pending_generator_names_the_owner_without_fake_outputs(provider, role):
    module = providers()
    with pytest.raises(module.GeneratorUnavailable, match=f"Role {role}"):
        module.load_generator(provider)


def test_unknown_provider_is_not_dynamically_imported():
    with pytest.raises(ValueError, match="provider"):
        providers().load_generator("some.arbitrary.module")


def test_role_c_factory_is_the_documented_insertion_point(monkeypatch, stub_generator):
    module = providers()
    monkeypatch.setattr(module, "load_v4", lambda: stub_generator)
    assert module.load_generator("v4") is stub_generator


def test_provider_rejects_wrong_model_source(monkeypatch, stub_generator):
    module = providers()
    monkeypatch.setattr(module, "load_v5", lambda: stub_generator)
    with pytest.raises(ValueError, match="V5"):
        module.load_generator("v5")


@pytest.mark.parametrize("provider,role", [("v4", "C"), ("v5", "D")])
def test_missing_weight_file_is_attributed_to_model_owner(monkeypatch, provider, role):
    module = providers()

    def missing_file():
        raise FileNotFoundError("checkpoint missing")

    monkeypatch.setattr(module, f"load_{provider}", missing_file)
    with pytest.raises(module.GeneratorUnavailable, match=f"Role {role}"):
        module.load_generator(provider)


def test_incomplete_factory_object_is_a_clear_contract_error(monkeypatch):
    module = providers()
    monkeypatch.setattr(module, "load_v4", lambda: object())
    with pytest.raises(ValueError, match="ReplyGenerator"):
        module.load_generator("v4")
