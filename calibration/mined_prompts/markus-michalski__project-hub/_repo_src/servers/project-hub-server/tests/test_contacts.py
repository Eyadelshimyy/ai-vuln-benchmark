"""Tests for contact CRUD operations."""
import pytest
from tools.contacts import (
    _find_duplicate_shared_contact,
    _match_kind,
    _normalize_name,
    add_contact,
    delete_contact,
    list_contacts,
    list_shared_contacts,
    update_contact,
)
from tools.projects import create_project
from tools.search import search_contacts


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    return create_project("Contact Test Project")


def test_add_contact_basic(project):
    c = add_contact(project["id"], name="Alice Müller", role="PM")
    assert c["id"] is not None
    assert c["name"] == "Alice Müller"
    assert c["role"] == "PM"
    assert c["type"] == "internal"


def test_add_contact_external(project):
    c = add_contact(
        project["id"],
        name="Bob External",
        role="Merchant PM",
        contact_type="external",
        email="bob@merchant.com",
        company="Acme GmbH",
    )
    assert c["type"] == "external"
    assert c["email"] == "bob@merchant.com"
    assert c["company"] == "Acme GmbH"


def test_list_contacts_all(project):
    add_contact(project["id"], name="Internal One", contact_type="internal")
    add_contact(project["id"], name="External One", contact_type="external")

    result = list_contacts(project["id"])
    assert len(result["items"]) == 2
    assert result["total"] == 2


def test_list_contacts_filter_by_type(project):
    add_contact(project["id"], name="Internal A", contact_type="internal")
    add_contact(project["id"], name="Internal B", contact_type="internal")
    add_contact(project["id"], name="External A", contact_type="external")

    internals = list_contacts(project["id"], contact_type="internal")
    externals = list_contacts(project["id"], contact_type="external")

    assert len(internals["items"]) == 2
    assert internals["total"] == 2
    assert len(externals["items"]) == 1


def test_list_contacts_limit(project):
    for i in range(5):
        add_contact(project["id"], name=f"Contact {i}")

    result = list_contacts(project["id"], limit=3)
    assert len(result["items"]) == 3
    assert result["total"] == 5


def test_list_contacts_pagination(project):
    for i in range(5):
        add_contact(project["id"], name=f"Contact {i:02d}")

    page1 = list_contacts(project["id"], limit=3, offset=0)
    page2 = list_contacts(project["id"], limit=3, offset=3)

    assert len(page1["items"]) == 3
    assert len(page2["items"]) == 2
    assert page1["total"] == 5


def test_list_contacts_empty(project):
    result = list_contacts(project["id"])
    assert result["items"] == []
    assert result["total"] == 0


def test_update_contact_patches_only_provided_fields(project):
    c = add_contact(project["id"], name="Original Name", role="PM", email="old@example.com")

    updated = update_contact(c["id"], role="Tech Lead")

    assert updated["role"] == "Tech Lead"
    assert updated["name"] == "Original Name"
    assert updated["email"] == "old@example.com"


def test_update_contact_multiple_fields(project):
    c = add_contact(project["id"], name="Jane", role="PM")

    updated = update_contact(c["id"], name="Jane Doe", email="jane@example.com", company="ACME")

    assert updated["name"] == "Jane Doe"
    assert updated["email"] == "jane@example.com"
    assert updated["company"] == "ACME"


def test_delete_contact_removes_it(project):
    c = add_contact(project["id"], name="To Be Deleted")

    result = delete_contact(c["id"])

    assert result is True
    assert list_contacts(project["id"])["items"] == []


def test_delete_contact_not_found_returns_false(project):
    assert delete_contact(99999) is False


def test_contacts_isolated_per_project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Project Alpha")
    p2 = create_project("Project Beta")

    add_contact(p1["id"], name="Alpha Contact")

    assert list_contacts(p1["id"])["total"] == 1
    assert list_contacts(p2["id"])["total"] == 0


# ---------------------------------------------------------------------------
# Shared contacts
# ---------------------------------------------------------------------------

def test_add_shared_contact(project):
    c = add_contact(project["id"], name="Shared Person", role="PM", is_shared=True)
    assert c["is_shared"] == 1


def test_add_non_shared_contact_default(project):
    c = add_contact(project["id"], name="Local Person", role="Dev")
    assert c["is_shared"] == 0


def test_list_shared_contacts_returns_only_shared(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project("Shared Host")
    add_contact(p["id"], name="Shared One", is_shared=True)
    add_contact(p["id"], name="Shared Two", is_shared=True)
    add_contact(p["id"], name="Local Only", is_shared=False)

    result = list_shared_contacts()
    names = [c["name"] for c in result["items"]]
    assert "Shared One" in names
    assert "Shared Two" in names
    assert "Local Only" not in names
    assert result["total"] == 2


def test_list_shared_contacts_includes_project_info(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project("Info Host")
    add_contact(p["id"], name="Cross Person", is_shared=True)

    result = list_shared_contacts()
    item = result["items"][0]
    assert "project_name" in item
    assert item["project_name"] == "Info Host"


def test_update_contact_can_set_is_shared(project):
    c = add_contact(project["id"], name="Initially Local")
    assert c["is_shared"] == 0

    updated = update_contact(c["id"], is_shared=True)
    assert updated["is_shared"] == 1


def test_update_contact_can_unshare(project):
    c = add_contact(project["id"], name="Initially Shared", is_shared=True)
    assert c["is_shared"] == 1

    updated = update_contact(c["id"], is_shared=False)
    assert updated["is_shared"] == 0


def test_search_contacts_includes_shared_from_other_project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Search Project A")
    p2 = create_project("Search Project B")

    # Shared contact lives in p2 but should appear when searching in p1
    add_contact(p2["id"], name="Riverty PM", role="PM", is_shared=True)
    add_contact(p1["id"], name="Local Contact", role="Dev", is_shared=False)

    results = search_contacts("Riverty", project_id=p1["id"])
    names = [c["name"] for c in results]
    assert "Riverty PM" in names


def test_search_contacts_does_not_include_non_shared_from_other_project(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Filter Project A")
    p2 = create_project("Filter Project B")

    add_contact(p2["id"], name="Private Bob", role="Dev", is_shared=False)

    results = search_contacts("Private Bob", project_id=p1["id"])
    assert results == []


def test_add_external_contact_shared_raises(project):
    import pytest
    with pytest.raises(ValueError, match="External contacts cannot be shared"):
        add_contact(project["id"], name="External Corp", contact_type="external", is_shared=True)


def test_update_contact_shared_external_raises(project):
    import pytest
    c = add_contact(project["id"], name="Ext Contact", contact_type="external")
    with pytest.raises(ValueError, match="External contacts cannot be shared"):
        update_contact(c["id"], is_shared=True)


# ---------------------------------------------------------------------------
# Duplicate shared contact prevention
# ---------------------------------------------------------------------------

def test_add_shared_duplicate_name_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Dup Project A")
    p2 = create_project("Dup Project B")

    add_contact(p1["id"], name="Max Mustermann", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Max Mustermann", is_shared=True)


def test_add_shared_duplicate_email_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Email Dup A")
    p2 = create_project("Email Dup B")

    add_contact(p1["id"], name="Alice One", email="alice@example.com", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Alice Two", email="alice@example.com", is_shared=True)


def test_add_shared_duplicate_name_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Case Dup A")
    p2 = create_project("Case Dup B")

    add_contact(p1["id"], name="max mustermann", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="MAX MUSTERMANN", is_shared=True)


def test_add_non_shared_contact_with_shared_name_blocked(tmp_path, monkeypatch):
    # Even a non-shared add is blocked if a shared contact with that name exists
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Non-Shared Dup A")
    p2 = create_project("Non-Shared Dup B")

    add_contact(p1["id"], name="Common Name", is_shared=True)
    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Common Name", is_shared=False)


def test_update_promote_duplicate_name_raises(tmp_path, monkeypatch):
    # Create local contact FIRST, then add shared one in another project, then try to promote
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Promote Dup A")
    p2 = create_project("Promote Dup B")

    local = add_contact(p2["id"], name="Shared Person", is_shared=False)
    add_contact(p1["id"], name="Shared Person", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        update_contact(local["id"], is_shared=True)


def test_update_reshare_same_contact_allowed(project):
    # Re-saving is_shared=True on an already-shared contact must not raise
    c = add_contact(project["id"], name="Already Shared", is_shared=True)
    updated = update_contact(c["id"], is_shared=True, role="Updated Role")
    assert updated["is_shared"] == 1
    assert updated["role"] == "Updated Role"


def test_add_non_shared_contact_blocked_if_shared_exists(tmp_path, monkeypatch):
    # Even a non-shared add is blocked if a shared contact with that name exists
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Block Non-Shared A")
    p2 = create_project("Block Non-Shared B")

    add_contact(p1["id"], name="Global Employee", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Global Employee", is_shared=False)


def test_rename_contact_to_shared_name_raises(tmp_path, monkeypatch):
    # Renaming a local contact to match a shared contact name must be blocked
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Rename Block A")
    p2 = create_project("Rename Block B")

    add_contact(p1["id"], name="Shared Alice", is_shared=True)
    local = add_contact(p2["id"], name="Local Bob", is_shared=False)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        update_contact(local["id"], name="Shared Alice")


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "variant",
    [
        "Jan-Kalle Wulf",       # hyphen instead of space
        "jan kalle wulf",       # casing
        "Wulf, Jan Kalle",      # Teams "Lastname, Firstname" format
        "Jan  Kalle   Wulf",    # collapsed whitespace
        "Jan Kalle Wulf ",      # trailing whitespace
    ],
)
def test_name_variants_fold_to_same_normalized_name(variant):
    assert _normalize_name(variant) == _normalize_name("Jan Kalle Wulf")


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Michaela Müller", "Michaela Mueller"),   # umlaut transliteration
        ("Jörg Weiß", "Joerg Weiss"),              # umlaut + eszett
        ("JÖRG WEIẞ", "Joerg Weiss"),              # capital eszett U+1E9E
        ("José García", "Jose Garcia"),            # accents
        ("Søren Kirkegård", "Soeren Kirkegaard"),  # slashed o, ring above
        ("Michał Nowak", "Michal Nowak"),          # stroked l, no NFKD decomposition
    ],
)
def test_normalize_folds_transliterations_and_accents(a, b):
    assert _normalize_name(a) == _normalize_name(b)


def test_normalize_handles_decomposed_unicode():
    # NFD input ("u" + combining diaeresis) must fold like the precomposed form, not
    # lose the diaeresis and become "muller"
    nfd = "Michaela Müller"
    assert _normalize_name(nfd) == _normalize_name("Michaela Müller")
    assert _normalize_name(nfd) == "michaela mueller"


@pytest.mark.parametrize("name", ["Иван Петров", "王伟", "محمد علي"])
def test_normalize_keeps_non_latin_scripts(name):
    # Stripping to [a-z0-9] would empty these out and silently disable dedup for them
    assert _normalize_name(name) != ""


def test_non_latin_duplicates_are_still_detected():
    assert _match_kind("Иван Петров", "Иван Петров") == "exact"


def test_normalize_keeps_different_people_apart():
    assert _normalize_name("Markus Ruppel") != _normalize_name("Marius Westhaus")


# ---------------------------------------------------------------------------
# Match classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Jan Kalle Wulf", "Jan-Kalle Wulf"),
        ("Jan Kalle Wulf", "Wulf, Jan Kalle"),
        ("Michaela Ablinger", "Ablinger, Michaela"),
    ],
)
def test_match_kind_exact(a, b):
    assert _match_kind(a, b) == "exact"


@pytest.mark.parametrize(
    ("a", "b", "why"),
    [
        ("Thomas Michael", "Michael Thomas", "token permutation"),
        ("Jan Kalle Wulf", "Jan Wulf", "dropped middle name"),
        ("Anna Schmidt", "Anna Schmidt-Meyer", "double-barrelled surname"),
        ("Mathias Quetz", "Matthias Quetz", "spelling variant"),
        ("Ben Zimmermann", "Sven Zimmermann", "high similarity, distinct people"),
    ],
)
def test_match_kind_near(a, b, why):
    assert _match_kind(a, b) == "near", why


def test_match_kind_unrelated():
    assert _match_kind("Markus Ruppel", "Marius Westhaus") == ""


def test_token_permutation_is_near_not_exact():
    # Regression: sorting tokens for the exact comparison made "Thomas Michael" and
    # "Michael Thomas" collide as exact, which force cannot override — leaving no way
    # to add the second person at all.
    assert _match_kind("Thomas Michael", "Michael Thomas") == "near"


def test_permuted_distinct_person_can_be_added_with_force(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Permute A")
    p2 = create_project("Permute B")

    add_contact(p1["id"], name="Thomas Michael", is_shared=True)
    c = add_contact(p2["id"], name="Michael Thomas", is_shared=True, force=True)
    assert c["name"] == "Michael Thomas"


def test_dropped_middle_name_is_detected(tmp_path, monkeypatch):
    # "Jan Kalle Wulf" vs "Jan Wulf" scores 0.727 — below threshold, so plain similarity
    # would miss it. Dropping a middle name is as common as hyphenating one.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Subset A")
    p2 = create_project("Subset B")

    add_contact(p1["id"], name="Jan Kalle Wulf", is_shared=True)
    with pytest.raises(ValueError, match="very similar name already exists"):
        add_contact(p2["id"], name="Jan Wulf", is_shared=True)


def test_add_shared_duplicate_hyphen_variant_raises(tmp_path, monkeypatch):
    # The case that produced a real duplicate: SOP spells the name without a hyphen,
    # Teams spells it with one.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Hyphen Dup A")
    p2 = create_project("Hyphen Dup B")

    add_contact(p1["id"], name="Jan Kalle Wulf", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Jan-Kalle Wulf", is_shared=True)


def test_add_shared_duplicate_lastname_first_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Order Dup A")
    p2 = create_project("Order Dup B")

    add_contact(p1["id"], name="Michaela Ablinger", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Ablinger, Michaela", is_shared=True)


# ---------------------------------------------------------------------------
# Near-match detection and force override
# ---------------------------------------------------------------------------

def test_add_shared_near_duplicate_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Near Dup A")
    p2 = create_project("Near Dup B")

    add_contact(p1["id"], name="Mathias Quetz", is_shared=True)

    with pytest.raises(ValueError, match="very similar name already exists"):
        add_contact(p2["id"], name="Matthias Quetz", is_shared=True)


def test_add_shared_near_duplicate_force_allows(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Force Dup A")
    p2 = create_project("Force Dup B")

    add_contact(p1["id"], name="Mathias Quetz", is_shared=True)
    c = add_contact(p2["id"], name="Matthias Quetz", is_shared=True, force=True)

    assert c["name"] == "Matthias Quetz"


def test_force_does_not_override_exact_duplicate(tmp_path, monkeypatch):
    # force is an escape hatch for false positives, not a way to create real duplicates
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Force Exact A")
    p2 = create_project("Force Exact B")

    add_contact(p1["id"], name="Jan Kalle Wulf", is_shared=True)

    with pytest.raises(ValueError, match="shared contact with this name or email already exists"):
        add_contact(p2["id"], name="Jan-Kalle Wulf", is_shared=True, force=True)


def test_exact_match_wins_over_near_match(tmp_path, monkeypatch):
    # An exact match must be reported even when a near match is scanned first
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project("Precedence Host")

    add_contact(p["id"], name="Mathias Quetz", is_shared=True)
    add_contact(p["id"], name="Matthias Quetz", is_shared=True, force=True)

    existing, kind = _find_duplicate_shared_contact("Matthias Quetz")
    assert kind == "exact"
    assert existing["name"] == "Matthias Quetz"


def test_near_match_reports_closest_candidate(tmp_path, monkeypatch):
    # With several near matches, the error must name the likeliest one, not whichever
    # row SQLite returned first
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p = create_project("Closest Host")

    add_contact(p["id"], name="Ben Zimmermann", is_shared=True)
    add_contact(p["id"], name="Sven Zimmermann", is_shared=True, force=True)

    existing, kind = _find_duplicate_shared_contact("Sven Zimmermanns")
    assert kind == "near"
    assert existing["name"] == "Sven Zimmermann"


def test_update_near_duplicate_force_allows(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Update Force A")
    p2 = create_project("Update Force B")

    add_contact(p1["id"], name="Mathias Quetz", is_shared=True)
    local = add_contact(p2["id"], name="Local Person", is_shared=False)

    updated = update_contact(local["id"], name="Matthias Quetz", is_shared=True, force=True)
    assert updated["name"] == "Matthias Quetz"


def test_update_contact_rejects_positional_second_arg(project):
    # force is keyword-only: a positional dict must stay a TypeError instead of being
    # silently swallowed as a truthy force, leaving the update a no-op
    c = add_contact(project["id"], name="Positional Guard")
    with pytest.raises(TypeError):
        update_contact(c["id"], {"name": "Ignored"})  # type: ignore[arg-type]


def test_unrelated_names_are_not_near_matches(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Unrelated A")
    p2 = create_project("Unrelated B")

    add_contact(p1["id"], name="Markus Ruppel", is_shared=True)
    c = add_contact(p2["id"], name="Marius Westhaus", is_shared=True)

    assert c["name"] == "Marius Westhaus"


def test_non_shared_contacts_are_not_dedup_targets(tmp_path, monkeypatch):
    # Only shared contacts are deduplicated; project-local contacts stay independent
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Local Dup A")
    p2 = create_project("Local Dup B")

    add_contact(p1["id"], name="Same Name", is_shared=False)
    c = add_contact(p2["id"], name="Same Name", is_shared=False)

    assert c["name"] == "Same Name"


def test_local_add_is_not_blocked_by_near_match_against_shared(tmp_path, monkeypatch):
    # A heuristic name similarity must not refuse a legitimately project-scoped contact.
    # An exact match still blocks — see test_add_non_shared_contact_with_shared_name_blocked.
    monkeypatch.setattr("tools.projects.get_docs_root", lambda: tmp_path)
    p1 = create_project("Local Near A")
    p2 = create_project("Local Near B")

    add_contact(p1["id"], name="Mathias Quetz", is_shared=True)
    c = add_contact(p2["id"], name="Matthias Quetz", is_shared=False)

    assert c["name"] == "Matthias Quetz"
    assert c["is_shared"] == 0
