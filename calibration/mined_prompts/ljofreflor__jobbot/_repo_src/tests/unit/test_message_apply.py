"""LinkedIn-message apply: detect the channel, never send."""

from __future__ import annotations

from jobbot.portals.message_apply import asks_for_linkedin_message


def test_imperative_message_is_an_apply_route() -> None:
    assert asks_for_linkedin_message(
        "Si te interesa, mandame un mensaje indicando que te interesa este puesto."
    )
    assert asks_for_linkedin_message(
        "Interesados: enviar por mensaje interno su CV y publicaciones."
    )
    assert asks_for_linkedin_message("Send me a DM with your CV.")
    assert asks_for_linkedin_message("Message me if you want the brief.")


def test_past_tense_chatter_is_not_an_apply_route() -> None:
    assert not asks_for_linkedin_message("Te mandé un mensaje ayer sobre el trekking.")
    assert not asks_for_linkedin_message("El mensaje de la marca llega a negocio.")
    assert not asks_for_linkedin_message("")
