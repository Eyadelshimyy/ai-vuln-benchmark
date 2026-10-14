"""The car-picture download can come back as a tiny error-JSON body under HTTP 200 — e.g. the
transient `{"code":39,...}` right after a freshly-accepted car share — instead of the ZIP.
command_client must NOT hand that back as 'the package': the web layer would cache it as
car_picture_pkg.zip and then serve no image until a manual ?refresh=1. Only real ZIP bytes
(magic 'PK') are returned; anything else is a transient failure → retry, then None."""
import command_client as C


class _FakeVehicle:
    vin = "LVIN0000000000001"
    car_type = "B10"


class _FakeApi:
    def __init__(self, body):
        self._body = body
        self.downloads = 0

    def login(self):
        pass

    def get_vehicle_list(self):
        return [_FakeVehicle()]

    def close(self):
        pass

    def get_car_picture(self, vehicle):
        return {"data": {"key": "picture-key"}}

    def download_car_picture_package(self, picture_key=None):
        self.downloads += 1
        return self._body


def _run(monkeypatch, body):
    """Fresh session whose every (re-)connect yields a fake api returning `body` for the
    download. Returns (result, total downloads across attempts)."""
    made = []

    def factory():
        api = _FakeApi(body)
        made.append(api)
        return api

    monkeypatch.setattr(C, "_make_client", factory)
    result = C.LeapmotorSession().get_car_picture_package()
    return result, sum(a.downloads for a in made)


def test_error_json_body_is_not_returned_as_package(monkeypatch):
    result, downloads = _run(
        monkeypatch, b'{"code":39,"message":"Information verification failed, try again later"}')
    assert result is None
    assert downloads == 2          # both attempts tried; neither got cached as the package


def test_real_zip_is_returned(monkeypatch):
    zip_bytes = b"PK\x03\x04" + b"\x00" * 32
    result, downloads = _run(monkeypatch, zip_bytes)
    assert result == zip_bytes
    assert downloads == 1          # first attempt succeeds → no needless retry


def test_empty_body_is_not_returned(monkeypatch):
    result, _ = _run(monkeypatch, b"")
    assert result is None
