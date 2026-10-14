def view():
    redirect("https://example.com/home")
    redirect(request.args.get("next"))
    safe = url_has_allowed_host_and_scheme(request.args.get("next"))
    redirect(safe)
