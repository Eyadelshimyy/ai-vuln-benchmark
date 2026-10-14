def view():
    render_template_string(request.args.get("q"))
