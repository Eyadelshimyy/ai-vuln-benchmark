class Box:
    def run(self):
        self.payload = request.args.get("q")
        eval(self.payload)


class Other:
    def run(self):
        eval(self.payload)
