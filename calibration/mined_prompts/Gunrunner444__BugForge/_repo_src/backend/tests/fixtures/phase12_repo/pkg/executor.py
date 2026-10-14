class Executor:
    def run(self, value):
        eval(value)


class Reporter:
    def run(self, value):
        return value
