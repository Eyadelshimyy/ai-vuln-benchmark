class Executor:
    def run(self, value):
        eval(value)

    @staticmethod
    def stat(value):
        eval(value)

    @classmethod
    def build(cls, value):
        eval(value)


class Reporter:
    def run(self, value):
        return value
