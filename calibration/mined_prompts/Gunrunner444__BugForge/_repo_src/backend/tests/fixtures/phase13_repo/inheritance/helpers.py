class Base:
    def run(self, value):
        eval(value)


class Child(Base):
    pass
