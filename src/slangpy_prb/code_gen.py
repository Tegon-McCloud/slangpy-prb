class CodeBuilder:    
    def __init__(self):
        super().__init__()

        self.segments: list[str] = []
        self.indentation = 0

    def inc_indent(self):
        self.indentation += 1

    def dec_indent(self):
        if self.indentation > 0:
            self.indentation -= 1

    def append_indent(self):
        self.segments.append("    " * self.indentation)

    def append_code(self, code: str):
        self.segments.append(code)

    def newline(self):
        self.append_code("\n")

    def append_line(self, code: str):
        self.append_indent()
        self.append_code(code)
        self.newline()

    def append_code_indented(self, code: str):
        for line in code.splitlines():
            self.append_line(line)

    def begin_block(self):
        self.append_line("{")
        self.inc_indent()

    def end_block(self):
        self.dec_indent()
        self.append_line("}")

    def declare(self, type: str, name: str, value: str | None = None):
        if value != None:
            self.append_line(f"{type} {name} = {value};")
        else:
            self.append_line(f"{type} {name};")

    def assign(self, name: str, value: str):
        self.append_line(f"{name} = {value};")

    def build(self) -> str:
        return "".join(self.segments)