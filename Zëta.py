import re

class Zeta:
    def __init__(self, type_, value=None):
        self.type_ = type_
        self.value = value

    @staticmethod
    def lexer(text):
        TOKEN_PATTERNS = [
            (r'\brepeat\b', 'LOOP'),
            (r'\bintent\b', 'SEMANTIC'),
            (r'\bencrypt\b', 'SECURITY'),
            (r'\basync\b', 'PARALLEL'),
            (r'\bfunction\b', 'FUNCTION'),
            (r'\bmath\b', 'MATH'),
            (r'\bvar\b', 'VARIABLE'),
            (r'\bstring\b', 'STRING'),
            (r'\bboolean\b', 'BOOLEAN'),
            (r'\btrue\b', 'BOOLEAN_TRUE'),
            (r'\bfalse\b', 'BOOLEAN_FALSE'),
            (r'\bnull\b', 'NULL'),
            (r'\bundefined\b', 'UNDEFINED'),
            (r'\bclass\b', 'CLASS'),
            (r'\bobject\b', 'OBJECT'),
            (r'\bthis\b', 'THIS'),
            (r'\bnew\b', 'NEW'),
            (r'\breturn\b', 'RETURN'),
            (r'\bimport\b', 'IMPORT'),
            (r'\bexport\b', 'EXPORT'),
            (r'\bpublic\b', 'PUBLIC'),
            (r'\bprivate\b', 'PRIVATE'),
            (r'\bprotected\b', 'PROTECTED'),
            (r'\bstatic\b', 'STATIC'),
            (r'\bconst\b', 'CONSTANT'),
            (r'\blet\b', 'LET'),
            (r'\bif\b', 'IF'),
            (r'\bthen\b', 'THEN'),
            (r'\buntil\b', 'UNTIL'),
            (r'\bfor\b', 'FOR'),
            (r'\bprint\b', 'PRINT'),
            (r'==', 'EQUALS'),
            (r'!=', 'NOT_EQUALS'),
            (r'<=', 'LESS_THAN_EQUALS'),
            (r'>=', 'GREATER_THAN_EQUALS'),
            (r'<', 'LESS_THAN'),
            (r'>', 'GREATER_THAN'),
            (r'#.*', None),         # Ignore Python-style comments
            (r'//.*', None),        # Ignore C/C++/JavaScript-style comments
            (r'\d+', 'NUMBER'),
            (r'[+\-*/()]', "OPERATOR"),
            (r'=', 'ASSIGN'),
            (r'\s+', None),
            (r'\bclass\b', 'CLASS'),
            (r'\{', 'LBRACE'),
            (r'\}', 'RBRACE'),
            (r':', 'COLON'),  # <-- Add this line to recognize colon
            (r';', 'SEMICOLON'),
            (r'\(', 'LPAREN'),
            (r'\)', 'RPAREN'),
            (r'\[', 'LBRACKET'),
            (r'\]', 'RBRACKET'),
            (r',', 'COMMA'),
            (r'\.', 'DOT'),
            (r'[a-zA-Z_]\w*', 'IDENTIFIER'),  # Identifiers (variable names, function names, etc.)
            (r'\".*?\"', 'STRING_LITERAL'),  # Double quotes
            (r'\'.*?\'', 'STRING_LITERAL'),  # Single quotes
            (r'\"[^"]*\"', 'STRING_LITERAL'),  # Double quotes with escaped characters
            (r'\'[^\']*\'', 'STRING_LITERAL')  # Single quotes with escaped characters
        ]
        tokens = []
        text = text.strip()  # Remove leading/trailing whitespace
        while text:
            matched = False
            for pattern, token_type in TOKEN_PATTERNS:
                match = re.match(pattern, text)
                if match:
                    matched = True
                    if token_type:
                        tokens.append((match.group(), token_type))
                    text = text[len(match.group()):].lstrip()  # Also strip leading whitespace after each match
                    break
            if not matched:
                # Show more context for debugging
                raise ValueError(f"Unexpected character: {text[:10]!r}")
        return tokens

class Parser:
    def __init__(self, tokens, variables=None):
        self.tokens = tokens
        self.pos = 0
        self.variables = variables if variables is not None else {}

    def find_token_positions(tokens, target_tokens):
        """Find positions of specific tokens in the token list."""
        positions = {token: [] for token in target_tokens}
        for index, (token, _) in enumerate(tokens):
            if token in positions:
                positions[token].append(index)
        return positions

    def index_tokens(self):
        """Creates a dictionary mapping tokens to their positions (all occurrences)."""
        token_map = {}
        for index, (token, token_type) in enumerate(self.tokens):
            if token not in token_map:
                token_map[token] = []
            token_map[token].append(index)
        return token_map

    def parse_statement(self):
        """Parse a statement based on the first token."""
        if self.tokens[0][0] == "let":
            return self.parse_variable()
        elif self.tokens[0][0] == "repeat":
            return self.parse_loop()
        elif self.tokens[0][0] == "if":
            return self.parse_condition_and_result()
        elif self.tokens[0][0] == "encrypt":
            return self.parse_condition_and_result()
        elif self.tokens[0][0] == "async":
            return self.parse_condition_and_result()
        elif self.tokens[0][0] == "intent":
            return self.parse_condition_and_result()
        elif self.tokens[0][0] == "func":
            return self.parse_function()
        elif self.tokens[0][0] == "math":
            return self.parse_math()
        elif self.tokens[0][0] == "print":
            return self.parse_print()
        elif isinstance(self.tokens[0][0], str) and self.tokens[0][1] == "IDENTIFIER":  
            # Handle variable assignment or function call
            if self.pos + 2 < len(self.tokens) and self.tokens[self.pos + 1][1] == "ASSIGN":
                return self.parse_variable()
            elif self.tokens[self.pos][0] in self.variables:
                func = self.variables[self.tokens[self.pos][0]]
                if isinstance(func, dict) and func.get("type") == "function":
                    self.pos += 1  # Move past the function name
                    return {"call_function": self.tokens[self.pos - 1][0]}
                else:
                    raise SyntaxError(f"Unknown statement or variable: {self.tokens[0][0]}")
            else:
                raise SyntaxError(f"Unknown statement or variable: {self.tokens[0][0]}")
        # Function call support: if the first token is an identifier and matches a function in variables
        elif self.tokens[0][1] == "IDENTIFIER" and self.tokens[0][0] in self.variables:
            func = self.variables[self.tokens[0][0]]
            if isinstance(func, dict) and func.get("type") == "function":
                return {"call_function": self.tokens[0][0]}
            else:
                raise SyntaxError(f"Unknown statement or variable: {self.tokens[0][0]}")
        else:
            raise SyntaxError(f"Unexpected token {self.tokens[0][0]}")

    def parse_variable(self):
        self.pos += 1  # Skip 'let'
        if self.pos + 3 >= len(self.tokens):
            raise SyntaxError("Incomplete variable assignment")
        var_type, _ = self.tokens[self.pos]
        self.pos += 1
        var_name, _ = self.tokens[self.pos]
        self.pos += 1
        if self.tokens[self.pos][1] != "ASSIGN":
            raise SyntaxError(f"Expected '=', found {self.tokens[self.pos][0]}")
        self.pos += 1
        value, value_type = self.tokens[self.pos]
        self.pos += 1
        # Store variable in symbol table
        self.variables[var_name] = value
        return f"Variable `{var_name}` of type `{var_type}` assigned value `{value}`"

    def parse_loop(self):
        self.pos += 1  # Skip 'repeat'
        if self.pos < len(self.tokens) and self.tokens[self.pos][1] == "NUMBER":
            count_token, _ = self.tokens[self.pos]
            self.pos += 1
            if self.pos < len(self.tokens) and self.tokens[self.pos][0] == ":":
                self.pos += 1
            # Indicate to the interpreter that a loop is needed and collect body lines
            return {"loop_count": int(count_token), "body_lines": self.collect_body_lines()}
        else:
            raise SyntaxError("Expected repeat <number>:")

    def parse_condition_and_result(self):
        """Extracts the condition between `if` and `then`."""
        token_positions = self.index_tokens()
        if "if" in token_positions and "then" in token_positions:
            if_index = token_positions["if"][0]
            then_index = token_positions["then"][0]
            if then_index > if_index:
                condition_tokens = self.tokens[if_index + 1 : then_index]
                condition = " ".join(tok for tok, _ in condition_tokens)
                result_tokens = self.tokens[then_index + 1 : len(self.tokens)]
                result = " ".join(tok for tok, _ in result_tokens)
                return f"Parsed Condition: {condition} and result {result}"
            else:
                raise SyntaxError("`then` appears before `if`—invalid syntax.")
        else:
            raise SyntaxError("Missing `if` or `then` in statement.")

    def parse_function(self):
        self.pos += 1
        func_name, _ = self.tokens[self.pos]
        self.pos += 1
        # Optionally expect a colon
        if self.pos < len(self.tokens) and self.tokens[self.pos][0] == ":":
            self.pos += 1
        # Collect function body lines
        body_lines = self.collect_body_lines()
        # Store function definition (not callable yet, but parsed)
        self.variables[func_name] = {"type": "function", "body_lines": body_lines}
        return f"Function {func_name} defined with {len(body_lines)} lines"

    def collect_body_lines(self):
        # Prompt for lines until 'end' is entered
        lines = []
        while True:
            line = input("...body> ")
            if line.strip() == "end":
                break
            lines.append(line)
        return lines

    def parse_math(self):
        self.pos += 1  # Skip 'math'
        num1_token, _ = self.tokens[self.pos]
        op_token, _ = self.tokens[self.pos + 1]
        num2_token, _ = self.tokens[self.pos + 2]
        self.pos += 3
        try:
            num1 = int(num1_token)
            num2 = int(num2_token)
        except ValueError:
            raise SyntaxError("Invalid number format.")
        result = None
        op_word = op_token
        if op_token == '+':
            result = num1 + num2
            op_word = '+'
        elif op_token == '-':
            result = num1 - num2
            op_word = '-'
        elif op_token == '*':
            result = num1 * num2
            op_word = '*'
        elif op_token == '/':
            result = num1 / num2
            op_word = '/'
        elif op_token == '<':
            result = num1 < num2
            op_word = "less than"
        elif op_token == '>':
            result = num1 > num2
            op_word = "greater than"
        elif op_token == '<=':
            result = num1 <= num2
            op_word = "less than or equal to"
        elif op_token == '>=':
            result = num1 >= num2
            op_word = "greater than or equal to"
        elif op_token == '==':
            result = num1 == num2
            op_word = "equal to"
        elif op_token == '!=':
            result = num1 != num2
            op_word = "not equal to"
        else:
            raise SyntaxError(f"Unsupported operator {op_token}")
        return f"Result of {num1} {op_word} {num2} is {result}"

    def parse_print(self):
        self.pos += 1  # Skip 'print'
        # Support print{...}
        if self.pos < len(self.tokens) and self.tokens[self.pos][1] == "LBRACE":
            self.pos += 1  # Skip '{'
            # Collect tokens until '}'
            content_tokens = []
            while self.pos < len(self.tokens) and self.tokens[self.pos][1] != "RBRACE":
                content_tokens.append(self.tokens[self.pos])
                self.pos += 1
            if self.pos >= len(self.tokens) or self.tokens[self.pos][1] != "RBRACE":
                raise SyntaxError("Expected '}' after print{")
            self.pos += 1  # Skip '}'
            # Only one token: variable or string
            if len(content_tokens) == 1:
                token, token_type = content_tokens[0]
                if token_type == "IDENTIFIER":
                    if token in self.variables:
                        return str(self.variables[token])
                    else:
                        return f"Undefined variable: {token}"
                elif token_type == "STRING_LITERAL":
                    # Remove quotes
                    return token[1:-1]
            # Multiple tokens: join as string
            return " ".join(token for token, _ in content_tokens)
        else:
            # Fallback: old print behavior
            to_print_tokens = self.tokens[self.pos:]
            if not to_print_tokens:
                return ""
            if len(to_print_tokens) == 1 and to_print_tokens[0][1] == "IDENTIFIER":
                var_name = to_print_tokens[0][0]
                if var_name in self.variables:
                    return str(self.variables[var_name])
            to_print = " ".join(token[0] for token in to_print_tokens)
            return f"{to_print}"

class Interpreter:
    def __init__(self, parser):
        self.parser = parser

    def interpret(self):
        result = self.parser.parse_statement()
        if isinstance(result, dict) and "loop_count" in result:
            loop_count = result["loop_count"]
            body_lines = result.get("body_lines", [])
            for _ in range(loop_count):
                for line in body_lines:
                    body_tokens = Zeta.lexer(line)
                    body_parser = Parser(body_tokens, self.parser.variables)
                    body_interpreter = Interpreter(body_parser)
                    body_interpreter.interpret()
                self.parser.variables = body_parser.variables
        # Function call support
        elif isinstance(result, dict) and "call_function" in result:
            func_name = result["call_function"]
            func = self.parser.variables.get(func_name)
            if func and isinstance(func, dict) and func.get("type") == "function":
                for line in func.get("body_lines", []):
                    body_tokens = Zeta.lexer(line)
                    body_parser = Parser(body_tokens, self.parser.variables)
                    body_interpreter = Interpreter(body_parser)
                    body_interpreter.interpret()
                self.parser.variables = body_parser.variables
            else:
                print(f"Function {func_name} not defined.")
        else:
            if result:
                print(result)
            else:
                print("No valid statement to interpret.")

if __name__ == "__main__":
    # Simple REPL loop for coding in your language
    variables = {}
    while True:
        try:
            text = input("Zëta> ")
            if text.strip().lower() in {"exit", "quit"}:
                break
            tokens = Zeta.lexer(text)
            parser = Parser(tokens, variables)
            interpreter = Interpreter(parser)
            interpreter.interpret()
            # Update variables for next statement
            variables = parser.variables
        except Exception as e:
            print(f"Error: {e}")