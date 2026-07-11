import ast
import operator


OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPERATORS:
        return OPERATORS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported math expression.")


def solve_math(text: str) -> str:
    expression = " ".join(text.replace("calculate", "").replace("solve", "").split())
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as exc:
        return f"Could not solve math expression: {exc}"


def can_solve_math(text: str) -> bool:
    """Return whether this is a safe, exact arithmetic expression for the local solver."""
    expression = " ".join(text.replace("calculate", "").replace("solve", "").split())
    try:
        tree = ast.parse(expression, mode="eval")
        _eval(tree.body)
        return True
    except Exception:
        return False
