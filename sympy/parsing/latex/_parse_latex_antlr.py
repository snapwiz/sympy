# Ported from latex2sympy by @augustt198
# https://github.com/augustt198/latex2sympy
# See license in LICENSE.txt
import re
from importlib.metadata import version
import sympy
from sympy import ImmutableMatrix as Matrix
from sympy.abc import n
from sympy.external import import_module
from sympy.logic.boolalg import BooleanAtom
from sympy.physics.quantum.state import Bra, Ket
from sympy.printing.str import StrPrinter
from sympy.vector import Vector, CoordSys3D, matrix_to_vector
from . import LaTeXParsingContext
from .errors import LaTeXParsingError

coeff = r'([A-Wa-w]|\d+(\.\d+)?|-?\d+/-?\d+|\\frac\{[^-]+?\}\{[^-]+?\}|' \
        + r'\d+ *(\d+/\d+|\\frac\{[^-]+?\}\{[^-]+?\})|\\sqrt\{\d+\}|\\pi|)'
fraction_plus_re = re.compile(
    r'(-?)\s*(\d+\.?\d*)(\s*(\\frac{{{}}}{{{}}})|\s+({}/{}|\(?\d+\)?/\(?\d+\)?))'.format(coeff, coeff, coeff, coeff))
fraction_derivative_re = re.compile(r'(\\frac{)(d(\^\d)?)(}{d[^}]*})((\([^()]*\)|{[^{}]*}|[^\+\-])*)')
multi_differential_re = re.compile(r'd[^d]+')
number_error_re = re.compile(
    r'(,\d{,2}(,|$))|(^0\d{1,}(,|$))|(\d{4,},)|(,\d{4,}$)'
)

LaTeXParser = LaTeXLexer = MathErrorListener = PredictionMode = None

try:
    LaTeXParser = import_module('sympy.parsing.latex._antlr.latexparser',
                                import_kwargs={'fromlist': ['LaTeXParser']}).LaTeXParser
    LaTeXLexer = import_module('sympy.parsing.latex._antlr.latexlexer',
                               import_kwargs={'fromlist': ['LaTeXLexer']}).LaTeXLexer
    PredictionMode = import_module('antlr4.atn.PredictionMode',
                               import_kwargs={'fromlist': ['PredictionMode']}).PredictionMode
except Exception as e:
    pass

ErrorListener = import_module('antlr4.error.ErrorListener',
                              warn_not_installed=True,
                              import_kwargs={'fromlist': ['ErrorListener']}
                              )


class Structure():
    """
    Brief:
        Class representing structure

    TODO: rethink on need and placement of this class
    Q. Can it be replaced with any python class, like list, set, etc ?

    Description:
        Class containing elements.
        Two structure_type are possible:
            list:
                Elements order is important
            set:
                Elements order is not important

    """

    def __init__(self, structure_type):
        self.structure_type = structure_type
        self.elements = []

    def append(self, element):
        self.elements.append(element)

    def sort(self):
        for inner_struct in self.elements:
            if (isinstance(inner_struct, Structure)):
                inner_struct.sort()
                if (inner_struct.structure_type != 'list'):
                    inner_struct.elements.sort(key=str)

    def sortFirstLevel(self):
        if (self.structure_type != 'list'):
            self.elements.sort(key=str)

    def union(self, other):
        for element in other.elements:
            if element not in self.elements:
                self.elements.append(element)
        return self

    def complement(self, other):
        for element in other.elements:
            if element in self.elements:
                self.elements.remove(element)
        return self

    def intersect(self, other):
        self_elements = list(self.elements)
        for element in self_elements:
            if element not in other.elements:
                self.elements.remove(element)
        return self

    def contains(self, other):
        return other in self.elements

    def subs(self, subs_list):
        elements = []
        for element in self.elements:
            elements.append(element.subs(subs_list))
        self.elements = elements

    def clear(self):
        if not self.structure_type == 'list':
            elements = []
            for element in self.elements:
                if element not in elements:
                    elements.append(element)
            self.elements = elements

    def __eq__(self, other):

        if (isinstance(other, Structure)):
            if (self.structure_type == other.structure_type):
                return self.elements == other.elements

        else:
            if self.structure_type == 'set' or self.structure_type == 'list':
                if len(self.elements) == 1:
                    return other == self.elements[0]
                return other == self.elements
        return 0

    def __str__(self):
        return str(self.elements)

    def __repr__(self) -> str:
        return (
            f"Structure({self.structure_type}) {{"
            + (", ".join(repr(e) for e in self.elements))
            + "}"
        )


if ErrorListener:
    class MathErrorListener(ErrorListener.ErrorListener):  # type:ignore # noqa:F811
        def __init__(self, src):
            super(ErrorListener.ErrorListener, self).__init__()
            self.src = src

        def syntaxError(self, recog, symbol, line, col, msg, e):
            fmt = "%s\n%s\n%s"
            marker = "~" * col + "^"

            if msg.startswith("missing"):
                err = fmt % (msg, self.src, marker)
            elif msg.startswith("no viable"):
                err = fmt % ("I expected something else here", self.src, marker)
            elif msg.startswith("mismatched"):
                names = LaTeXParser.literalNames
                expected = [
                    names[i] for i in e.getExpectedTokens() if i < len(names)
                ]
                if len(expected) < 10:
                    expected = " ".join(expected)
                    err = (fmt % ("I expected one of these: " + expected, self.src,
                                  marker))
                else:
                    err = (fmt % ("I expected something else here", self.src,
                                  marker))
            else:
                err = fmt % ("I don't understand this", self.src, marker)
            raise LaTeXParsingError(err)


def process_sympy(source, parser, strict: bool):
        math_e = parser.math()
        if relation := math_e.relation():
            if strict and (relation.start.start != 0 or relation.stop.stop != len(source) - 1):
                raise LaTeXParsingError("Invalid LaTeX")
            return convert_relation(relation)
        elif struct_relation := math_e.struct_relation():
            if strict and (struct_relation.start.start != 0 or struct_relation.stop.stop != len(source) - 1):
                raise LaTeXParsingError("Invalid LaTeX")
            return convert_struct_relation(struct_relation)
        elif equation_list := math_e.equation_list():
            if strict and (equation_list.start.start != 0 or equation_list.stop.stop != len(source) - 1):
                raise LaTeXParsingError("Invalid LaTeX")
            return convert_equation_list(equation_list)
        raise LaTeXParsingError("Latex2SympyError")

def process_set(source: str, parser, strict: bool):
    try:
        struct_f = parser.struct_form()
        if strict and (struct_f.start.start != 0 or struct_f.stop.stop != len(source) - 1):
            raise LaTeXParsingError("Invalid LaTeX")
        expr = convert_struct_form(struct_f)
        default_struct = LaTeXParsingContext.getOption('default_struct')
        if (not isinstance(expr, Structure)):
            structObject = Structure(default_struct)
            structObject.append(expr)
            expr = structObject

        if (expr.structure_type == 'any'):
            expr.structure_type = default_struct
            expr.clear()
        return expr
    except LaTeXParsingError:
        raise
    except Exception as e:
        raise LaTeXParsingError("Failed parsing set") from e

def build_parser(input: str, matherror):
    antlr4 = import_module('antlr4')

    stream = antlr4.InputStream(input)
    lex = LaTeXLexer(stream)
    lex.removeErrorListeners()
    lex.addErrorListener(matherror)

    tokens = antlr4.CommonTokenStream(lex)
    parser = LaTeXParser(tokens)

    # remove default console error listener
    parser.removeErrorListeners()
    parser.addErrorListener(matherror)
    return parser


def parse_latex(source, strict=False):
    antlr4 = import_module('antlr4')

    if None in [antlr4, MathErrorListener] or \
            not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError("LaTeX parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime) or"
                          " conda (antlr-python-runtime), version 4.11")

    source = source.strip()
    matherror = MathErrorListener(source)

    parser = build_parser(source, matherror)

    if LaTeXParsingContext.getOption('force_set') is True:
        return process_set(source, parser, strict)
    else:
        if re.search(r'\d\s+\d', source):
            raise LaTeXParsingError('FractionError: space not allowed between digits')
        try:
            parser._interp.predictionMode = PredictionMode.SLL
            return process_sympy(source, parser, strict)
        except Exception:
            pass
        try:
            parser = build_parser(source, matherror)
            return process_sympy(source, parser, strict)
        except LaTeXParsingError:
            raise
        except Exception as e:
            raise LaTeXParsingError(f"Error parsing '{source}'") from e

def convert_struct_form(form):
    if(len(form.value()) == 1):
        return convert_value(form.value()[0])
    else:
        structObject = Structure('any')
        for x in form.value():
            structObject.append(convert_value(x))
    return structObject

def convert_value(value):
    if value.struct_value():
        return convert_struct_value(value.struct_value())
    elif value.relation():
        return convert_relation(value.relation())

def convert_struct_value(form):
    l_par = form.left_parentheses().getText()
    r_par = form.right_parentheses().getText()
    if((l_par == '(' and r_par == ')')
        or (l_par == '[' and r_par == ']')):
        name = 'list'

    elif(l_par == '{' and r_par == '}'):
        name = 'set'

    else:
        name = l_par + r_par
    structObject = Structure(name)
    for x in form.value():
        structObject.append(convert_value(x))

    return structObject

def convert_relation(rel):
    if rel.expr():
        return convert_expr(rel.expr())

    lh = convert_relation(rel.relation(0))
    rh = convert_relation(rel.relation(1))
    if isinstance(lh, list):
        return lh + [rh]
    if isinstance(rh, list):
        return rh + [lh]
    if rel.LT():
        return sympy.StrictLessThan(lh, rh, evaluate=False)
    elif rel.LTE():
        return sympy.LessThan(lh, rh, evaluate=False)
    elif rel.GT():
        return sympy.StrictGreaterThan(lh, rh, evaluate=False)
    elif rel.GTE():
        return sympy.GreaterThan(lh, rh, evaluate=False)
    elif rel.EQUAL():
        return sympy.Eq(lh, rh, evaluate=False)
    elif rel.NEQ():
        return sympy.Ne(lh, rh, evaluate=False)
    elif rel.EQUIV():
        return sympy.Eq(lh, rh, evaluate=False)

def convert_struct_relation(rel):
    if rel.struct_expr():
        return convert_struct_expr(rel.struct_expr())

    lh = convert_struct_relation(rel.struct_relation(0))
    rh = convert_struct_relation(rel.struct_relation(1))

    if rel.EQUAL():
        return (lh, '=', rh)

def convert_equation_list(equation_list):
    equations = []
    for equation in equation_list.equation():
        equations.append(convert_equation(equation))
    return equations

def convert_equation(equation):
    lh = convert_relation(equation.relation(0))
    rh = convert_relation(equation.relation(1))
    if isinstance(lh, list):
        return lh + [rh]
    if isinstance(rh, list):
        return rh + [lh]
    if equation.LT():
        return sympy.StrictLessThan(lh, rh,evaluate=False)
    elif equation.LTE():
        return sympy.LessThan(lh, rh,evaluate=False)
    elif equation.GT():
        return sympy.StrictGreaterThan(lh, rh,evaluate=False)
    elif equation.GTE():
        return sympy.GreaterThan(lh, rh,evaluate=False)
    elif equation.EQUAL():
        return sympy.Eq(lh, rh,evaluate=False)
    elif equation.NEQ():
        return sympy.Ne(lh, rh,evaluate=False)
    elif equation.EQUIV():
        return sympy.Eq(lh, rh,evaluate=False)

def convert_struct_expr(rel):
    if rel.struct_value():
        return convert_struct_value(rel.struct_value())

    lh = convert_struct_expr(rel.struct_expr(0))
    rh = convert_struct_expr(rel.struct_expr(1))

    if rel.SET_ADD():
        return (lh, '+', rh)
    elif rel.SET_SUB():
        return (lh, '-', rh,)
    elif rel.SET_INTERSECT():
        return (lh, '*', rh)

def convert_expr(expr):
    if expr.additive():
        return convert_add(expr.additive())
    elif expr.set_notation_sub_expr():
        return convert_set_notation_expr(expr.set_notation_sub_expr())
    elif expr.interval_expr():
        return convert_interval_expr(expr.interval_expr())

def convert_interval_expr(int_expr):
    if int_expr.interval():
        return handle_interval(int_expr.interval())

    int_expr1 = convert_interval_expr(int_expr.interval_expr(0))
    if int_expr.struct_value():
        int_expr2 = convert_struct_value(int_expr.struct_value())
        int_expr2 = sympy.FiniteSet(*int_expr2.elements)
    elif int_expr.atom():
        int_expr2 = convert_atom(int_expr.atom())
        int_expr2 = sympy.FiniteSet(int_expr2)
    else:
        int_expr2 = convert_interval_expr(int_expr.interval_expr(1))

    common_symbol = None
    try:
        for symbol in sympy.Intersection(int_expr1.boundary, int_expr2.boundary):
            if isinstance(symbol, sympy.Symbol):
                common_symbol = symbol
                break
    except Exception:
        pass

    if int_expr.SET_ADD():
        if common_symbol:
            union = int_expr1.subs(common_symbol, 0).union(int_expr2.subs(common_symbol, 0))
            if not isinstance(union, sympy.Union):
                return union
        return sympy.Union(int_expr1, int_expr2)
    if int_expr.SET_SUB():
        return sympy.Complement(int_expr1, int_expr2)
    if int_expr.SET_INTERSECT():
        return sympy.Intersection(int_expr1, int_expr2)

def convert_add(add):
    if add.ADD():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        if isinstance(lh, Vector):
            if not isinstance(rh, Vector):
                raise LaTeXParsingError("Latex2SympyError")
            return lh + rh
        else:
            # Matrix size validation
            if isinstance(lh, sympy.Matrix) and isinstance(rh, sympy.Matrix):
                if lh.shape != rh.shape:
                    raise LaTeXParsingError('Latex2SympyError')
                return lh + rh

            # NOTE: if either is zero, do not return "0+x" but "x"
            # Corresponding test case: https://github.com/snapwiz/math-engine/blob/d822a2a38480971ce2041f8af12453e6808563f1/tests.csv#L13527
            # if lh == 0: return rh
            # if rh == 0: return lh
            return sympy.Add(lh, rh, evaluate=False)
    elif add.SUB():
        lh = convert_add(add.additive(0))
        rh = convert_add(add.additive(1))
        if isinstance(lh, Vector):
            if not isinstance(rh, Vector):
                raise LaTeXParsingError('Latex2SympyError')
            return lh - rh
        # Matrix size validation
        if isinstance(lh, sympy.Matrix) and isinstance(rh, sympy.Matrix):
            if lh.shape != rh.shape:
                raise LaTeXParsingError('Latex2SympyError')
            return lh - rh
        # NOTE: evaluate if rh is integer/float so that "a-2" does not become "a-1*2"
        # SYMPY_1.3.0-UPGRADE_FIX
        evaluate = ( isinstance(rh, (sympy.Integer, sympy.Float)) and rh >=0 ) or isinstance(rh, sympy.Symbol)
        # evaluate = False
        # if hasattr(rh, "is_Atom") and rh.is_Atom:
        #     return sympy.Add(lh, -1 * rh, evaluate=False)
        return sympy.Add(lh, sympy.Mul(-1, rh, evaluate=evaluate), evaluate=False)
    else:
        return convert_mp(add.mp())


def convert_mp(mp):
    if hasattr(mp, 'mp'):
        mp_left = mp.mp(0)
        mp_right = mp.mp(1)
    else:
        mp_left = mp.mp_nofunc(0)
        mp_right = mp.mp_nofunc(1)

    if mp.MUL() or mp.CMD_TIMES() or mp.CMD_CDOT():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        if isinstance(lh, sympy.Matrix) and isinstance(rh, sympy.Matrix):
            if lh.shape[1] != rh.shape[0]:
                raise LaTeXParsingError('Latex2SympyError')
            return lh * rh
        return sympy.Mul(lh, rh, evaluate=False)
    elif mp.DIV() or mp.CMD_DIV() or mp.COLON():
        lh = convert_mp(mp_left)
        rh = convert_mp(mp_right)
        if rh == 0:
            raise LaTeXParsingError('ZeroDivisionError: division by zero')
        return sympy.Mul(lh, sympy.Pow(rh, -1, evaluate=False), evaluate=False)
    else:
        if hasattr(mp, 'unary'):
            return convert_unary(mp.unary())
        else:
            return convert_unary(mp.unary_nofunc())


def convert_unary(unary):
    if hasattr(unary, 'unary'):
        nested_unary = unary.unary()
    else:
        nested_unary = unary.unary_nofunc()
    if hasattr(unary, 'postfix_nofunc'):
        first = unary.postfix()
        tail = unary.postfix_nofunc()
        postfix = [first] + tail
    else:
        postfix = unary.postfix()

    if unary.ADD():
        return convert_unary(nested_unary)
    elif unary.SUB():
        numabs = convert_unary(nested_unary)

        # NOTE: evaluate is True to avoid "-1" be treated as "-1*1"
        # fixes: 4309 EV-18331-33,,"−10a−12b+18c−16","−10a−12b+18c−16",symbolic:isSimplified,true
        # breaks: 15681	EV-34659-13
        # SYMPY_1.3.0-UPGRADE_FIX
        evaluate = ( isinstance(numabs, (sympy.Integer, sympy.Float)) and numabs >=0 ) or isinstance(numabs, sympy.Symbol)
        # evaluate = False # NOTE: temporary to try 15681
        return sympy.Mul(-1, numabs, evaluate=evaluate)

        # Use Integer(-n) instead of Mul(-1, n)
        # return -numabs
    elif postfix:
        return convert_postfix_list(postfix)


def convert_postfix_list(arr, i=0):
    if i >= len(arr):
        raise LaTeXParsingError("Index out of bounds")

    res = convert_postfix(arr[i])

    is_commutative = LaTeXParsingContext.getOption('is_commutative')
    if (
        is_commutative is True
        and hasattr(res, "name")
        and res.name
        in [
            "overrightarrow",
            "vec",
            "square",
            "parallelogram",
        ]
    ):
        is_commutative = False
    with LaTeXParsingContext(is_commutative=is_commutative):
        if isinstance(res, (sympy.Expr, sympy.Eq, sympy.Matrix)):
            if i == len(arr) - 1:
                return res  # nothing to multiply by
            else:
                if i > 0:
                    # todo: check for is_coomutative
                    left = convert_postfix(arr[i - 1])
                    right = convert_postfix(arr[i + 1])
                    if isinstance(left, (sympy.Expr, sympy.Eq)) and isinstance(
                        right, (sympy.Expr, sympy.Eq)
                    ):
                        left_syms = convert_postfix(arr[i - 1]).atoms(sympy.Symbol)
                        right_syms = convert_postfix(arr[i + 1]).atoms(sympy.Symbol)
                        # if the left and right sides contain no variables and the
                        # symbol in between is 'x', treat as multiplication.
                        if not (left_syms or right_syms) and str(res) == "x":
                            return convert_postfix_list(arr, i + 1)
                rh = convert_postfix_list(arr, i + 1)
                # NOTE: This logic is added to fix unit tests broken during sympy-1.3.0 to sympy-1.13.3 upgrade
                # Example: unit test line 8827 is broken as "1"
                # However, it maybe wrong
                # SYMPY_1.3.0-UPGRADE_FIX
                if res == 1:
                    return rh
                return sympy.Mul(res, rh, evaluate=False)
        else:  # must be derivative
            wrt = res[0]
            if i == len(arr) - 1:
                return res
            else:
                expr = convert_postfix_list(arr, i + 1)
                result = sympy.Derivative(expr, wrt)
                if LaTeXParsingContext.getOption('evaluate_derivative'):
                    return result.doit()
                return result


def do_subs(expr, at):
    if at.expr():
        at_expr = convert_expr(at.expr())
        syms = at_expr.atoms(sympy.Symbol)
        if len(syms) == 0:
            return expr
        elif len(syms) > 0:
            sym = next(iter(syms))
            return expr.subs(sym, at_expr)
    elif at.equality():
        lh = convert_expr(at.equality().expr(0))
        rh = convert_expr(at.equality().expr(1))
        return expr.subs(lh, rh)


def convert_postfix(postfix):
    if hasattr(postfix, 'exp'):
        exp_nested = postfix.exp()
    else:
        exp_nested = postfix.exp_nofunc()

    exp = convert_exp(exp_nested)
    for op in postfix.postfix_op():
        if op.BANG():
            if isinstance(exp, list):
                raise LaTeXParsingError("Cannot apply postfix to derivative")
            exp = sympy.factorial(exp, evaluate=False)
        elif op.eval_at():
            ev = op.eval_at()
            at_b = None
            at_a = None
            if ev.eval_at_sup():
                at_b = do_subs(exp, ev.eval_at_sup())
            if ev.eval_at_sub():
                at_a = do_subs(exp, ev.eval_at_sub())
            if at_b is not None and at_a is not None:
                exp = sympy.Add(at_b, -1 * at_a, evaluate=False)
            elif at_b is not None:
                exp = at_b
            elif at_a is not None:
                exp = at_a

    return exp

def convert_exp(exp):
    if hasattr(exp, 'exp'):
        exp_nested = exp.exp()
    else:
        exp_nested = exp.exp_nofunc()

    if exp_nested:
        base = convert_exp(exp_nested)
        if isinstance(base, list):
            raise LaTeXParsingError("Cannot raise derivative to power")
        if exp.atom():
            exponent = convert_atom(exp.atom())
        elif exp.expr():
            exponent = convert_expr(exp.expr())
        # FIXME: exponent can be unbound
        return sympy.Pow(base, exponent, evaluate=False)
    else:
        if hasattr(exp, 'comp'):
            return convert_comp(exp.comp())
        else:
            return convert_comp(exp.comp_nofunc())

def convert_comp(comp):
    if comp.group():
        return convert_expr(comp.group().expr())
    elif comp.abs_group():
        return sympy.Abs(convert_expr(comp.abs_group().expr()), evaluate=False)
    elif comp.atom():
        return convert_atom(comp.atom())
    elif hasattr(comp, 'frac') and comp.frac():  # todo:
        return convert_frac(comp.frac())
    elif comp.floor():
        return convert_floor(comp.floor())
    elif comp.ceil():
        return convert_ceil(comp.ceil())
    elif comp.func():
        return convert_func(comp.func())
    elif comp.vector():
        return convert_vector(comp.vector())

#todo: revisit
def convert_atom(atom):
    if atom.LETTER():
        sname = atom.LETTER().getText()
        if atom.subexpr():
            if atom.subexpr().expr():  # subscript is expr
                subscript = convert_expr(atom.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(atom.subexpr().atom())
            sname += '_{' + StrPrinter().doprint(subscript) + '}'
        if atom.SINGLE_QUOTES():
            sname += atom.SINGLE_QUOTES().getText()  # put after subscript for easy identify
        if sname == 'E':
            return sympy.E
        if sname == 'I':
            return sympy.I
        if LaTeXParsingContext.getOption('is_commutative') is False:
            return sympy.Symbol(sname, commutative=False)
        return sympy.Symbol(sname, real=LaTeXParsingContext.getOption('real_symbol'))
    elif atom.SYMBOL():
        s = atom.SYMBOL().getText()[1:]
        if s == "infty":
            return sympy.oo
        elif s == '%':
            return sympy.Rational(1, 100)
        elif s == 'pi':
            return sympy.pi
        else:
            if atom.subexpr():
                subscript = None
                if atom.subexpr().expr():  # subscript is expr
                    subscript = convert_expr(atom.subexpr().expr())
                else:  # subscript is atom
                    subscript = convert_atom(atom.subexpr().atom())
                subscriptName = StrPrinter().doprint(subscript)
                s += '_{' + subscriptName + '}'
            if LaTeXParsingContext.getOption('is_commutative') is False:
                return sympy.Symbol(s, commutative=False)
            return sympy.Symbol(s, real=LaTeXParsingContext.getOption('real_symbol'))
    elif atom.NUMBER():
        s = atom.NUMBER().getText()
        if (number_error_re.search(s)):
            # TODO: this regex can be possibly moved to the grammar
            raise LaTeXParsingError('NumberError: invalid number')
        s = s.replace(",", "")
        return sympy.Number(s)
    elif atom.DIFFERENTIAL():
        var = get_differential_var(atom.DIFFERENTIAL())
        return sympy.Symbol('d' + var.name, real=LaTeXParsingContext.getOption('real_symbol'))
    elif atom.mathit():
        text = rule2text(atom.mathit().mathit_text())
        return sympy.Symbol(text, real=LaTeXParsingContext.getOption('real_symbol'))
    elif atom.angle():
        text = rule2text(atom.angle().angle_points())
        points = text.split()
        if len(points) == 3:
            if points[2] < points[0]:
                text = points[2] + ' ' + points[1] + ' ' + points[0]
            else:
                text = points[0] + ' ' + points[1] + ' ' + points[2]
        text = 'angle' + text # mark symbol as angle
        return sympy.Symbol(text, real=LaTeXParsingContext.getOption('real_symbol'))
    elif atom.frac():
        return convert_frac(atom.frac())
    elif atom.binom():
        return convert_binom(atom.binom())
    elif atom.bra():
        val = convert_expr(atom.bra().expr())
        return Bra(val)
    elif atom.ket():
        val = convert_expr(atom.ket().expr())
        return Ket(val)


def rule2text(ctx):
    stream = ctx.start.getInputStream()
    # starting index of starting token
    startIdx = ctx.start.start
    # stopping index of stopping token
    stopIdx = ctx.stop.stop

    return stream.getText(startIdx, stopIdx)

# todo: revisit
def convert_frac(frac):
    diff_op = False
    partial_op = False
    if frac.lower and frac.upper:
        lower_itv = frac.lower.getSourceInterval()
        lower_itv_len = lower_itv[1] - lower_itv[0] + 1
        if (frac.lower.start == frac.lower.stop
                and frac.lower.start.type == LaTeXLexer.DIFFERENTIAL):
            wrt = get_differential_var_str(frac.lower.start.text)
            deg = 1
            diff_op = True
        elif (lower_itv_len == 2 and frac.lower.start.type == LaTeXLexer.SYMBOL
              and frac.lower.start.text == '\\partial'
              and (frac.lower.stop.type == LaTeXLexer.LETTER
                   or frac.lower.stop.type == LaTeXLexer.SYMBOL)):
            partial_op = True
            wrt = frac.lower.stop.text
            if frac.lower.stop.type == LaTeXLexer.SYMBOL:
                wrt = wrt[1:]
        elif frac.lower.start.type == LaTeXLexer.DIFFERENTIAL:
            wrt = get_differential_var_str(frac.lower.start.text)
            deg = int(re.match(r'd[\w]\^\{?(\d+)\}?', frac.lower.getText()).group(1))
            diff_op = True

        if diff_op or partial_op:
            wrt = sympy.Symbol(wrt, real=LaTeXParsingContext.getOption('real_symbol'))
            if (diff_op and frac.upper.start == frac.upper.stop
                    and frac.upper.start.type == LaTeXLexer.LETTER
                    and frac.upper.start.text == 'd'):
                return [wrt]
            elif (partial_op and frac.upper.start == frac.upper.stop
                  and frac.upper.start.type == LaTeXLexer.SYMBOL
                  and frac.upper.start.text == '\\partial'):
                return [wrt]
            upper_text = rule2text(frac.upper)

            expr_top = None
            if diff_op and upper_text.startswith('d'):
                # TODO: these regex matches can be possibly moved to grammar
                upper_text = re.sub(r'd(\^([\d]+|\{[\d]+\}))?', '', upper_text)
                upper_text = re.sub(r'([\w])(\*?\(x\))', r'\1', upper_text)
                expr_top = parse_latex(re.sub(r'd(\^[\d]+)?', '', upper_text))
            elif partial_op and frac.upper.start.text == '\\partial':
                expr_top = parse_latex(upper_text[len('\\partial'):])

            if expr_top:
                result = expr_top
                for i in range(deg):
                    result = sympy.Derivative(result, wrt)
                    if LaTeXParsingContext.getOption('evaluate_derivative'):
                        result = result.doit()
                return result
    if frac.upper:
        expr_top = convert_expr(frac.upper)
    else:
        expr_top = sympy.Number(frac.upperd.text)
    if frac.lower:
        expr_bot = convert_expr(frac.lower)
    else:
        expr_bot = sympy.Number(frac.lowerd.text)
    if expr_bot == 0:
        raise LaTeXParsingError('ZeroDivisionError: division by zero')
    inverse_denom = sympy.Pow(expr_bot, -1, evaluate=False)
    if expr_top == 1:
        return inverse_denom
    else:
        return sympy.Mul(expr_top, inverse_denom, evaluate=False)

def convert_binom(binom):
    expr_n = convert_expr(binom.n)
    expr_k = convert_expr(binom.k)
    return sympy.binomial(expr_n, expr_k, evaluate=False)

def convert_floor(floor):
    val = convert_expr(floor.val)
    return sympy.floor(val, evaluate=False)

def convert_ceil(ceil):
    val = convert_expr(ceil.val)
    return sympy.ceiling(val, evaluate=False)

def convert_func(func):
    if func.func_normal():
        if func.L_PAREN():  # function called with parenthesis
            arg = convert_func_arg(func.func_arg())[0]
        else:
            arg = convert_func_arg(func.func_arg_noparens())[0]

        name = func.func_normal().start.text[1:]

        # change arc<trig> -> a<trig>
        if name in [
                "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot"
        ]:
            name = "a" + name[3:]
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if name == "exp":
            expr = sympy.exp(arg, evaluate=False)

        if name in ("log", "lg", "ln"):
            if func.subexpr():
                if func.subexpr().expr():
                    base = convert_expr(func.subexpr().expr())
                else:
                    base = convert_atom(func.subexpr().atom())
            elif name == "lg":  # ISO 80000-2:2019
                base = 10
            elif name == "log":
                base = 10 if LaTeXParsingContext.getOption('log_base_10') else sympy.E
            if name == "ln":
                expr = sympy.log(arg, evaluate=False)
            else:
                expr = sympy.log(arg, base, evaluate=False)

        if (name=="exp"):
            expr = getattr(sympy.functions, name)(arg, evaluate=False)
        func_pow = None
        should_pow = True
        if func.supexpr():
            if func.supexpr().expr():
                func_pow = convert_expr(func.supexpr().expr())
            else:
                func_pow = convert_atom(func.supexpr().atom())

        if name in [
            "sin", "cos", "tan", "csc", "sec", "cot",
            "sinh", "cosh", "tanh", "coth", "csch", "sech",
            "arcsinh", "arccosh", "arctanh", "arccoth", "arccsch", "arcsech"
        ]:
            if func_pow == -1:
                name = "a" + name
                should_pow = False
            name = name.replace("arc", "a")
            if name in ["asin", "acos"] and arg.is_number and (arg < -1 or 1 < arg):
                raise LaTeXParsingError(f'TrigError: invalid argument "{arg}" for {name}')
            if name in ["acsc", "asec"] and arg.is_number and (-1 < arg and arg < 1):
                raise LaTeXParsingError(f'TrigError: invalid argument "{arg}" for {name}')
            if name in ["acosh"] and arg.is_number and arg < 1:
                raise LaTeXParsingError(f'TrigError: invalid argument "{arg}" for {name}')
            expr = getattr(sympy.functions, name)(arg, evaluate=False)

        if func_pow and should_pow:
            expr = sympy.Pow(expr, func_pow, evaluate=False)

        if name == "abs":
            expr = sympy.Abs(arg, evaluate=False)

        if name== 'Re':
            expr = sympy.re(arg, evaluate=False)

        if name == 'Im':
            expr = sympy.im(arg, evaluate=False)

        if name == 'arg':
            expr = sympy.arg(arg, evaluate=False)

        if name == 'overline':
            expr = arg

        return expr
    elif func.LETTER() or func.SYMBOL():
        if func.LETTER():
            fname = func.LETTER().getText()
        elif func.SYMBOL():
            fname = func.SYMBOL().getText()[1:]
        fname = str(fname)  # can't be unicode
        if func.subexpr():
            subscript = None
            if func.subexpr().expr():  # subscript is expr
                subscript = convert_expr(func.subexpr().expr())
            else:  # subscript is atom
                subscript = convert_atom(func.subexpr().atom())
            subscriptName = StrPrinter().doprint(subscript)
            fname += '_{' + subscriptName + '}'
        if func.SINGLE_QUOTES():
            fname += func.SINGLE_QUOTES().getText()
        input_args = func.args()
        output_args = []
        while input_args.args():  # handle multiple arguments to function
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))
        return sympy.Function(fname)(*output_args)
    elif func.func_composition():
        fnames = convert_func_composition(func.func_composition())
        input_args = func.args()
        output_args = []
        while input_args.args():                        # handle multiple arguments to function
            output_args.append(convert_expr(input_args.expr()))
            input_args = input_args.args()
        output_args.append(convert_expr(input_args.expr()))

        fnames.reverse()
        func_composition = sympy.Function(fnames[0])(*output_args)
        for fname in fnames[1:]:
            func_composition = sympy.Function(fname)(func_composition)
        return func_composition
    elif func.FUNC_INT():
        return handle_integral(func)
    elif func.FUNC_IINT():
        return handle_iintegral(func)
    elif func.FUNC_OINT():
        return handle_ointegral(func)
    elif func.FUNC_SQRT():
        expr = convert_expr(func.base)
        if func.root:
            r = convert_expr(func.root)
            return sympy.root(expr, r, evaluate=False)
        else:
            return sympy.sqrt(expr, evaluate=False)
    elif func.FUNC_OVERLINE():
        expr = convert_expr(func.base)
        return sympy.conjugate(expr, evaluate=False)
    elif func.FUNC_SUM():
        return handle_sum_or_prod(func, "summation")
    elif func.FUNC_PROD():
        return handle_sum_or_prod(func, "product")
    elif func.FUNC_LIM():
        return handle_limit(func)
    elif func.FUNC_MATRIX_START():
        return handle_matrix(func)
    elif func.FUNC_MATRIX_DETERMINENT_START():
        return handle_matrix_determinent(func)
    elif func.FUNC_AL_MATRIX_PIECEWISE_START():
        return handle_matrix_piecewise(func)
    elif func.FUNC_AR_MATRIX_PIECEWISE_START():
        return handle_matrix_piecewise(func)
    elif func.FUNC_PIECEWISE_START():
        return handle_piecewise(func)
    elif func.FUNC_CALCULATION_START():
        return handle_calculation(func)

def convert_func_arg(arg):
    args = []
    if hasattr(arg, 'expr'):
        args.append(convert_expr(arg.expr()))
    else:
        args.append(convert_mp(arg.mp_nofunc()))
    if hasattr(arg, 'func_arg') and arg.func_arg():
        args += convert_func_arg(arg.func_arg())
    return args

def convert_func_composition(func_composition,):
    fnames = []
    fnames.append(convert_func_name(func_composition.func_name()[0]))

    if func_composition.func_composition():
        fnames = fnames + convert_func_composition(func_composition.func_composition())
    else:
        fnames.append(convert_func_name(func_composition.func_name()[1]))

    return fnames

def convert_func_name(func_name):
    if func_name.LETTER():
        fname = func_name.LETTER().getText()
    elif func_name.SYMBOL():
        fname = func_name.SYMBOL().getText()[1:]
    fname = str(fname) # can't be unicode
    return fname

def handle_matrix(matrix):
    matrix = matrix.matrix()
    m_list = []
    for row in matrix.matrix_row():
        m_list.append(convert_matrix_row(row))
    return sympy.Matrix(m_list)

def handle_matrix_determinent(matrix):
    matrix = matrix.matrix()
    m_list = []
    for row in matrix.matrix_row():
        m_list.append(convert_matrix_row(row))
    try:
        return sympy.Matrix(m_list).det()
    except Exception as e:
        raise LaTeXParsingError('Error handling matrix determinent') from e


def handle_piecewise_func(piecewise_func):
    try:
        expr = convert_expr(piecewise_func.expr())
        try:
            if piecewise_func.OTHERWISE():
                return expr, 'otherwise'
        except Exception:
            pass
        if piecewise_func.relation() is None:
            raise LaTeXParsingError('PiecewiseFunctionError: invalid relation expression')
        rel = convert_relation(piecewise_func.relation())

        if isinstance(rel.lhs, sympy.core.relational.Relational):
            l_rel = rel.lhs
            r_rel = type(rel)(rel.lhs.rhs,rel.rhs)
            rel = sympy.And(l_rel, r_rel)
        return expr, rel
    except LaTeXParsingError:
        raise
    except Exception as e:
        raise LaTeXParsingError('Error handling piecewise function') from e


def handle_matrix_piecewise(matrix_piecewise):
    if matrix_piecewise.matrix_relation():
        return handle_matrix_relation(matrix_piecewise)

    matrix_piecewise = matrix_piecewise.matrix_piecewise()
    try:
        piecewise_items = []

        otherwise_expr = None
        otherwise = sympy.S.Reals

        for matrix_piecewise_func in matrix_piecewise.matrix_piecewise_func():
            # parse expr and cond
            expr, cond = handle_piecewise_func(matrix_piecewise_func)

            if cond == 'otherwise':
                if otherwise_expr is not None:
                    raise LaTeXParsingError('PiecewiseFunctionError: invalid otherwise expression')
                otherwise_expr = expr
                continue

            try:
                otherwise = cond.as_set().symmetric_difference(otherwise)
            except Exception:
                pass

            piecewise_items.append((expr, cond))

        if otherwise_expr is not None:
            piecewise_items.append((otherwise_expr, otherwise))
        # return sympy.Piecewise(*piecewise_items)
        return piecewise_items
    except LaTeXParsingError:
        raise
    except Exception as e:
        raise LaTeXParsingError('Error handling matrix piecewise') from e


def handle_matrix_relation(matrix_relation):
    matrix_relation = matrix_relation.matrix_relation()
    try:
        relations = []

        for relation in matrix_relation.relation():
            relations.append(convert_relation(relation))

        return relations
    except LaTeXParsingError:
        raise
    except Exception as e:
        raise LaTeXParsingError("Error handling Matrix relation") from e


def handle_piecewise(piecewise):
    piecewise = piecewise.piecewise()
    try:
        piecewise_items = []

        for piecewise_func in piecewise.piecewise_func():
            # parse expr and cond
            expr, cond = handle_piecewise_func(piecewise_func)

            piecewise_items.append((expr, cond))

        # return sympy.Piecewise(*piecewise_items)
        return piecewise_items
    except LaTeXParsingError:
        raise
    except Exception as e:
        raise LaTeXParsingError("Error handling Piecewise expression") from e


def handle_calculation(calculation):
    calculation = calculation.calculation()

    try:
        if calculation.calculation_add():
            ops = calculation.calculation_add().NUMBER()
            l_op = sympy.Number(ops[0].getText())
            r_op = sympy.Number(ops[1].getText())
            res = sympy.Number(ops[2].getText())

            if sympy.Eq(res, sympy.Add(l_op, r_op)) == False:
                raise LaTeXParsingError('AdditionError: False equality')

            return sympy.Eq(res, sympy.Add(l_op, r_op, evaluate=False), evaluate=False)

        elif calculation.calculation_sub():
            ops = calculation.calculation_sub().NUMBER()
            l_op = sympy.Number(ops[0].getText())
            r_op = -1 * sympy.Number(ops[1].getText())
            res = sympy.Number(ops[2].getText())

            if sympy.Eq(res, sympy.Add(l_op, r_op)) == False:
                raise LaTeXParsingError('SubtractionError: False equality')

            return sympy.Eq(res, sympy.Add(l_op, r_op, evaluate=False), evaluate=False)

        elif calculation.calculation_mul():
            ops = calculation.calculation_mul().NUMBER()
            l_op = sympy.Number(ops[0].getText())
            r_op = sympy.Number(ops[1].getText())
            res = sympy.Number(ops[2].getText())

            if sympy.Eq(res, sympy.Mul(l_op, r_op)) == False:
                raise LaTeXParsingError('MultiplicationError: False equality')

            return sympy.Eq(res, sympy.Mul(l_op, r_op, evaluate=False), evaluate=False)

        elif calculation.calculation_div():
            ops = calculation.calculation_div().NUMBER()
            res = sympy.Number(ops[0].getText())
            r_op = sympy.Number(ops[1].getText())
            l_op = sympy.Number(ops[2].getText())

            return sympy.Eq(res, sympy.Mul(l_op, sympy.Pow(r_op, -1, evaluate=False), evaluate=False), evaluate=False)
    except LaTeXParsingError:
        raise
    except Exception as e:
        raise LaTeXParsingError('Unknown error in parsing calculation') from e


def convert_matrix_row(row):
    row_list = []
    for expr in row.expr():
        row_list.append(convert_expr(expr))
    return row_list


def convert_set_notation_expr(set_notation_expr):
    if set_notation_expr.set_notation_sub():
        return handle_set_notation(set_notation_expr.set_notation_sub())

    set_notation1 = convert_set_notation_expr(set_notation_expr.set_notation_sub_expr(0))
    set_notation2 = convert_set_notation_expr(set_notation_expr.set_notation_sub_expr(1))

    common_symbol = None
    try:
        for symbol in sympy.Intersection(set_notation1.boundary, set_notation2.boundary):
            if isinstance(symbol, sympy.Symbol):
                common_symbol = symbol
                break
    except Exception:
        pass

    if set_notation_expr.SET_ADD():
        if common_symbol:
            union = set_notation1.subs(common_symbol, 0).union(set_notation2.subs(common_symbol, 0))
            if not isinstance(union, sympy.Union):
                return union
        return sympy.Union(set_notation1, set_notation2)
    if set_notation_expr.SET_SUB():
        return sympy.Complement(set_notation1, set_notation2)
    if set_notation_expr.SET_INTERSECT():
        return sympy.Intersection(set_notation1, set_notation2)

def handle_set_notation(sub):
    if sub.LETTER():
        var = sympy.Symbol(sub.LETTER().getText(), real=LaTeXParsingContext.getOption('real_symbol'))
    elif sub.SYMBOL():
        var = sympy.Symbol(sub.SYMBOL().getText()[1:], real=LaTeXParsingContext.getOption('real_symbol'))
    else:
        var = sympy.Symbol('x', real=LaTeXParsingContext.getOption('real_symbol'))
    rel = convert_relation(sub.relation())
    sol = sympy.S.Reals
    while isinstance(rel,sympy.core.relational.Relational):
        future_rel = rel.lhs
        if isinstance(rel.lhs,sympy.core.relational.Relational):
            future_rel = rel.lhs
            rel = type(rel)(rel.lhs.rhs,rel.rhs)

        if not isinstance(rel,BooleanAtom):
            sol_tmp = sympy.solveset(rel,var,sympy.S.Reals)
            sol = sympy.Intersection(sol,sol_tmp)
        rel = future_rel
    return sol

def handle_interval(interval):
    left_bool = False
    right_bool = False

    left_expr = convert_expr(interval.expr(0)).simplify()
    right_expr = convert_expr(interval.expr(1)).simplify()

    if interval.interval_opr(0).L_PAREN() or interval.interval_opr(0).R_BRACKET():
        left_bool = True
        left_par = '('
    if interval.interval_opr(1).R_PAREN() or interval.interval_opr(1).L_BRACKET():
        right_bool = True
        right_par = ')'

    # For calculation, parse as point by default
    if (
        LaTeXParsingContext.getOption('prefer_point_over_interval') and
        left_bool and
        right_bool
    ):
        return sympy.Point(left_expr, right_expr)

    if (
        (interval.interval_opr(0).L_PAREN() and interval.interval_opr(1).L_BRACKET())
        or (interval.interval_opr(1).L_PAREN() and interval.interval_opr(0).L_BRACKET())
        or (interval.interval_opr(0).R_PAREN() and interval.interval_opr(1).R_BRACKET())
        or (interval.interval_opr(1).R_PAREN() and interval.interval_opr(0).R_BRACKET())
    ):
        raise LaTeXParsingError('IntervalError: invalid grouping')

    if (
        (left_expr == sympy.oo or left_expr == -sympy.oo) and not left_bool
    ) or (
        (right_expr == sympy.oo or right_expr == -sympy.oo) and not right_bool
    ):
        raise LaTeXParsingError('IntervalError: invalid use of infinity in interval')
    res = sympy.Interval(left_expr, right_expr, left_bool, right_bool)
    if res is sympy.EmptySet and left_expr != right_expr:
        # TODO: why should this be a error at parser level ?
        raise LaTeXParsingError('IntervalError: Set with different boundaries appears to be empty')
    return res

def handle_integral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = sympy.Symbol(s[2:], real=LaTeXParsingContext.getOption('real_symbol'))
                else:
                    int_var = sympy.Symbol(s[1:], real=LaTeXParsingContext.getOption('real_symbol'))
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            # Assume dx by default
            int_var = sympy.Symbol('x', real=LaTeXParsingContext.getOption('real_symbol'))

    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        if func.supexpr().atom():
            upper = convert_atom(func.supexpr().atom())
        else:
            upper = convert_expr(func.supexpr().expr())
        # NOTE: doing .doit() evaluates the integral.
        result = sympy.Integral(integrand, (int_var, lower, upper))
        if LaTeXParsingContext.getOption('evaluate_integral'):
            return result.doit()
        return result
    else:
        return sympy.Integral(integrand, int_var)

def handle_iintegral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_vars = None
    if func.MULTI_DIFFERENTIAL():
        int_vars = get_multi_differential_var(func.MULTI_DIFFERENTIAL())
    else:
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_vars.append(sympy.Symbol(s[2:], real=LaTeXParsingContext.getOption('real_symbol')))
                else:
                    int_vars.append(sympy.Symbol(s[1:], real=LaTeXParsingContext.getOption('real_symbol')))
                integrand = integrand.subs(sym, 1)
        if not int_vars:
            # Assume dx by default
            int_vars.append(sympy.Symbol('x', real=LaTeXParsingContext.getOption('real_symbol')))

    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        result = sympy.Integral(integrand, *int_vars, lower)
        if LaTeXParsingContext.getOption('evaluate_integral'):
            return result.doit()
        return result
    else:
        return sympy.Integral(integrand, *int_vars)

def handle_ointegral(func):
    if func.additive():
        integrand = convert_add(func.additive())
    elif func.frac():
        integrand = convert_frac(func.frac())
    else:
        integrand = 1

    int_var = None
    if func.DIFFERENTIAL():
        int_var = get_differential_var(func.DIFFERENTIAL())
    else:
        for sym in integrand.atoms(sympy.Symbol):
            s = str(sym)
            if len(s) > 1 and s[0] == 'd':
                if s[1] == '\\':
                    int_var = sympy.Symbol(s[2:], real=LaTeXParsingContext.getOption('real_symbol'))
                else:
                    int_var = sympy.Symbol(s[1:], real=LaTeXParsingContext.getOption('real_symbol'))
                int_sym = sym
        if int_var:
            integrand = integrand.subs(int_sym, 1)
        else:
            # Assume dx by default
            int_var = sympy.Symbol('x', real=LaTeXParsingContext.getOption('real_symbol'))

    if func.subexpr():
        if func.subexpr().atom():
            lower = convert_atom(func.subexpr().atom())
        else:
            lower = convert_expr(func.subexpr().expr())
        result = sympy.Integral(integrand, (int_var, lower))
        if LaTeXParsingContext.getOption('evaluate_integral'):
            return result.doit()
        return result
    else:
        return sympy.Integral(integrand, int_var)

def handle_sum_or_prod(func, name):
    val = convert_mp(func.mp())
    if func.subeq():
        iter_var = convert_expr(func.subeq().equality().expr(0))
        start = convert_expr(func.subeq().equality().expr(1))
    else:
        for atom in val.atoms():
            if isinstance(atom, sympy.Symbol):
                iter_var = atom
                break
        else:
            iter_var = n

        if func.subexpr().expr():
            start = convert_expr(func.subexpr().expr())
        else:
            start = convert_atom(func.subexpr().atom())

    if func.supexpr().expr():  # ^{expr}
        end = convert_expr(func.supexpr().expr())
    else:  # ^atom
        end = convert_atom(func.supexpr().atom())

    if name == "summation":
        result = sympy.Sum(val, (iter_var, start, end))
        if LaTeXParsingContext.getOption('evaluate_summation_op'):
            return result.doit()
        return result
    elif name == "product":
        result = sympy.Product(val, (iter_var, start, end))
        if LaTeXParsingContext.getOption('evaluate_product_op'):
            return result.doit()
        return result

def handle_limit(func):
    sub = func.limit_sub()
    if sub.LETTER():
        var = sympy.Symbol(sub.LETTER().getText(), real=LaTeXParsingContext.getOption('real_symbol'))
    elif sub.SYMBOL():
        var = sympy.Symbol(sub.SYMBOL().getText()[1:], real=LaTeXParsingContext.getOption('real_symbol'))
    else:
        var = sympy.Symbol('x', real=LaTeXParsingContext.getOption('real_symbol'))
    if sub.SUB():
        direction = "-"
    elif sub.ADD():
        direction = "+"
    else:
        direction = "+-"
    approaching = convert_expr(sub.expr())
    content = convert_mp(func.mp())

    result = sympy.Limit(content, var, approaching, direction)
    if LaTeXParsingContext.getOption('evaluate_limit'):
        return result.doit()
    return result


def get_differential_var(d):
    text = get_differential_var_str(d.getText())
    return sympy.Symbol(text, real=LaTeXParsingContext.getOption('real_symbol'))

def get_multi_differential_var(d):
    diffs = multi_differential_re.findall(d.getText())
    int_vars = []
    for diff in diffs:
        int_vars.append(sympy.Symbol(get_differential_var_str(diff)))
    return int_vars

def get_differential_var_str(text):
    for i in range(1, len(text)):
        c = text[i]
        if not (c == " " or c == "\r" or c == "\n" or c == "\t"):
            idx = i
            break
    text = text[idx:]
    if text[0] == "\\":
        text = text[1:]
    return text


def convert_vector(vector):
    args = convert_func_arg(vector.func_arg())
    C = CoordSys3D('C')
    v = matrix_to_vector(Matrix(args), C)
    return v
