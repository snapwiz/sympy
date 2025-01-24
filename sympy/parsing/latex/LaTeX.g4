/*
 ANTLR4 LaTeX Math Grammar

 Ported from latex2sympy by @augustt198 https://github.com/augustt198/latex2sympy See license in
 LICENSE.txt
 */

/*
 After changing this file, it is necessary to run `python setup.py antlr` in the root directory of
 the repository. This will regenerate the code in `sympy/parsing/latex/_antlr/*.py`.
 */

grammar LaTeX;

options {
	language = Python3;
}

EWS: '\\ ' -> skip;
WS: [ \t\r\n]+ -> skip;
THINSPACE: ('\\,' | '\\thinspace') -> skip;
MEDSPACE: ('\\:' | '\\medspace') -> skip;
THICKSPACE: ('\\;' | '\\thickspace') -> skip;
QUAD: '\\quad' -> skip;
QQUAD: '\\qquad' -> skip;
NEGTHINSPACE: ('\\!' | '\\negthinspace') -> skip;
NEGMEDSPACE: '\\negmedspace' -> skip;
NEGTHICKSPACE: '\\negthickspace' -> skip;
CMD_LEFT: '\\left' -> skip;
CMD_RIGHT: '\\right' -> skip;

IGNORE:
	(
		'\\vrule'
		| '\\vcenter'
		| '\\vbox'
		| '\\vskip'
		| '\\vspace'
		| '\\hfil'
		| '\\*'
		| '\\-'
		| '\\.'
		| '\\/'
		| '\\"'
		| '\\('
		| '\\='
	) -> skip;

ADD: '+';
SUB: ( '-' | '\u2212' );
MUL: '*';
DIV: '/';

SET_ADD: '\\cup';
SET_SUB: '\\backslash' | '\\setminus';
SET_INTERSECT: '\\cap';

L_PAREN: '\\left'? '(';
R_PAREN: '\\right'? ')';
L_BRACE: '\\left'? '{';
R_BRACE: '\\right'? '}';
L_BRACE_LITERAL: '\\{';
R_BRACE_LITERAL: '\\}';
L_BRACKET: '\\left'? '[';
R_BRACKET: '\\right'? ']';

// TODO: These were found to be not used anywhere in old math-engine code
// LEFT_PARENTHESES: L_PAREN | L_BRACE | L_BRACKET;
// RIGHT_PARENTHESES: R_PAREN | R_BRACE | R_BRACKET;

BAR: '\\left'? '|' | '\\right'? '|';

HLINE: '\\hline';

OVERLINE: '\\overline';

SMASH_BIG: '\\smash{\\big)}';

R_BAR: '\\right|';
L_BAR: '\\left|';

L_ANGLE: '\\langle';
R_ANGLE: '\\rangle';
FUNC_LIM: '\\lim';
LIM_APPROACH_SYM:
	'\\to'
	| '\\rightarrow'
	| '\\Rightarrow'
	| '\\longrightarrow'
	| '\\Longrightarrow';
FUNC_INT:
    '\\int'
    | '\\int\\limits';
FUNC_IINT: '\\iint';
FUNC_OINT: '\\oint';
FUNC_SUM: '\\sum';
FUNC_PROD: '\\prod';

FUNC_EXP: '\\exp';
FUNC_LOG: '\\log';
FUNC_LG: '\\lg';
FUNC_LN: '\\ln';
FUNC_SIN: '\\sin';
FUNC_COS: '\\cos';
FUNC_TAN: '\\tan';
FUNC_CSC: '\\csc';
FUNC_SEC: '\\sec';
FUNC_COT: '\\cot';

FUNC_ARCSIN: '\\arcsin';
FUNC_ARCCOS: '\\arccos';
FUNC_ARCTAN: '\\arctan';
FUNC_ARCCSC: '\\arccsc';
FUNC_ARCSEC: '\\arcsec';
FUNC_ARCCOT: '\\arccot';

FUNC_SINH: '\\sinh';
FUNC_SECH: '\\sech';
FUNC_COSH: '\\cosh';
FUNC_CSCH: '\\csch';
FUNC_TANH: '\\tanh';
FUNC_COTH: '\\coth';
FUNC_ARCSINH: '\\arcsinh';
FUNC_ARCSECH: '\\arcsech';
FUNC_ARCCOSH: '\\arccosh';
FUNC_ARCCSCH: '\\arccsch';
FUNC_ARCTANH: '\\arctanh';
FUNC_ARCCOTH: '\\arccoth';
FUNC_MATRIX_START: '\\begin{bmatrix}';
FUNC_MATRIX_END: '\\end{bmatrix}';
FUNC_MATRIX_DETERMINENT_START: '\\begin{vmatrix}';
FUNC_MATRIX_DETERMINENT_END: '\\end{vmatrix}';
FUNC_AL_MATRIX_PIECEWISE_START: '\\begin{almatrix}';
FUNC_AL_MATRIX_PIECEWISE_END: '\\end{almatrix}';
FUNC_AR_MATRIX_PIECEWISE_START: '\\begin{armatrix}';
FUNC_AR_MATRIX_PIECEWISE_END: '\\end{armatrix}';
FUNC_PIECEWISE_START: '\\begin{array}{lc}';
FUNC_ARRAY_END: '\\end{array}';
FUNC_CALCULATION_START: '\\begin{array}{r}';

L_FLOOR: '\\lfloor';
R_FLOOR: '\\rfloor';
L_CEIL: '\\lceil';
R_CEIL: '\\rceil';

FUNC_SQRT: '\\sqrt';
FUNC_ABS: '\\abs';
FUNC_RE: '\\Re';
FUNC_IM: '\\Im';
FUNC_ARG: '\\arg';
// TODO: FUNC_OVERLINE was not used in code
// FUNC_OVERLINE: '\\overline';

CMD_TIMES: '\\times';
CMD_CDOT: '\\cdot';
CMD_DIV: '\\div';
CMD_FRAC:
    '\\frac'
    | '\\dfrac'
    | '\\cfrac'
    | '\\tfrac';
CMD_BINOM: '\\binom';
CMD_DBINOM: '\\dbinom';
CMD_TBINOM: '\\tbinom';

CMD_MATHIT: '\\mathit';
CMD_ANGLE: '\\angle';

CMD_CIRCLE: '\\circ';

UNDERSCORE: '_';
CARET: '^';
COLON: ':';
SEMI_COLON: ';';
AMP: '&';

fragment WS_CHAR: [ \t\r\n];
DIFFERENTIAL: 'd' WS_CHAR*? ([a-zA-Z] | '\\' [a-zA-Z]+);
MULTI_DIFFERENTIAL: DIFFERENTIAL+;

LETTER: [a-zA-Z];
DIGIT: [0-9];

EQUAL: (('&' WS_CHAR*?)? '=') | ('=' (WS_CHAR*? '&')?) | '=' | '\\eq';
NEQ: '\\neq';
LT: '<' | '\\lt';
LTE: ('<=' | '\\leq' | '\\le' | LTE_Q | LTE_S);
LTE_Q: '\\leqq';
LTE_S: '\\leqslant';
GT: '>' | '\\gt';
GTE: '>=' | '\\geq' | '\\ge' | GTE_Q | GTE_S;
EQUIV: '\\equiv';
OTHERWISE: 'otherwise';

GTE_Q: '\\geqq';
GTE_S: '\\geqslant';

BANG: '!';

SINGLE_QUOTES: '\''+;

SYMBOL: '\\' ( [a-zA-Z]+ | '%');

// TODO: see if this can be converted into lexer symbol i.e. LEFT_PARENTHESES, RIGHT_PARENTHESES
left_parentheses: L_PAREN | L_BRACE | L_BRACKET;
right_parentheses: R_PAREN | R_BRACE | R_BRACKET;

math: relation | struct_relation | equation_list;

relation:
	relation (EQUAL | LT | LTE | GT | GTE | NEQ | EQUIV) relation
	| expr;

equation: relation (EQUAL | LT | LTE | GT | GTE | NEQ | EQUIV) relation;

equation_list: equation (SEMI_COLON equation | ',' equation)*;

struct_relation:
    struct_relation (EQUAL) struct_relation
    | struct_expr;

struct_expr:
    struct_expr (SET_ADD | SET_SUB | SET_INTERSECT) struct_expr
    | L_PAREN struct_expr (SET_ADD | SET_SUB | SET_INTERSECT) struct_expr R_PAREN
    | struct_value;

struct_form:
     value ('~' value)*;

struct_value:
    left_parentheses
    value? ('~' value)*
    right_parentheses;

value:
    struct_value
    | relation;

interval_opr:
    L_PAREN | L_BRACKET | R_PAREN | R_BRACKET;

interval:
    interval_opr expr '~' expr interval_opr;

interval_expr:
    interval_expr (SET_ADD | SET_SUB | SET_INTERSECT) (interval_expr | struct_value | atom)
    | L_PAREN interval_expr (SET_ADD | SET_SUB | SET_INTERSECT) (interval_expr | struct_value | atom) R_PAREN
    | interval;

equality: expr EQUAL expr;

expr: set_notation_sub_expr | interval_expr | additive;

additive: additive (ADD | SUB) additive | mp;

// mult part
mp:
	mp (MUL | CMD_TIMES | CMD_CDOT | DIV | CMD_DIV | COLON) mp
	| unary;

mp_nofunc:
	mp_nofunc (
		MUL
		| CMD_TIMES
		| CMD_CDOT
		| DIV
		| CMD_DIV
		| COLON
	) mp_nofunc
	| unary_nofunc;

unary: (ADD | SUB) unary | postfix+;

unary_nofunc:
	(ADD | SUB) unary_nofunc
	| postfix postfix_nofunc*;

postfix: exp postfix_op*;
postfix_nofunc: exp_nofunc postfix_op*;
postfix_op: BANG | eval_at;

eval_at:
	BAR (eval_at_sup | eval_at_sub | eval_at_sup eval_at_sub);

eval_at_sub: UNDERSCORE L_BRACE (expr | equality) R_BRACE;

eval_at_sup: CARET L_BRACE (expr | equality) R_BRACE;

exp: exp CARET (atom | L_BRACE expr R_BRACE) subexpr? | comp;

exp_nofunc:
	exp_nofunc CARET (atom | L_BRACE expr R_BRACE) subexpr?
	| comp_nofunc;

comp:
	group
	| abs_group
	| func
	| atom
	| floor
	| ceil
  | vector;

comp_nofunc:
	group
	| abs_group
	| atom
	| floor
	| ceil;

group:
	L_PAREN expr R_PAREN
	| L_BRACKET expr R_BRACKET
	| L_BRACE expr R_BRACE
	| L_BRACE_LITERAL expr R_BRACE_LITERAL;

abs_group: BAR expr BAR;

number: DIGIT+ (',' DIGIT DIGIT DIGIT)* ('.' DIGIT+)?;

atom: (LETTER | SYMBOL) (subexpr? SINGLE_QUOTES? | SINGLE_QUOTES? subexpr?)
	| number
	| DIFFERENTIAL
	| mathit
	| frac
	| binom
	| bra
	| ket
  | angle;

angle: CMD_ANGLE angle_points;
angle_points: LETTER+ LETTER+ LETTER+ | LETTER+;

bra: L_ANGLE expr (R_BAR | BAR);
ket: (L_BAR | BAR) expr R_ANGLE;

mathit: CMD_MATHIT L_BRACE mathit_text R_BRACE;
mathit_text: LETTER*;

frac: CMD_FRAC (upperd = DIGIT | L_BRACE upper = expr R_BRACE)
    (lowerd = DIGIT | L_BRACE lower = expr R_BRACE);

binom:
	(CMD_BINOM | CMD_DBINOM | CMD_TBINOM) L_BRACE n = expr R_BRACE L_BRACE k = expr R_BRACE;

floor: L_FLOOR val = expr R_FLOOR;
ceil: L_CEIL val = expr R_CEIL;

func_normal:
	FUNC_EXP
	| FUNC_LOG
	| FUNC_LG
	| FUNC_LN
	| FUNC_SIN
	| FUNC_COS
	| FUNC_TAN
	| FUNC_CSC
	| FUNC_SEC
	| FUNC_COT
	| FUNC_ARCSIN
	| FUNC_ARCCOS
	| FUNC_ARCTAN
	| FUNC_ARCCSC
	| FUNC_ARCSEC
	| FUNC_ARCCOT
	| FUNC_SINH
	| FUNC_COSH
	| FUNC_TANH
  | FUNC_SECH
  | FUNC_CSCH
  | FUNC_COTH
  | FUNC_ARCSINH
  | FUNC_ARCCOSH
  | FUNC_ARCTANH
  | FUNC_ARCSECH
  | FUNC_ARCCSCH
  | FUNC_ARCCOTH
  | FUNC_EXP
  | FUNC_ABS
  | FUNC_RE
  | FUNC_IM
  | FUNC_ARG
  | OVERLINE;

func_name:
    LETTER | SYMBOL;

func_composition:
    func_name CMD_CIRCLE (func_name | func_composition);

func:
	func_normal (subexpr? supexpr? | supexpr? subexpr?) (
		L_PAREN func_arg R_PAREN
		| func_arg_noparens
	)
	| (LETTER | SYMBOL) (subexpr? SINGLE_QUOTES? | SINGLE_QUOTES? subexpr?) // e.g. f(x), f_1'(x)
	L_PAREN args R_PAREN
  | L_PAREN func_composition R_PAREN // e.g. (f∘g)(x)
  L_PAREN args R_PAREN
	| FUNC_INT (subexpr supexpr | supexpr subexpr)? (
		additive? DIFFERENTIAL
		| frac
		| additive
	)
    | FUNC_IINT
    (subexpr)?
    (additive? MULTI_DIFFERENTIAL | frac | additive)
    | FUNC_OINT
    (subexpr)?
    (additive? DIFFERENTIAL | frac | additive)
	| FUNC_SQRT (L_BRACKET root = expr R_BRACKET)? L_BRACE base = expr R_BRACE
	// | FUNC_OVERLINE L_BRACE base = expr R_BRACE
  | FUNC_SUM
  (subeq supexpr | subexpr supexpr | supexpr subeq | supexpr subexpr)
  mp
  | FUNC_PROD
  (subeq supexpr | supexpr subeq)
  mp
	| FUNC_LIM limit_sub mp
  | FUNC_MATRIX_START matrix FUNC_MATRIX_END
  | FUNC_MATRIX_DETERMINENT_START matrix FUNC_MATRIX_DETERMINENT_END
  | FUNC_AL_MATRIX_PIECEWISE_START matrix_piecewise FUNC_AL_MATRIX_PIECEWISE_END
  | FUNC_AR_MATRIX_PIECEWISE_START matrix_piecewise FUNC_AR_MATRIX_PIECEWISE_END
  | L_BRACE FUNC_PIECEWISE_START piecewise FUNC_ARRAY_END '.'
  | FUNC_CALCULATION_START calculation FUNC_ARRAY_END
  | FUNC_AL_MATRIX_PIECEWISE_START matrix_relation FUNC_AL_MATRIX_PIECEWISE_END
  | FUNC_AR_MATRIX_PIECEWISE_START matrix_relation FUNC_AR_MATRIX_PIECEWISE_END;

args: (expr '~' args) | expr;

limit_sub:
	UNDERSCORE L_BRACE (LETTER | SYMBOL) LIM_APPROACH_SYM expr (
		CARET ((L_BRACE (ADD | SUB) R_BRACE) | ADD | SUB)
	)? R_BRACE;

set_notation_sub:
    L_BRACE
    (LETTER | SYMBOL)
    BAR
    relation
    R_BRACE;

set_notation_sub_expr:
    set_notation_sub_expr (SET_ADD | SET_SUB | SET_INTERSECT) set_notation_sub_expr
    | L_PAREN set_notation_sub_expr (SET_ADD | SET_SUB | SET_INTERSECT) set_notation_sub_expr R_PAREN
    | set_notation_sub;

matrix_row:
    expr ('&' expr)*;
matrix:
    matrix_row ('\\\\' matrix_row)*;

func_arg: expr | (expr '~' func_arg);
func_arg_noparens: mp_nofunc;

subexpr: UNDERSCORE (atom | L_BRACE expr R_BRACE);
supexpr: CARET (atom | L_BRACE expr R_BRACE);

subeq: UNDERSCORE L_BRACE equality R_BRACE;
supeq: UNDERSCORE L_BRACE equality R_BRACE;

vector: LT func_arg GT;

piecewise_func: expr AMP 'if' relation;
piecewise: piecewise_func ('\\\\' piecewise_func)*;

matrix_piecewise_func: expr | expr (SEMI_COLON | AMP | 'if') relation | expr OTHERWISE;
matrix_piecewise: matrix_piecewise_func ('\\\\' matrix_piecewise_func)*;

calculation_add: number '\\\\' ADD number '\\\\' HLINE number;
calculation_sub: number '\\\\' SUB number '\\\\' HLINE number;
calculation_mul: number '\\\\' (CMD_TIMES | MUL) number '\\\\' HLINE number;
calculation_div: number '\\\\' number L_BRACE OVERLINE L_BRACE SMASH_BIG number R_BRACE R_BRACE;
calculation: calculation_add | calculation_sub | calculation_mul | calculation_div;

matrix_relation: relation ('\\\\' relation)*;
