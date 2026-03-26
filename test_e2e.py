#!/usr/bin/env python3
"""End-to-end test of Forge features."""
import sys
sys.path.insert(0, "/home/ubuntu/forge")
import os
os.environ.setdefault("DISPLAY", ":99")

from forge.engine.session import ForgeSession
s = ForgeSession()
s.eval('addpath("/home/ubuntu/forge/ForgeHome/tiga")')

passed = 0
failed = 0

def check(name, code, expected=None, contains=None):
    global passed, failed
    try:
        r = s.eval(code).strip()
        if expected is not None:
            if r == expected:
                print(f"  PASS: {name}")
                passed += 1
            else:
                print(f"  FAIL: {name}: got '{r}' expected '{expected}'")
                failed += 1
        elif contains is not None:
            if contains in r:
                print(f"  PASS: {name}")
                passed += 1
            else:
                print(f"  FAIL: {name}: '{contains}' not in '{r}'")
                failed += 1
        else:
            print(f"  PASS: {name} => {r[:50]}")
            passed += 1
    except Exception as e:
        print(f"  FAIL: {name}: {e}")
        failed += 1

print("=== Forge E2E Test Suite ===\n")

# --- Math ---
print("Math builtins:")
check("abs", "abs(-5)", "5")
check("sqrt", "sqrt(16)", "4")
check("sin", "sin(0)", "0")
check("cos", "cos(0)", "1")
check("exp", "exp(0)", "1")
check("log", "log(1)", "0")
check("mod", "mod(10,3)", "1")
check("ceil", "ceil(3.2)", "4")
check("floor", "floor(3.8)", "3")
check("round", "round(3.5)", "4")

# --- Linear Algebra ---
print("\nLinear algebra:")
check("eye", "eye(3)", contains="1")
check("zeros", "zeros(2)", contains="0")
check("ones", "ones(2)", contains="1")
check("det", "det([1 0; 0 1])", "1")
check("inv", "inv([2 0; 0 2])", contains="0.5000")
check("eig", "eig([2 0; 0 3])", contains="2")
check("norm", "norm([3 4])", "5")
check("cross", "cross([1 0 0], [0 1 0])", contains="1")
check("dot", "dot([1 2 3], [4 5 6])", "32")

# --- Strings ---
print("\nStrings:")
check("chr", "chr(65)", "A")
check("upper", 'upper("hello")', "HELLO")
check("lower", 'lower("HELLO")', "hello")
check("num2str", "num2str(42)", "42")
check("strcat", 'strcat("a", "b", "c")', "abc")
check("strtrim", 'strtrim("  hi  ")', "hi")
check("sprintf", 'sprintf("x=%d", 42)', "x=42")

# --- Types ---
print("\nType system:")
check("class double", "class(3.14)", "double")
check("class char", 'class("hi")', "char")
check("class logical", "class(true)", "logical")
check("class cell", "class({1,2})", "cell")
check("class struct", 'class(struct("a",1))', "struct")
check("ischar", 'ischar("hello")', "1")
check("isnumeric", "isnumeric(42)", "1")
check("islogical", "islogical(true)", "1")
check("isfield", 'isfield(struct("x",1), "x")', "1")
check("isscalar", "isscalar(42)", "1")
check("isvector", "isvector([1 2 3])", "1")

# --- Control flow ---
print("\nControl flow:")
check("for loop", "x=0; for i=1:5; x=x+i; end; x", "15")
check("while loop", "x=1; while x<10; x=x*2; end; x", "16")
check("if/else", "if 3>2; x=1; else; x=0; end; x", "1")
check("switch", "x=2; switch x; case 1; y=10; case 2; y=20; end; y", "20")

# --- TIGA ---
print("\nTIGA functions:")
check("findspan", "findspan(2, 2, 0.5, [0 0 0 1 1 1])", "2")
check("gaussQuad", "[p,w]=gaussQuad(3); length(p)", "3")
check("basisfun", "basisfun(2, 0.5, 2, [0 0 0 1 1 1])", contains="0")

# --- Plotting ---
print("\nPlotting:")
check("figure", "figure(99);", "")
check("plot", 'plot([1 2 3], [1 4 9]);', "")
check("title", 'title("Test", "FontSize", 14);', "")
check("xlabel", 'xlabel("X");', "")
check("legend", 'legend("data", "Location", "northeast");', "")
check("saveas", 'saveas(99, "/tmp/e2e_test.png");', "")

# --- Workspace ---
print("\nWorkspace:")
check("whos", "whos", contains="double")
check("who", "who", contains="x")

# --- String operations (R90) ---
print("\nString operations (R90):")
check("strrep", 'strrep("hello world", "world", "there")', "hello there")
check("strfind", 'strfind("hello world", "o")', contains="5")
check("regexprep", 'regexprep("abc 123", "\\d+", "NUM")', "abc NUM")

# --- Container utilities (R90) ---
print("\nContainer utilities (R90):")
check("arrayfun", "arrayfun(@(x) x^2, [1 2 3])", contains="4")
check("cat dim1", "cat(1, [1 2], [3 4])", contains="3")
check("num2cell", 'class(num2cell([1 2 3]))', "cell")
check("cell2mat", "cell2mat({1, 2, 3})", contains="2")

# --- File I/O (R94) ---
print("\nFile I/O (R94):")
check("fopen/fclose", 'fid = fopen("/tmp/e2e_io.txt", "w"); fclose(fid)', "0")
check("fileread", 'dlmwrite("/tmp/e2e_data.csv", [1 2; 3 4]); dlmread("/tmp/e2e_data.csv", ",")', contains="3")
check("tempdir", "tempdir", contains="/tmp")



    # === R105-R111 Tests ===
print("\nR105 — eval/feval/strjoin:")
check("eval", 'eval("2+3")', "5")
check("feval", 'feval("sin", pi/2)', "1")
check("strjoin", 'strjoin({"hello", "world"}, "-")', "hello-world")

print("\nR108 — sprintf/class/assert/char concat:")
check("sprintf", 'sprintf("x=%d", 3)', "x=3")
check("class_double", 'class(3.14)', "double")
check("class_char", 'class("hello")', "char")
check("char_concat", '["hello", " ", "world"]', "hello world")
check("assert_true", 'assert(1)', "1")

print("\nR110 — Fast TIGA assembly:")
check("tiga_setup", """
addpath("/home/ubuntu/forge/ForgeHome/tiga");
p = 2; nel = 4;
Xi_r = [0 0 0 0.25 0.5 0.75 1 1 1];
Xi_t = [0 0 0 1 1 1];
n_r = 6; n_t = 3;
r_cp = linspace(0.5, 1.5, n_r);
CPx = zeros(n_r, n_t); CPy = zeros(n_r, n_t); Cw = ones(n_r, n_t);
for i = 1:n_r
    r = r_cp(i);
    CPx(i,1)=r; CPy(i,1)=0;
    CPx(i,2)=r; CPy(i,2)=r;
    CPx(i,3)=0; CPy(i,3)=r;
    Cw(i,2) = 1/sqrt(2);
end
[K,F,free] = tiga_assemble_2d(p, Xi_r, Xi_t, CPx, CPy, Cw);
fprintf("%d %d\\n", size(K,1), length(free));
""", "18 12")

print("\nAdvanced syntax:")
check("switch_case", """
x = 2;
switch x
    case 1; r = "one";
    case 2; r = "two";
    otherwise; r = "other";
end
r
""", "two")
check("try_catch", """
try; error("test"); catch e; r = "caught"; end
r
""", "caught")
check("do_until", """
x = 0; do; x = x + 1; until (x >= 5); x
""", "5")
check("func_handle", """
f = @(x) x^2 + 1;
f(3)
""", "10")
check("nested_func_def", """
function y = sq(x); y = x^2; end
sq(7)
""", "49")


print("\nR113 type predicates:")
check("ischar_true", "ischar('hello')", "1")
check("isnumeric_true", "isnumeric(3.14)", "1")
check("islogical_true", "islogical(true)", "1")
check("ischar_false", "ischar(42)", "0")
check("exist_sin", "exist('sin') > 0", "1")
check("class_double", "class(3.14)", "double")
check("class_char", "class('hello')", "char")
check("fieldnames_struct", """
s.x = 1; s.y = 2;
f = fieldnames(s);
f{1}
""", "x")

print("\nR115 LA decompositions:")
check("schur_basic", """
A = [1 2; 3 4];
[T, U] = schur(A);
err = norm(U*T*U' - A);
err < 1e-10
""", "1")
check("hess_basic", """
A = [1 2 3; 4 5 6; 7 8 9];
[P, H] = hess(A);
err = norm(P*H*P' - A);
err < 1e-10
""", "1")
check("expm_basic", """
A = zeros(2);
B = expm(A);
norm(B - eye(2)) < 1e-10
""", "1")
check("condest_basic", """
A = eye(3);
c = condest(A);
c == 1
""", "1")
check("conv2_basic", """
A = ones(3);
B = ones(2);
C = conv2(A, B, 'valid');
size(C, 1) == 2 && C(1,1) == 4
""", "1")

print("\nR118 help/date/time:")
check("version_str", "version()", contains="Forge")
check("date_format", "d = date(); length(d) > 5", "1")
check("clock_vec", "c = clock(); length(c) == 6", "1")
check("now_positive", "now() > 0", "1")
check("help_basic", "h = help('sin'); ischar(h)", "1")
check("lookfor_basic", "r = lookfor('sin'); ischar(r)", "1")

print("\nString operations:")
check("strcmp_true", "strcmp('abc', 'abc')", "1")
check("strcmp_false", "strcmp('abc', 'xyz')", "0")
check("sprintf_format", "sprintf('%d + %d = %d', 1, 2, 3)", "1 + 2 = 3")
check("char_concat", "['hello', ' ', 'world']", "hello world")

print("\nAdditional math:")
check("mod_op", "mod(10, 3)", "1")
check("rem_op", "rem(10, 3)", "1")
check("sign_neg", "sign(-5)", "-1")
check("fix_pos", "fix(3.7)", "3")
check("fix_neg", "fix(-3.7)", "-3")


print("\nCell assignment (R119 fix):")
check("cell_assign_scalar", "c = cell(1,3); c{1} = 42; c{1}", "42")
check("cell_assign_modify", "c = {1, 2, 3}; c{2} = 99; c{2}", "99")
check("cell_2d_assign", "c = cell(2,3); c{1,2} = 7; c{1,2}", "7")
check("cell_loop_build", "c = cell(1,4); for i=1:4; c{i} = i^2; end; c{3}", "9")

print(f"\n=== Results: {passed} passed, {failed} failed ===")
