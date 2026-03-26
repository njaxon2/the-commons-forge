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


print("\nR124 tilde output + array ops:")
check("tilde_sort", "[~, idx] = sort([3 1 2]); idx(1)", "2")
check("tilde_eig", "[~, D] = eig([1 0; 0 2]); D(2,2)", "2")
check("fliplr", "fliplr([1 2 3])", "3    2    1")
check("rot90", "A = [1 2; 3 4]; B = rot90(A); B(1,1)", "2")
check("circshift", "x = circshift([10 20 30], 1); x(1)", "30")
check("squeeze_size", "size(squeeze(ones(1,3,1)))", "1    3")
check("wilkinson", "W = wilkinson(5); W(3,3)", "0")
check("movmean", "x = movmean([1 2 3 4 5], 3); x(3)", "3")
check("sylvester", "A=eye(2); B=2*eye(2); C=3*eye(2); X=sylvester(A,B,C); X(1,1)", "1")
check("sscanf_int", "sscanf('42', '%d')", "42")
check("display_sci", "x = 1e-10; x < 1", "1")
check("display_complex", "z = 3 + 4i; real(z)", "3")


print("\nR128-R129 char indexing + switch strings:")
check("char_index", "s = 'hello'; s(1)", "h")
check("char_range", "s = 'hello'; s(1:3)", "hel")
check("char_end", "s = 'hello'; s(end)", "o")
check("switch_string", """
s = 'b';
switch s
    case 'a'; r = 1;
    case 'b'; r = 2;
    otherwise; r = 0;
end
r
""", "2")
check("struct_dynamic_field", """
st.x = 10; st.y = 20;
f = fieldnames(st);
st.(f{1})
""", "10")
check("anonymous_compose", "f = @(x) x^2; g = @(x) f(x) + 1; g(3)", "10")
check("growing_vector", "v=[]; for i=1:4; v=[v, i^2]; end; v(3)", "9")
check("nested_for_sum", "t=0; for i=1:3; for j=1:3; t=t+1; end; end; t", "9")


print("\nR130 logical indexing assignment:")
check("logical_assign_vec", "x = [1 2 3 4 5]; x(x > 3) = 0; sum(x)", "6")
check("logical_assign_mat", "A = [1 2; 3 4]; A(A > 2) = 0; A(2,1)", "0")
check("logical_assign_scalar", "A = magic(3); A(A > 5) = -1; A(1,1)", "-1")


print("\nR131 IGA Poisson solver from M-code:")
check("derbasisfun", "dN = derbasisfun(2, 0.5, 2, [0 0 0 1 1 1]); length(dN)", "3")
check("gaussQuad_builtin", "[gp, gw] = gaussQuad(3); length(gp)", "3")
check("iga_convergence", """
function err = poisson_err(nel, p)
    ncp = nel + p;
    Xi = zeros(1, ncp + p + 1);
    for i = 1:p+1; Xi(i) = 0; Xi(ncp + i) = 1; end
    for i = 1:nel-1; Xi(p + 1 + i) = i / nel; end
    [gp, gw] = gaussQuad(p + 1);
    K = zeros(ncp); F = zeros(ncp, 1);
    for e = 1:nel
        xi_lo = Xi(e + p); xi_hi = Xi(e + p + 1);
        if xi_hi - xi_lo < 1e-14; continue; end
        Jxi = (xi_hi - xi_lo) / 2;
        for q = 1:length(gp)
            xi = xi_lo + (1 + gp(q)) / 2 * (xi_hi - xi_lo);
            span = findspan(ncp - 1, p, xi, Xi);
            N = basisfun(span, xi, p, Xi);
            dN = derbasisfun(span, xi, p, Xi);
            for a = 1:p+1
                I = span - p + a;
                F(I) = F(I) + N(a) * sin(pi * xi) * Jxi * gw(q);
                for b = 1:p+1
                    J = span - p + b;
                    K(I, J) = K(I, J) + dN(a) * dN(b) * Jxi * gw(q);
                end
            end
        end
    end
    free = 2:ncp-1;
    u = zeros(ncp, 1);
    u(free) = K(free, free) \ F(free);
    err_sq = 0;
    for e = 1:nel
        xi_lo = Xi(e + p); xi_hi = Xi(e + p + 1);
        if xi_hi - xi_lo < 1e-14; continue; end
        Jxi = (xi_hi - xi_lo) / 2;
        for q = 1:length(gp)
            xi = xi_lo + (1 + gp(q)) / 2 * (xi_hi - xi_lo);
            span = findspan(ncp - 1, p, xi, Xi);
            N = basisfun(span, xi, p, Xi);
            uh = 0;
            for a = 1:p+1; I = span - p + a; uh = uh + N(a) * u(I); end
            exact = sin(pi * xi) / (pi^2);
            err_sq = err_sq + (uh - exact)^2 * Jxi * gw(q);
        end
    end
    err = sqrt(err_sq);
end
e4 = poisson_err(4, 2);
e8 = poisson_err(8, 2);
ratio = e4 / e8;
ratio > 7 && ratio < 12
""", "1")


print("\nR134 common functions:")
check("cast_double", "x = cast(int32(5), 'double'); class(x)", "double")
check("mpower_2x2", "A = [1 1; 0 1]; B = mpower(A, 3); B(1,2)", "3")
check("cputime_pos", "cputime() > 0", "1")
check("eps_default", "eps < 1e-10", "1")
check("nextpow2_1000", "nextpow2(1000)", "10")
check("log1p_small", "abs(log1p(1e-15) - 1e-15) < 1e-25", "1")
check("expm1_small", "abs(expm1(1e-15) - 1e-15) < 1e-25", "1")
check("sinc_zero", "sinc(0)", "1")
check("unwrap_phase", "length(unwrap([0 3 6])) == 3", "1")
check("colon_func", "length(colon(1, 5))", "5")
check("colon_step", "length(colon(0, 0.5, 2))", "5")
check("ndgrid_2d", "[X, Y] = ndgrid([1 2], [3 4]); X(1,2)", "1")
check("allclose_true", "allclose([1 2 3], [1 2 3])", "1")
check("allclose_false", "allclose([1 2 3], [1 2 4])", "0")
check("nthroot_neg", "nthroot(-8, 3)", "-2")
check("perms_3", "size(perms([1 2 3]), 1)", "6")
check("nchoosek_scalar", "nchoosek(5, 2)", "10")


print("\nR135 math & special functions (1000+ milestone):")
check("gamma_5", "gamma(5)", "24")
check("gammaln_1", "abs(gammaln(1)) < 1e-15", "1")
check("erf_0", "erf(0)", "0")
check("erfc_0", "erfc(0)", "1")
check("erfinv_rt", "abs(erf(erfinv(0.5)) - 0.5) < 1e-10", "1")
check("besselj_0", "abs(besselj(0, 0) - 1) < 1e-10", "1")
check("gcd_12_8", "gcd(12, 8)", "4")
check("bitget_5_1", "bitget(5, 1)", "1")
check("bitget_5_2", "bitget(5, 2)", "0")
check("bitset_4_1", "bitset(4, 1)", "5")
check("bitshift_left", "bitshift(1, 3)", "8")
check("kron_eye", "K = kron(eye(2), [1 2; 3 4]); K(1,1)", "1")
check("blkdiag_2", "B = blkdiag([1 0; 0 1], [2]); size(B, 1)", "3")
check("pinv_ident", "max(max(abs(pinv(eye(3)) - eye(3)))) < 1e-10", "1")
check("polyfit_line", "p = polyfit([1 2 3], [2 4 6], 1); abs(p(1) - 2) < 1e-10", "1")
check("roots_quad", "r = roots([1 0 -4]); max(abs(sort(r) - [-2; 2])) < 1e-10", "1")
check("fft_len", "length(fft([1 2 3 4]))", "4")
check("trapz_const", "trapz([1 1 1 1])", "3")


print("\nR136 ODE, optimization, integration, time:")
check("fzero_sin", "abs(fzero(@sin, [3, 4]) - pi) < 1e-10", "1")
check("fminbnd_x2", "abs(fminbnd(@(x) x^2, -1, 1)) < 1e-6", "1")
check("integral_sin", "abs(integral(@sin, 0, pi) - 2) < 1e-10", "1")
check("quad_exp", "abs(quad(@exp, 0, 1) - (exp(1) - 1)) < 1e-10", "1")
check("tic_toc", "tic(); x = sum(1:10000); toc() >= 0", "1")
check("clock_len", "length(clock())", "6")
check("eomday_feb", "eomday(2024, 2)", "29")
check("eomday_feb_noleap", "eomday(2023, 2)", "28")
check("is_leap_2024", "is_leap_year(2024)", "1")
check("is_leap_2023", "is_leap_year(2023)", "0")
check("tempdir_nonempty", "length(tempdir()) > 0", "1")
check("lsqnonneg_basic", "x = lsqnonneg(eye(3), [1; 2; 3]); abs(x(1) - 1) < 1e-10", "1")


print("\nR138 data analysis, signal, string, matrix tests:")
check("intmax_int32", "intmax('int32')", "2147483647")
check("intmin_int32", "intmin('int32')", "-2147483648")
check("flintmax_val", "flintmax() == 2^53", "1")
check("realmin_pos", "realmin() > 0", "1")
check("smoothdata_len", "length(smoothdata([1 2 3 4 5 6 7]))", "7")
check("ismissing_nan", "sum(ismissing([1 NaN 3 NaN]))", "2")
check("movmedian_3", "x = movmedian([1 5 2 8 3], 3); x(2)", "2")
check("movmax_3", "x = movmax([1 5 2 8 3], 3); x(4)", "8")
check("downsample_2", "length(downsample(1:10, 2))", "5")
check("upsample_3", "length(upsample([1 2 3], 3))", "9")
check("medfilt1_3", "length(medfilt1([1 5 2 8 3]))", "5")
check("strsplit_basic", "c = strsplit('hello world'); numel(c)", "2")
check("strrep_basic", "strrep('hello world', 'world', 'forge')", "hello forge")
check("regexprep_dig", "regexprep('abc123def', '[0-9]+', 'NUM')", "abcNUMdef")
check("issymmetric_eye", "issymmetric(eye(3))", "1")
check("istriu_eye", "istriu(eye(3))", "1")
check("isdiag_eye", "isdiag(eye(3))", "1")
check("isdefinite_eye", "isdefinite(eye(3))", "1")
check("rcond_eye", "abs(rcond(eye(3)) - 1) < 0.1", "1")
check("sqrtm_4eye", "S = sqrtm(4*eye(2)); abs(S(1,1) - 2) < 1e-10", "1")


print("\nR140 control, image, trig:")
check("lqr_basic", "A = [0 1; 0 -1]; B = [0; 1]; Q = eye(2); R = 1; [K, S, e] = lqr(A, B, Q, R); size(K, 2)", "2")
check("care_basic", "A = [0 1; 0 -1]; B = [0; 1]; Q = eye(2); R = 1; [X, L, G] = care(A, B, Q, R); issymmetric(X)", "1")
check("place_basic", "A = [0 1; 0 0]; B = [0; 1]; K = place(A, B, [-1 -2]); size(K, 2)", "2")
check("cot_pi4", "abs(cot(pi/4) - 1) < 1e-10", "1")
check("sec_0", "abs(sec(0) - 1) < 1e-10", "1")
check("csc_pi2", "abs(csc(pi/2) - 1) < 1e-10", "1")
check("cospi_half", "abs(cospi(0.5)) < 1e-15", "1")
check("sinpi_1", "abs(sinpi(1)) < 1e-15", "1")
check("acot_1", "abs(acot(1) - pi/4) < 1e-10", "1")
check("im2double_scale", "x = im2double(255*ones(2)); abs(x(1,1) - 1) < 1e-10", "1")


print("\nR141 degree trig, strings, type conversion:")
check("cotd_45", "abs(cotd(45) - 1) < 1e-10", "1")
check("secd_0", "abs(secd(0) - 1) < 1e-10", "1")
check("cscd_90", "abs(cscd(90) - 1) < 1e-10", "1")
check("upper_hello", "upper('hello')", "HELLO")
check("lower_HELLO", "lower('HELLO')", "hello")
check("strrep_test", "strrep('abcabc', 'b', 'x')", "axcaxc")
check("double_char", "x = double('A'); x(1)", "65")
check("logical_conv", "x = logical([1 0 1]); sum(x)", "2")


print("\nR142 rref, stats, array extras:")
check("rref_2x3", "R = rref([1 2 3; 4 5 6]); R(1,1)", "1")
check("rref_ident", "R = rref([2 0; 0 3]); R(2,2)", "1")
check("repelem_2", "x = repelem([1 2 3], 1, 2); length(x)", "6")
check("logspace_3", "x = logspace(0, 2, 3); abs(x(2) - 10) < 1e-10", "1")
check("meshgrid_sz", "[X, Y] = meshgrid(1:3, 1:2); size(X, 1)", "2")
check("normpdf_0", "abs(normpdf(0) - 1/sqrt(2*pi)) < 1e-10", "1")
check("normcdf_0", "abs(normcdf(0) - 0.5) < 1e-10", "1")
check("norminv_05", "abs(norminv(0.5)) < 1e-10", "1")
check("vecnorm_2", "abs(vecnorm([3 4]) - 5) < 1e-10", "1")
check("peaks_size", "[X, Y, Z] = peaks(10); size(Z, 1)", "10")


print("\nR144 distributions, file ops:")
check("unifrnd_range", "x = unifrnd(0, 1, 100, 1); min(x) >= 0 && max(x) <= 1", "1")
check("exppdf_0", "abs(exppdf(0, 1) - 1) < 1e-10", "1")
check("expcdf_1", "abs(expcdf(0) - 0) < 1e-10", "1")
check("tcdf_sym", "abs(tcdf(0, 5) - 0.5) < 1e-10", "1")
check("tinv_05", "abs(tinv(0.5, 10)) < 1e-10", "1")
check("chi2cdf_0", "abs(chi2cdf(0, 3)) < 1e-10", "1")
check("isfile_test", "isfile('test_e2e.py')", "1")
check("isfolder_test", "isfolder('forge')", "1")
check("fileparts_test", "[p, n, e] = fileparts('/tmp/test.txt'); e", ".txt")
check("fullfile_test", "fullfile('/tmp', 'test.txt')", "/tmp/test.txt")
check("pwd_nonempty", "length(pwd()) > 0", "1")


print("\nR145 gallery, tolerance, gradient:")
check("gallery_lehmer", "A = gallery('lehmer', 3); A(1,1)", "1")
check("gallery_minij", "A = gallery('minij', 3); A(2,3)", "2")
check("uniquetol_close", "length(uniquetol([1 1.0001 2 2.0001], 0.001))", "2")
check("rescale_01", "x = rescale([0 5 10]); abs(x(3) - 1) < 1e-10", "1")
check("clip_range", "x = clip([-1 0 5 10], 0, 5); x(1) == 0 && x(4) == 5", "1")
check("diff_vec", "x = diff([1 3 6 10]); x(1) == 2 && x(2) == 3 && x(3) == 4", "1")
check("cummax_vec", "x = cummax([1 3 2 5 4]); x(3)", "3")
check("cummin_vec", "x = cummin([5 3 4 1 2]); x(4)", "1")
check("maxk_3", "[v, i] = maxk([5 1 3 2 4], 3); v(1)", "5")
check("mink_2", "[v, i] = mink([5 1 3 2 4], 2); v(1)", "1")
check("hadamard_4", "H = hadamard(4); abs(abs(det(H)) - 16) < 1e-10", "1")


print("\nR146 spline, interp, strings, waveforms, stats:")
check("spline_eval", "x = [0 1 2 3]; y = [0 1 0 1]; yi = spline(x, y, 1.5); abs(yi) < 2", "1")
check("pchip_eval", "x = [0 1 2 3]; y = [0 1 0 1]; yi = pchip(x, y, 1.5); abs(yi) < 2", "1")
check("num2str_int", "num2str(42)", "42")
check("str2num_val", "str2num('3.14') > 3.1", "1")
check("dec2hex_255", "dec2hex(255)", "FF")
check("hex2dec_FF", "hex2dec('FF')", "255")
check("dec2bin_10", "dec2bin(10)", "1010")
check("bin2dec_1010", "bin2dec('1010')", "10")
check("sprintf_fmt", "sprintf('%d + %d = %d', 1, 2, 3)", "1 + 2 = 3")
check("rectpuls_center", "rectpuls(0)", "1")
check("tripuls_center", "tripuls(0)", "1")
check("nanmean_val", "abs(nanmean([1 NaN 3]) - 2) < 1e-10", "1")
check("nansum_val", "nansum([1 NaN 3])", "4")
check("geomean_val", "abs(geomean([1 2 4 8]) - 2.8284) < 0.01", "1")
check("sinint_0", "abs(sinint(0)) < 1e-10", "1")
check("fspecial_avg", "h = fspecial('average', 3); abs(h(1,1) - 1/9) < 1e-10", "1")
check("conv2_ones", "A = ones(3); h = ones(2); B = conv2(A, h, 'same'); B(2,2) == 4", "1")


print("\nR147 plotting, stats, strings, utilities:")
check("randperm_len", "length(randperm(10))", "10")
check("issorted_yes", "issorted([1 2 3 4])", "1")
check("issorted_no", "issorted([3 1 2])", "0")
check("mode_val", "mode([1 2 2 3 3 3])", "3")
check("range_val", "range([1 5 3 8 2])", "7")
check("iqr_val", "abs(iqr([1 2 3 4 5 6 7]) - 3) < 1e-10", "1")
check("zscore_mean0", "x = zscore([1 2 3 4 5]); abs(mean(x)) < 1e-10", "1")
check("detrend_flat", "x = detrend(1:10); abs(mean(x)) < 1e-10", "1")
check("movmean_3", "x = movmean([1 2 3 4 5], 3); abs(x(2) - 2) < 1e-10", "1")
check("accumarray_sum", "x = accumarray([1;1;2;2;3], [10;20;30;40;50]); x(1)", "30")
check("blanks_5", "length(blanks(5))", "5")
check("strtrim_ws", "strtrim('  hello  ')", "hello")
check("contains_yes", "contains('hello world', 'world')", "1")
check("startsWith_yes", "startsWith('hello', 'hel')", "1")
check("endsWith_yes", "endsWith('hello', 'llo')", "1")
check("erase_pat", "length(erase('hello world', 'world'))", "6")
check("replace_str", "replace('hello', 'ell', 'ELL')", "hELLo")
check("count_occ", "count('abcabc', 'abc')", "2")
check("extractBefore_at", "extractBefore('hello@world', '@')", "hello")
check("extractAfter_at", "extractAfter('hello@world', '@')", "world")
check("bitand_val", "bitand(12, 10)", "8")
check("bitor_val", "bitor(12, 10)", "14")
check("bitxor_val", "bitxor(12, 10)", "6")
check("height_mat", "height(ones(3,4))", "3")
check("width_mat", "width(ones(3,4))", "4")
check("gcf_handle", "gcf() >= 1", "1")
check("gca_handle", "gca() >= 1", "1")

print(f"\n=== Results: {passed} passed, {failed} failed ===")
