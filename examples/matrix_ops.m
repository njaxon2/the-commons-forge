% matrix_ops.m  --  Common matrix operations in Octave
% Demonstrates creation, decomposition, and solving linear systems.

% --- Create matrices ---------------------------------------------------
A = [4, 7, 2; 3, 5, 1; 6, 8, 9];
b = [1; 2; 3];

% --- Basic properties --------------------------------------------------
fprintf('Size of A: %d x %d\n', rows(A), columns(A));
fprintf('Determinant: %.4f\n', det(A));
fprintf('Rank: %d\n', rank(A));
fprintf('Trace: %.4f\n', trace(A));

% --- Solve Ax = b ------------------------------------------------------
x = A \ b;
disp('Solution x ='); disp(x);

% --- Eigenvalues and eigenvectors --------------------------------------
[V, D] = eig(A);
eigenvalues = diag(D);
fprintf('Eigenvalues: '); disp(eigenvalues');

% --- Singular Value Decomposition --------------------------------------
[U, S, Vt] = svd(A);
fprintf('Singular values: '); disp(diag(S)');

% --- Matrix operations -------------------------------------------------
I = eye(3);
A_inv = inv(A);
A_t = A';                          % transpose
A_sq = A * A;                      % matrix multiply
A_elem = A .* A;                   % element-wise multiply

% --- Verify inverse ----------------------------------------------------
residual = norm(A * A_inv - I, 'fro');
fprintf('Inverse residual (Frobenius): %e\n', residual);

% --- Cholesky on a symmetric positive-definite matrix ------------------
P = A' * A;                        % guaranteed SPD
L = chol(P, 'lower');
fprintf('Cholesky check norm: %e\n', norm(L * L' - P, 'fro'));
