# Forge IDE - Function Reference

**Total built-in functions: 702**

## Elementary Functions (27 functions)

| Function | Description |
|----------|-------------|
| `acosd` | Inverse cosine, result in degrees. |
| `acot` | Inverse cotangent. |
| `acotd` | Inverse cotangent, result in degrees. |
| `acoth` | Inverse hyperbolic cotangent. |
| `acsc` | Inverse cosecant. |
| `acscd` | Inverse cosecant, result in degrees. |
| `acsch` | Inverse hyperbolic cosecant. |
| `asec` | Inverse secant. |
| `asecd` | Inverse secant, result in degrees. |
| `asech` | Inverse hyperbolic secant. |
| `asind` | Inverse sine, result in degrees. |
| `atan2d` | Two-argument inverse tangent, result in degrees. |
| `atand` | Inverse tangent, result in degrees. |
| `cosd` | Cosine of argument in degrees. |
| `cospi` | cos(pi * x), exact at half-integer multiples. |
| `cot` | Cotangent. |
| `cotd` | Cotangent of argument in degrees. |
| `coth` | Hyperbolic cotangent. |
| `csc` | Cosecant. |
| `cscd` | Cosecant of argument in degrees. |
| `csch` | Hyperbolic cosecant. |
| `sec` | Secant. |
| `secd` | Secant of argument in degrees. |
| `sech` | Hyperbolic secant. |
| `sind` | Sine of argument in degrees. |
| `sinpi` | sin(pi * x), exact at integer multiples. |
| `tand` | Tangent of argument in degrees. |

## General Math (32 functions)

| Function | Description |
|----------|-------------|
| `accumarray` |  |
| `bincoeff` |  |
| `cart2pol` |  |
| `cart2sph` |  |
| `circshift` |  |
| `cplxpair` |  |
| `cumtrapz` |  |
| `deal` |  |
| `deg2rad` |  |
| `flip` |  |
| `gradient` |  |
| `idivide` |  |
| `int2str` |  |
| `integral` |  |
| `interp1` |  |
| `isequal` |  |
| `isequaln` |  |
| `logspace` |  |
| `nextpow2` |  |
| `pol2cart` |  |
| `polyarea` |  |
| `postpad` |  |
| `prepad` |  |
| `rad2deg` |  |
| `rat` |  |
| `repelem` |  |
| `rescale` |  |
| `shiftdim` |  |
| `sortrows` |  |
| `sph2cart` |  |
| `trapz` |  |
| `xor` |  |

## Special Functions (21 functions)

| Function | Description |
|----------|-------------|
| `beta` |  |
| `betainc` |  |
| `betaincinv` |  |
| `betaln` |  |
| `cosint` |  |
| `ellipke` |  |
| `expint` |  |
| `factor` |  |
| `factorial` |  |
| `gammainc` |  |
| `gammaincinv` |  |
| `isprime` |  |
| `lcm` |  |
| `legendre` |  |
| `nchoosek` |  |
| `nthroot` |  |
| `primes` |  |
| `reallog` |  |
| `realpow` |  |
| `realsqrt` |  |
| `sinint` |  |

## Linear Algebra (29 functions)

| Function | Description |
|----------|-------------|
| `bandwidth` |  |
| `cond` |  |
| `condeig` |  |
| `condest` |  |
| `cross` |  |
| `expm` |  |
| `funm` |  |
| `isbanded` |  |
| `isdefinite` |  |
| `isdiag` |  |
| `ishermitian` |  |
| `issymmetric` |  |
| `istril` |  |
| `istriu` |  |
| `linsolve` |  |
| `logm` |  |
| `lscov` |  |
| `normest` |  |
| `null` |  |
| `ols` |  |
| `orth` |  |
| `planerot` |  |
| `rank` |  |
| `rref` |  |
| `subspace` |  |
| `tensorprod` |  |
| `trace` |  |
| `vech` |  |
| `vecnorm` |  |

## Polynomials & Interpolation (18 functions)

| Function | Description |
|----------|-------------|
| `compan` |  |
| `conv` |  |
| `deconv` |  |
| `mkpp` |  |
| `mpoles` |  |
| `pchip` |  |
| `poly` |  |
| `polyder` |  |
| `polyfit` |  |
| `polyint` |  |
| `polyreduce` |  |
| `polyval` |  |
| `polyvalm` |  |
| `ppval` |  |
| `residue` |  |
| `roots` |  |
| `spline` |  |
| `unmkpp` |  |

## Set Operations (9 functions)

| Function | Description |
|----------|-------------|
| `intersect` | Set intersection of two arrays. |
| `ismember` | Test whether elements of a are members of b. |
| `ismembertol` | Test membership within a tolerance. |
| `powerset` | Power set (all subsets) of the unique elements of x. |
| `setdiff` | Set difference of two arrays (elements in a but not in b). |
| `setxor` | Set exclusive or of two arrays. |
| `union` | Set union of two arrays. |
| `unique` | Unique elements of array, sorted. |
| `uniquetol` | Unique elements within a tolerance. |

## Special Matrices (11 functions)

| Function | Description |
|----------|-------------|
| `gallery` | Generate a named test matrix. |
| `hadamard` | Hadamard matrix of order n (must be a power of 2). |
| `hankel` | Hankel matrix. |
| `hilb` | Hilbert matrix of order n. |
| `invhilb` | Inverse of the Hilbert matrix of order n (exact integer entries). |
| `magic` | Magic square of order n (n >= 3). |
| `pascal` | Pascal matrix of order n. |
| `rosser` | The classic 8x8 Rosser test matrix. |
| `toeplitz` | Toeplitz matrix. |
| `vander` | Vandermonde matrix. |
| `wilkinson` | Wilkinson tridiagonal test matrix of order n. |

## String Functions (32 functions)

| Function | Description |
|----------|-------------|
| `base2dec` | Convert string representation of number in given base to decimal. |
| `bin2dec` | Convert binary string to decimal number. |
| `blanks` | Return a string of N blank characters. |
| `cstrcat` | Concatenate strings without trimming trailing blanks. |
| `deblank` | Remove trailing blanks (spaces and tabs) from string. |
| `dec2base` | Convert decimal integer to string in given base. |
| `dec2bin` | Convert decimal to binary string. |
| `dec2hex` | Convert decimal to hexadecimal string. |
| `endsWith` | Check if string ends with suffix. |
| `erase` | Erase all occurrences of MATCH from S. |
| `hex2dec` | Convert hexadecimal string to decimal number. |
| `index` | Find first occurrence of string T in S (1-based). |
| `isletter` | Return logical array indicating which characters are letters. |
| `isstrprop` | Test character string properties. |
| `mat2str` | Convert matrix to string representation. |
| `native2unicode` | Convert native byte values to Unicode string. |
| `regexptranslate` | Translate string for use in regular expressions. |
| `rindex` | Find last occurrence of string T in S (1-based). |
| `startsWith` | Check if string starts with prefix. |
| `str2num` | Convert string to number. |
| `strcat` | Concatenate strings, trimming trailing blanks from char inputs. |
| `strchr` | Find characters in string, return indices (1-based). |
| `strjoin` | Join cell array of strings with delimiter. |
| `strjust` | Justify string. |
| `strsplit` | Split string at delimiter. |
| `strtok` | Split string at first delimiter token. |
| `strtrim` | Remove leading and trailing whitespace. |
| `strtrunc` | Truncate string to at most N characters. |
| `substr` | Extract substring (1-based offset). |
| `unicode2native` | Convert Unicode string to native byte values. |
| `untabify` | Replace tab characters with spaces. |
| `validatestring` | Validate string against list of valid options. |

## Time & Date (14 functions)

| Function | Description |
|----------|-------------|
| `addtodate` | Add a quantity of time units to a datenum. |
| `asctime` | Return date string from time structure or current time. |
| `calendar` | Return a calendar matrix for a given month. |
| `clock` | Return current date and time as [year month day hour minute second]. |
| `ctime` | Convert Unix timestamp to date string, or return current time string. |
| `date` | Return current date as a string in 'DD-Mon-YYYY' format. |
| `datenum` | Convert date to serial date number (MATLAB compatible). |
| `datestr` | Convert serial date number to date string. |
| `datevec` | Convert serial date number to date vector. |
| `eomday` | Return last day of month for given year and month. |
| `etime` | Elapsed time between two clock vectors (in seconds). |
| `is_leap_year` | Check if year is a leap year. |
| `now` | Return current date/time as a datenum serial date number. |
| `weekday` | Return day of week from datenum. |

## ODE Solvers (9 functions)

| Function | Description |
|----------|-------------|
| `decic` | Compute consistent initial conditions for ode15i. |
| `ode15i` | Solve fully implicit ODE f(t, y, y') = 0. |
| `ode15s` | Solve stiff ODE using implicit Runge-Kutta (Radau IIA) method. |
| `ode23` | Solve non-stiff ODE using Bogacki-Shampine RK(2,3) method. |
| `ode23s` | Solve stiff ODE using low-order implicit method. |
| `ode45` | Solve non-stiff ODE using Dormand-Prince RK(4,5) method. |
| `odeget` | Get value of ODE option from options structure. |
| `odeplot` | Default ODE output function for plotting. |
| `odeset` | Create or modify ODE solver options structure. |

## Optimization (13 functions)

| Function | Description |
|----------|-------------|
| `fminbnd` | Find minimum of single-variable function on bounded interval. |
| `fminsearch` | Find minimum of unconstrained multivariable function (Nelder-Mead). |
| `fminunc` | Find minimum of unconstrained multivariable function (BFGS). |
| `fsolve` | Solve system of nonlinear equations. |
| `fzero` | Find a zero of a univariate function. |
| `glpk` | Solve linear programming problem using scipy. |
| `humps` | Humps test function for optimization and integration. |
| `lsqnonneg` | Linear least squares with nonnegativity constraints. |
| `optimget` | Get optimization option value. |
| `optimset` | Create or modify optimization options structure. |
| `pqpnonneg` | Solve nonneg least squares using an active-set QP approach. |
| `qp` | Solve quadratic programming problem. |
| `sqp` | Solve nonlinear programming problem using SQP. |

## Computational Geometry (14 functions)

| Function | Description |
|----------|-------------|
| `convhull` | Compute convex hull of 2-D or 3-D points. |
| `delaunay` | Delaunay triangulation of 2-D points. |
| `delaunayn` | N-dimensional Delaunay triangulation. |
| `dsearchn` | Nearest-point search using KD-tree. |
| `griddata` | Interpolate scattered data onto grid. |
| `griddatan` | N-dimensional scattered data interpolation. |
| `inpolygon` | Test if points are inside polygon. |
| `rectint` | Compute area of intersection of rectangles. |
| `rotx` | 3x3 rotation matrix about the X-axis. |
| `roty` | 3x3 rotation matrix about the Y-axis. |
| `rotz` | 3x3 rotation matrix about the Z-axis. |
| `tsearchn` | Find enclosing simplex for query points. |
| `voronoi` | Compute Voronoi diagram for 2-D points. |
| `voronoin` | N-dimensional Voronoi diagram. |

## File I/O (33 functions)

| Function | Description |
|----------|-------------|
| `beep` | Produce a beep sound. |
| `bunzip2` | Decompress a bzip2 file. |
| `computer` | Return computer type string. |
| `copyfile` | Copy a file. |
| `csvread` | Read a CSV file into a numeric array. |
| `csvwrite` | Write a numeric array to a CSV file. |
| `delete` | Delete a file. |
| `dir` | List directory contents. |
| `dlmwrite` | Write array to a delimited text file. |
| `fileattrib` | Get file attributes. |
| `fileparts` | Split a file path into directory, name, and extension. |
| `fileread` | Read entire contents of a text file as a character string. |
| `fullfile` | Build full file path from parts. |
| `gunzip` | Decompress a gzip file. |
| `importdata` | Import data from a file, auto-detecting format. |
| `is_valid_file_id` | Test whether fid is a valid file identifier. |
| `isdeployed` | True if running in deployed (compiled) mode. |
| `isfile` | Test whether a path refers to an existing regular file. |
| `isfolder` | Test whether a path refers to an existing directory. |
| `ismac` | True if running on macOS. |
| `ispc` | True if running on a Windows system. |
| `isunix` | True if running on a Unix-like system (Linux, macOS, etc.). |
| `license` | Return license information. |
| `ls` | List directory contents (simple name listing). |
| `memory` | Return memory information as a ForgeStruct. |
| `mkdir` | Create a directory (and parents if needed). |
| `movefile` | Move or rename a file. |
| `tar` | Create a tar archive. |
| `untar` | Extract a tar archive. |
| `unzip` | Extract a zip archive. |
| `ver` | Return version information as a ForgeStruct. |
| `version` | Return version string. |
| `zip` | Create a zip archive. |

## Sparse Matrices (34 functions)

| Function | Description |
|----------|-------------|
| `bicg` | BiConjugate Gradient solver. |
| `bicgstab` | BiConjugate Gradient Stabilized solver. |
| `cgs` | Conjugate Gradient Squared solver. |
| `colperm` | Column ordering to reduce fill-in (approximate minimum degree). |
| `eigs` | Compute *k* largest-magnitude eigenvalues/vectors of sparse A. |
| `etreeplot` | Stub — elimination-tree plot (not yet implemented). |
| `full` | Convert sparse matrix to dense ForgeArray. |
| `gmres` | GMRES solver. |
| `gplot` | Stub — graph plot (not yet implemented). |
| `ichol` | Incomplete Cholesky (via ILU with low drop tolerance). |
| `ilu` | Incomplete LU factorisation. |
| `issparse` | Return True if *x* is a scipy sparse matrix. |
| `nnz` | Number of stored (explicit) nonzeros. |
| `nonzeros` | Return nonzero values as a ForgeArray column. |
| `nzmax` | Maximum number of nonzero entries (same as nnz for scipy). |
| `pcg` | Preconditioned Conjugate Gradient (wraps scipy CG). |
| `pcr` | Preconditioned Conjugate Residual (alias for CG). |
| `qmr` | Quasi-Minimal Residual solver. |
| `sparse` | Create sparse matrix.  1-arg: dense->sparse.  3+ args: COO triplets. |
| `spaugment` | Build augmented matrix [c*I, S; S', 0]. |
| `spconvert` | Build sparse from an Nx3 triplet matrix [i j v]. |
| `spdiags` | Sparse matrix from diagonals. |
| `speye` | Sparse identity matrix. |
| `spfun` | Apply *fun* to every nonzero element. |
| `spones` | Replace nonzero entries with ones. |
| `sprand` | Sparse random matrix (uniform distribution). |
| `sprandn` | Sparse random matrix (normal distribution). |
| `sprandsym` | Sparse random symmetric matrix. |
| `spstats` | Column-wise stats: (count, mean, variance). |
| `spy` | Return (row_indices, col_indices) of nonzero entries. |
| `svds` | Compute *k* largest singular values/vectors of sparse A. |
| `tfqmr` | Transpose-Free Quasi-Minimal Residual solver. |
| `treelayout` | Stub — tree layout (not yet implemented). |
| `treeplot` | Stub — tree plot (not yet implemented). |

## Plotting (51 functions)

| Function | Description |
|----------|-------------|
| `area` |  |
| `axis` | axis([xmin xmax ymin ymax]) or axis('equal'), axis('tight'), etc. |
| `bar` |  |
| `barh` |  |
| `cla` |  |
| `clf` |  |
| `close` |  |
| `colorbar` |  |
| `colormap` | Set the default colormap by name. |
| `contour` |  |
| `contour3` |  |
| `contourf` |  |
| `errorbar` |  |
| `figure` | Create or switch to figure *n*. |
| `fill` |  |
| `gca` |  |
| `gcf` |  |
| `grid` |  |
| `histogram` |  |
| `hold` | Toggle or set hold state. |
| `legend` |  |
| `line` |  |
| `loglog` |  |
| `mesh` |  |
| `meshc` |  |
| `meshz` |  |
| `pie` |  |
| `plot` | plot(y), plot(x,y), plot(x,y,fmt), plot(x1,y1,fmt1, x2,y2,fmt2,...). |
| `plot3` |  |
| `polar` |  |
| `print` | Octave-compatible 'print' command (saves to file). |
| `rectangle` | Draw rectangle at (x, y) with given width and height. |
| `saveas` | Save current figure to *filename*. |
| `scatter` |  |
| `scatter3` |  |
| `semilogx` |  |
| `semilogy` |  |
| `stairs` |  |
| `stem` |  |
| `subplot` |  |
| `surf` |  |
| `surfc` |  |
| `surfl` |  |
| `title` |  |
| `waterfall` |  |
| `xlabel` |  |
| `xlim` |  |
| `ylabel` |  |
| `ylim` |  |
| `zlabel` |  |
| `zlim` |  |

## Signal Processing (64 functions)

| Function | Description |
|----------|-------------|
| `arch_fit` | Fit ARCH(p) model to data using OLS. |
| `arch_rnd` | Simulate ARCH(p) process. |
| `arch_test` | Engle's ARCH test for conditional heteroscedasticity. |
| `arma_rnd` | Simulate ARMA(p,q) process. |
| `autoreg_matrix` | Autoregression matrix. |
| `bartlett` | Bartlett (triangular) window. |
| `besself` | Bessel/Thomson filter design. |
| `blackman` | Blackman window. |
| `butter` | Butterworth filter design. |
| `cheby1` | Chebyshev Type I filter design. |
| `cheby2` | Chebyshev Type II filter design. |
| `cwt` | Continuous wavelet transform. |
| `decimate` | Decrease sampling rate by integer factor. |
| `detrend` | Remove trend from data. |
| `diffpara` | Estimate fractional differencing parameter. |
| `durbinlevinson` | Durbin-Levinson algorithm for AR estimation. |
| `dwt` | Discrete wavelet transform (single level). |
| `ellip` | Elliptic (Cauer) filter design. |
| `fftconv` | FFT-based convolution. |
| `fftfilt` | FFT-based FIR filtering (overlap-add). |
| `fftshift` | Shift zero-frequency component to center. |
| `filter2` | Two-dimensional digital filter. |
| `filtfilt` | Zero-phase digital filtering. |
| `findpeaks` | Find local maxima in a signal. |
| `firwin` | FIR filter design using the window method. |
| `firwin2` | FIR filter design using the window method with arbitrary response. |
| `fractdiff` | Fractionally difference a time series. |
| `freqs` | Analog filter frequency response. |
| `freqz` | Digital filter frequency response. |
| `freqz_plot` | Plot frequency response from freqz output. |
| `grpdelay` | Group delay of a digital filter. |
| `hamming` | Hamming window. |
| `hanning` | Hanning window. |
| `hilbert` | Discrete-time analytic signal via Hilbert transform. |
| `hurst` | Estimate the Hurst exponent of a time series. |
| `idwt` | Inverse discrete wavelet transform (single level). |
| `ifftshift` | Inverse of fftshift. |
| `interp` | Increase sampling rate by integer factor. |
| `kaiser` | Kaiser window. |
| `kaiserord` | Estimate Kaiser window FIR filter order. |
| `lfilter` | Filter data with an IIR or FIR filter. |
| `movfun` | Apply function F over a moving window of length WLEN. |
| `movslice` | Compute start/end indices for a moving window. |
| `periodogram` | Periodogram power spectral density estimate. |
| `remez` | Parks-McClellan optimal FIR filter design. |
| `resample` | Resample signal to NUM samples using polyphase method. |
| `sinc` | Sinc function: sin(pi*x) / (pi*x). |
| `sinetone` | Generate a sine tone. |
| `sinewave` | Generate a sine wave of M samples. |
| `sosfilt` | Filter data with second-order sections. |
| `spectral_adf` | Spectral density from autocovariance via FFT (auto density function). |
| `spectral_xdf` | Cross spectral density from cross-covariance via FFT. |
| `spectrogram` | STFT-based spectrogram. |
| `spencer` | Spencer's 15-point moving average. |
| `ss2tf` | State-space to transfer function form. |
| `stft` | Short-time Fourier transform. |
| `synthesis` | Inverse STFT — reconstruct time signal from STFT output. |
| `tf2ss` | Transfer function to state-space form. |
| `tf2zpk` | Transfer function to zero-pole-gain form. |
| `unwrap` | Unwrap radian phase angles. |
| `xcorr` | Cross-correlation estimate. |
| `xcov` | Cross-covariance (mean-removed cross-correlation). |
| `yulewalker` | Yule-Walker method for AR parameter estimation. |
| `zpk2tf` | Zero-pole-gain to transfer function form. |

## Image Processing (49 functions)

| Function | Description |
|----------|-------------|
| `autumn` | Autumn colormap. MAP = autumn(N) returns an Nx3 array. |
| `bone` | Bone colormap. MAP = bone(N) returns an Nx3 array. |
| `brighten` | Brighten or darken a colormap. |
| `cmpermute` | Reorder a colormap. |
| `cmunique` | Remove duplicate entries from a colormap. |
| `colormap` | Set or get the current colormap. |
| `contrast` | Adjust image contrast. |
| `cool` | Cool colormap. MAP = cool(N) returns an Nx3 array. |
| `copper` | Copper colormap. MAP = copper(N) returns an Nx3 array. |
| `cubehelix` | Cubehelix colormap. MAP = cubehelix(N) returns an Nx3 array. |
| `dither` | Apply Floyd-Steinberg dithering to convert grayscale to binary. |
| `flag` | Flag colormap. MAP = flag(N) returns an Nx3 array. |
| `frame2im` | Convert movie frame to image. |
| `getframe` | Capture axes or figure as movie frame (stub). |
| `gray` | Gray (linear grayscale) colormap. MAP = gray(N) returns an Nx3 array. |
| `gray2ind` | Convert grayscale image to indexed image. |
| `hot` | Hot colormap. MAP = hot(N) returns an Nx3 array. |
| `hsv` | HSV colormap. MAP = hsv(N) returns an Nx3 array. |
| `hsv2rgb` | Convert HSV color values to RGB. |
| `im2double` | Convert image to double precision (float64) in [0,1]. |
| `im2frame` | Convert image to movie frame struct. |
| `image` | Display image object (stub). |
| `imagesc` | Display image with scaled colors (stub). |
| `imfinfo` | Return image file information as a dict. |
| `imformats` | Return or register image file formats. |
| `imread` | Read image from file. |
| `imshow` | Display image (stub — prints dimensions). |
| `imwrite` | Write image to file. |
| `ind2gray` | Convert indexed image to grayscale. |
| `ind2rgb` | Convert indexed image to RGB using colormap. |
| `iscolormap` | Check if input is a valid colormap. |
| `jet` | Jet colormap. MAP = jet(N) returns an Nx3 array. |
| `lines` | Lines colormap (default axes color order, cycled). |
| `movie` | Play movie frames (stub). |
| `ocean` | Ocean colormap. MAP = ocean(N) returns an Nx3 array. |
| `pink` | Pink colormap. MAP = pink(N) returns an Nx3 array. |
| `prism` | Prism colormap. MAP = prism(N) returns an Nx3 array. |
| `rainbow` | Rainbow colormap. MAP = rainbow(N) returns an Nx3 array. |
| `rgb2gray` | Convert RGB image to grayscale. |
| `rgb2hsv` | Convert RGB color values to HSV. |
| `rgb2ind` | Convert RGB image to indexed image. |
| `rgbplot` | Plot the RGB components of a colormap (stub). |
| `spinmap` | Spin the colormap (stub — no GUI). |
| `spring` | Spring colormap. MAP = spring(N) returns an Nx3 array. |
| `summer` | Summer colormap. MAP = summer(N) returns an Nx3 array. |
| `turbo` | Turbo colormap. MAP = turbo(N) returns an Nx3 array. |
| `viridis` | Viridis colormap. MAP = viridis(N) returns an Nx3 array. |
| `white` | White colormap (all ones). MAP = white(N) returns an Nx3 array. |
| `winter` | Winter colormap. MAP = winter(N) returns an Nx3 array. |

## Statistics (49 functions)

| Function | Description |
|----------|-------------|
| `bounds` | Return [min, max] of data. |
| `center` | Center data by subtracting the mean. |
| `corr` | Correlation coefficients. |
| `corrcoef` | Correlation coefficient matrix (alias for corr). |
| `corrcov` | Convert covariance matrix to correlation matrix. |
| `cov` | Covariance matrix. |
| `discrete_cdf` | CDF of a discrete distribution. |
| `discrete_inv` | Inverse CDF (quantile function) of a discrete distribution. |
| `discrete_pdf` | PDF (PMF) of a discrete distribution. |
| `discrete_rnd` | Random samples from a discrete distribution. |
| `empirical_cdf` | Empirical CDF. |
| `empirical_inv` | Empirical inverse CDF (quantile). |
| `empirical_pdf` | Empirical PDF (kernel density estimate). |
| `empirical_rnd` | Random samples from empirical distribution. |
| `histc` | Histogram bin counts (like Octave histc). |
| `iqr` | Interquartile range. |
| `kendall` | Kendall rank correlation coefficient. |
| `kurtosis` | Kurtosis of data. |
| `mad` | Median absolute deviation. |
| `mape` | Mean absolute percentage error. |
| `mean` | Arithmetic mean. |
| `meansq` | Mean squared value. |
| `median` | Median value. |
| `mode` | Most frequent value. |
| `moment` | Central moment of specified order. |
| `movmad` | Moving median absolute deviation. |
| `movmax` | Moving maximum. |
| `movmean` | Moving mean. |
| `movmedian` | Moving median. |
| `movmin` | Moving minimum. |
| `movprod` | Moving product. |
| `movstd` | Moving standard deviation. |
| `movsum` | Moving sum. |
| `movvar` | Moving variance. |
| `normalize` | Normalize data (z-score by default). |
| `prctile` | Percentiles of data. |
| `quantile` | Quantiles of data. |
| `range` | Range of data (max - min). |
| `ranks` | Rank of each element. |
| `rms` | Root mean square. |
| `rmse` | Root mean squared error. |
| `run_count` | Count runs of identical values. |
| `runlength` | Run-length encoding. |
| `skewness` | Skewness of data. |
| `spearman` | Spearman rank correlation coefficient. |
| `statistics` | Compute a vector of common statistics. |
| `std` | Standard deviation. |
| `var` | Variance. |
| `zscore` | Standardized z-scores. |

## Audio (7 functions)

| Function | Description |
|----------|-------------|
| `audioplayer` | Create an audioplayer object (stub). |
| `audiorecorder` | Create an audiorecorder object (stub). |
| `lin2mu` | Convert linear audio signal to mu-law encoding. |
| `mu2lin` | Convert mu-law encoded signal back to linear. |
| `record` | Record audio from microphone (stub). |
| `sound` | Play audio signal through speakers (stub). |
| `soundsc` | Play audio signal scaled to full range (stub). |

## Web & FTP (10 functions)

| Function | Description |
|----------|-------------|
| `ftp_cd` | Change remote directory on FTP connection. |
| `ftp_close` | Close an FTP connection. |
| `ftp_connect` | Connect to an FTP server. |
| `ftp_dir` | List files in current remote directory. |
| `ftp_get` | Download a file from the FTP server. |
| `ftp_put` | Upload a file to the FTP server. |
| `web` | Open URL in system browser (stub) or fetch content. |
| `weboptions` | Create web request options. |
| `webread` | Read content from a web service. |
| `webwrite` | Write data to a web service. |

## Control Systems (50 functions)

| Function | Description |
|----------|-------------|
| `acker` | Ackermann's formula for pole placement (SISO systems only). |
| `append` | Diagonal append of two systems. |
| `bandwidth` | Compute bandwidth (-3 dB frequency). |
| `blkdiag_sys` | Block diagonal combination of multiple systems. |
| `bode` | Bode plot data (magnitude and phase). |
| `connect` | General interconnection of systems. |
| `ctrb` | Compute the controllability matrix. |
| `dcgain` | Compute DC gain (steady-state gain) of a system. |
| `dlqr` | Linear Quadratic Regulator (discrete-time). |
| `dlyap` | Solve discrete Lyapunov equation A*X*A' - X + Q = 0. |
| `evalfr` | Evaluate system frequency response at a complex frequency. |
| `feedback` | Feedback connection. |
| `frd` | Create a frequency response data model. |
| `freqresp` | Compute frequency response. |
| `gram` | Compute Gramian matrix. |
| `impulse` | Impulse response of a system. |
| `initial` | Initial condition response (zero input). |
| `isstable` | Check if a system is stable. |
| `kalman` | Compute Kalman filter gain for continuous-time system. |
| `lqr` | Linear Quadratic Regulator (continuous-time). |
| `lsim` | Simulate system response to arbitrary input. |
| `lyap` | Solve continuous Lyapunov equation A*X + X*A' + Q = 0. |
| `margin` | Compute gain and phase margins. |
| `minreal` | Minimal realization via pole-zero cancellation. |
| `nichols` | Nichols chart data. |
| `nyquist` | Nyquist plot data. |
| `obsv` | Compute the observability matrix. |
| `parallel` | Parallel connection: sys1 + sys2. |
| `pid` | Create a PID controller as a transfer function. |
| `pidtune` | Auto-tune a PID controller using Ziegler-Nichols-like heuristics. |
| `place` | Pole placement for state feedback. |
| `pole` | Compute system poles. |
| `pzmap` | Compute pole-zero map data. |
| `rlocus` | Root locus data. |
| `series` | Cascade (series) connection: sys2 * sys1. |
| `ss` | Create a state-space representation. |
| `ss2tf` | Convert state-space to transfer function representation. |
| `ss2zpk` | Convert state-space to zero-pole-gain representation. |
| `ssdata` | Extract state-space data from a system. |
| `step` | Step response of a system. |
| `stepinfo` | Compute step response characteristics. |
| `tf` | Create a transfer function representation. |
| `tf2ss` | Convert transfer function to state-space representation. |
| `tf2zpk` | Convert transfer function to zero-pole-gain representation. |
| `tfdata` | Extract transfer function data from a system. |
| `zero` | Compute system zeros. |
| `zpk` | Create a zero-pole-gain representation. |
| `zpk2ss` | Convert zero-pole-gain to state-space representation. |
| `zpk2tf` | Convert zero-pole-gain to transfer function representation. |
| `zpkdata` | Extract zero-pole-gain data from a system. |

## Financial (25 functions)

| Function | Description |
|----------|-------------|
| `accrfrac` | Accrued interest fraction = settle_days / coupon_days. |
| `blsdelta` | Black-Scholes delta (call, put). |
| `blsgamma` | Black-Scholes gamma (same for call and put). |
| `blsimpv` | Implied volatility via bisection. |
| `blsprice` | Black-Scholes European call and put price. |
| `blsrho` | Black-Scholes rho (call, put). |
| `blstheta` | Black-Scholes theta (call, put). |
| `blsvega` | Black-Scholes vega. |
| `bndprice` | Clean price of a bond. |
| `bndyield` | Yield to maturity given bond price (Newton-Raphson). |
| `cfamounts` | Cash-flow amounts for a bond (array of coupon + principal at end). |
| `cont2disc` | Continuous to discrete compounding rate. |
| `disc2cont` | Discrete to continuous compounding rate. |
| `frontcon` | Efficient frontier via quadratic optimisation (analytic two-fund). |
| `fvfix` | Future value of fixed periodic payments. |
| `irr` | Internal rate of return (Newton-Raphson). |
| `nper` | Number of periods. |
| `npv` | Net present value of a cash-flow series. |
| `pmt` | Payment per period. |
| `portopt` | Basic mean-variance portfolio optimisation. |
| `portsim` | Simulate portfolio paths via geometric Brownian motion. |
| `pvfix` | Present value of fixed periodic payments. |
| `rate` | Solve for interest rate per period (Newton-Raphson). |
| `ret2tick` | Convert return series to price series (starting at 1). |
| `tick2ret` | Convert price series to return series. |

## Communications (19 functions)

| Function | Description |
|----------|-------------|
| `awgn` | Add white Gaussian noise to a signal. |
| `berawgn` | Theoretical BER for AWGN channel. |
| `biterr` | Bit error count and rate. |
| `convenc` | Convolutional encoder. |
| `eyediagram` | Prepare data for eye diagram (returns 2-D array of overlapped traces). |
| `fskdemod` | FSK demodulation (correlation-based). |
| `fskmod` | FSK modulation. |
| `huffmandeco` | Huffman decode binary data using a dictionary. |
| `huffmandict` | Build a Huffman dictionary. |
| `huffmanenco` | Huffman encode data using a dictionary. |
| `pskdemod` | PSK demodulation (hard decision). |
| `pskmod` | PSK modulation. |
| `qamdemod` | QAM demodulation (hard decision, minimum distance). |
| `qammod` | QAM modulation. |
| `rayleighchan` | Simple flat Rayleigh fading channel. |
| `ricianchan` | Simple flat Rician fading channel. |
| `scatterplot` | Extract I/Q data for scatter plot. |
| `symerr` | Symbol error count and rate. |
| `vitdec` | Viterbi decoder (hard decision). |

## Database (8 functions)

| Function | Description |
|----------|-------------|
| `forge_close_db` | Close a database connection. |
| `forge_database` | Connect to a database. |
| `forge_exec` | Execute a SQL statement (INSERT, UPDATE, DELETE, CREATE, etc.). |
| `forge_fetch` | Fetch query results as a struct (dict of column arrays). |
| `forge_insert` | Insert rows into a table. |
| `forge_sqlread` | Read an entire table (or subset) into a struct. |
| `forge_sqlwrite` | Write data to a table (bulk insert). |
| `forge_update` | Update rows in a table. |

## Parallel Computing (9 functions)

| Function | Description |
|----------|-------------|
| `forge_delete_pool` | Shut down and close a worker pool. |
| `forge_distributed` | Distribute an array across pool workers. |
| `forge_fetchOutputs` | Block until a ForgeFuture completes and return its result. |
| `forge_gather` | Gather a distributed array back into a single array. |
| `forge_gcp` | Get current pool (None if no pool is active). |
| `forge_parfeval` | Asynchronously evaluate a function on a pool worker. |
| `forge_parfor_helper` | Parallel map: apply func to each item in the list. |
| `forge_parpool` | Create a worker pool. |
| `forge_spmd_helper` | Single-Program-Multiple-Data execution. |

## Fuzzy Logic (20 functions)

| Function | Description |
|----------|-------------|
| `defuzz` | Defuzzify a fuzzy output using the specified method. |
| `forge_addmf` | Add a membership function to a variable. |
| `forge_addrule` | Add rules to a FIS. |
| `forge_addvar` | Add an input or output variable to a FIS. |
| `forge_evalfis` | Evaluate a fuzzy inference system. |
| `forge_genfis` | Generate a FIS from data using grid partitioning. |
| `forge_mamfis` | Create a Mamdani fuzzy inference system. |
| `forge_plotmf` | Compute membership function curves for plotting. |
| `forge_readfis` | Parse a FIS from string (.fis format). |
| `forge_showfis` | Return a text summary of a FIS. |
| `forge_sugfis` | Create a Sugeno fuzzy inference system. |
| `forge_writefis` | Serialize a FIS to a string (Octave .fis format compatible). |
| `gaussmf` | Gaussian membership function. |
| `gbellmf` | Generalized bell membership function. |
| `pimf` | Pi-shaped membership function (product of S and Z). |
| `sigmf` | Sigmoidal membership function. |
| `smf` | S-shaped membership function. |
| `trapmf` | Trapezoidal membership function. |
| `trimf` | Triangular membership function. |
| `zmf` | Z-shaped membership function. |

## Neural Networks (18 functions)

| Function | Description |
|----------|-------------|
| `forge_configure` | Auto-configure network from data dimensions. |
| `forge_crossentropy` | Cross-entropy performance metric. |
| `forge_feedforwardnet` | Create a feedforward neural network. |
| `forge_fitnet` | Create a function fitting (regression) network. |
| `forge_getwb` | Get all weights and biases as a single vector. |
| `forge_init_net` | Initialize (or reinitialize) network weights using Xavier/Glorot. |
| `forge_logsig` | Log-sigmoid transfer function. |
| `forge_mse_metric` | Mean squared error performance metric. |
| `forge_net_info` | Return a human-readable summary of the network architecture. |
| `forge_patternnet` | Create a pattern recognition (classification) network. |
| `forge_perform` | Compute network performance (loss) between targets and outputs. |
| `forge_purelin` | Linear transfer function. |
| `forge_relu` | Rectified Linear Unit transfer function. |
| `forge_setwb` | Set all weights and biases from a single vector. |
| `forge_sim` | Simulate (forward pass) the network. |
| `forge_softmax` | Softmax transfer function. |
| `forge_tansig` | Hyperbolic tangent sigmoid transfer function. |
| `forge_train` | Train the network using backpropagation with SGD + momentum. |

## Instrument Control (14 functions)

| Function | Description |
|----------|-------------|
| `forge_fclose_inst` | Close a simulated instrument connection. |
| `forge_fopen_inst` | Open a simulated instrument connection. |
| `forge_fprintf_inst` | Send a command string to a simulated instrument. |
| `forge_fread_inst` | Binary read from a simulated instrument. |
| `forge_fscanf_inst` | Read a response string from a simulated instrument. |
| `forge_fwrite_inst` | Binary write to a simulated instrument. |
| `forge_get_instrument_value` | Read an internal setting from a simulated instrument. |
| `forge_instrfind` | Find available simulated instrument resources. |
| `forge_instrhwinfo` | Return information about available simulated instruments. |
| `forge_query` | Send a query command and read the response (combined write+read). |
| `forge_serial` | Create a simulated serial instrument connection. |
| `forge_set_instrument_value` | Directly set an internal setting on a simulated instrument. |
| `forge_tcpip` | Create a simulated TCP/IP instrument connection. |
| `forge_visa` | Create a simulated VISA instrument connection. |

## Symbolic Math (19 functions)

| Function | Description |
|----------|-------------|
| `collect_sym` | Collect terms with respect to a variable. |
| `diff_sym` | Symbolic differentiation. |
| `double_sym` | Convert symbolic expression to numeric float (or numpy array for Matrix). |
| `dsolve_sym` | Solve an ordinary differential equation. |
| `expand_sym` | Expand products and powers. |
| `factor_sym` | Factor a polynomial expression. |
| `hessian_sym` | Hessian matrix of a scalar expression. |
| `int_sym` | Symbolic integration (indefinite or definite). |
| `jacobian_sym` | Jacobian matrix of vector-valued expression. |
| `latex_sym` | LaTeX representation of a symbolic expression. |
| `limit_sym` | Symbolic limit. |
| `pretty_sym` | Pretty-print a symbolic expression (Unicode text). |
| `simplify_sym` | Simplify a symbolic expression. |
| `solve_sym` | Solve algebraic equation(s). |
| `subs_sym` | Substitute old -> new in expression. |
| `sym` | Create a single symbolic variable. |
| `syms` | Create multiple symbolic variables. |
| `taylor_sym` | Taylor / Maclaurin series expansion. |
| `vpa_sym` | Variable-precision arithmetic evaluation. |

---
*Generated automatically. 708 functions documented.*